//! The option table, part two: options 68 to 161 of `arg_options.zig`'s
//! list, from `--rope-scaling` to `--check`.
//!
//! # Provenance
//!
//! Ports the rest of `common_params_parser_init` (arg.cpp:1391 @c1d0e7a00)
//! from `llama.cpp/common/arg.cpp` (v0.3.0, `c1d0e7a00`), as
//! `arg_options.zig` describes, with the helpers its handlers call:
//! `parse_tensor_buffer_overrides`, `kv_cache_type_from_str`,
//! `get_all_kv_cache_types`, `parse_device_list`,
//! `common_print_available_devices` and `common.h`'s MoE override regexes.
//! Each names what it ports and the line it began at.
//!
//! The multimodal projector's options (`mmproj_examples`), the server's
//! `-np` and `--rpc` -- added only when `llama_supports_rpc()`, never here,
//! though the question is still asked -- are not among them: no tool ported
//! admits them.

const std = @import("std");
const c = @import("tools_h");
const arg = @import("arg.zig");
const params_mod = @import("params.zig");
const log = @import("log.zig");
const io = @import("io.zig");
const options1 = @import("arg_options.zig");
const strings = @import("strings.zig");

const Params = params_mod.Params;
const Arg = arg.Arg;
const Context = arg.Context;
const throw = arg.throw;
const throwStr = arg.throwStr;
const S = [:0]const u8;

extern fn exit(code: c_int) noreturn;

fn tf(b: bool) [*:0]const u8 {
    return if (b) "true" else "false";
}

/// Ports `kv_cache_types` (arg.cpp:304 @c1d0e7a00).
const kv_cache_types = [_]c.ggml_type{
    c.GGML_TYPE_F32,  c.GGML_TYPE_F16,    c.GGML_TYPE_BF16,
    c.GGML_TYPE_Q8_0, c.GGML_TYPE_Q4_0,   c.GGML_TYPE_Q4_1,
    c.GGML_TYPE_IQ4_NL, c.GGML_TYPE_Q5_0, c.GGML_TYPE_Q5_1,
};

/// Ports `kv_cache_type_from_str` (arg.cpp:316 @c1d0e7a00).
fn kvCacheTypeFromStr(s: []const u8) !c.ggml_type {
    for (kv_cache_types) |t| {
        if (std.mem.eql(u8, std.mem.span(c.ggml_type_name(t)), s)) return t;
    }
    var buf: [512]u8 = undefined;
    return throwStr(std.fmt.bufPrint(&buf, "Unsupported cache type: {s}", .{s}) catch "Unsupported cache type");
}

/// Ports `get_all_kv_cache_types` (arg.cpp:325 @c1d0e7a00).
fn allKvCacheTypes(a: std.mem.Allocator) ![:0]const u8 {
    var s: std.ArrayList(u8) = .empty;
    for (kv_cache_types, 0..) |t, i| {
        try s.appendSlice(a, std.mem.span(c.ggml_type_name(t)));
        if (i != kv_cache_types.len - 1) try s.appendSlice(a, ", ");
    }
    return a.dupeSentinel(u8, s.items, 0);
}

/// Ports `parse_device_list` (arg.cpp:1116 @c1d0e7a00).
fn parseDeviceList(p: *Params, value: []const u8) !std.ArrayList(c.ggml_backend_dev_t) {
    var devices: std.ArrayList(c.ggml_backend_dev_t) = .empty;
    const dev_names = try strings.split(p.arena, value, ',');
    if (dev_names.items.len == 0) return throwStr("no devices specified");
    if (dev_names.items.len == 1 and std.mem.eql(u8, dev_names.items[0], "none")) {
        try devices.append(p.arena, null);
    } else {
        c.ggml_backend_load_all();
        for (dev_names.items) |device| {
            const z = try p.arena.dupeSentinel(u8, device, 0);
            const dev = c.ggml_backend_dev_by_name(z.ptr);
            if (dev == null or c.ggml_backend_dev_type(dev) == c.GGML_BACKEND_DEVICE_TYPE_CPU) {
                return throw("invalid device: %s", .{z.ptr});
            }
            try devices.append(p.arena, dev);
        }
        try devices.append(p.arena, null);
    }
    return devices;
}

/// Ports `common_print_available_devices` (arg.cpp:1138 @c1d0e7a00).
pub fn printAvailableDevices(a: std.mem.Allocator) !void {
    const MiB: usize = 1024 * 1024;
    var devices: std.ArrayList(c.ggml_backend_dev_t) = .empty;

    c.ggml_backend_load_all();

    var i: usize = 0;
    while (i < c.ggml_backend_dev_count()) : (i += 1) {
        const dev = c.ggml_backend_dev_get(i);
        if (c.ggml_backend_dev_type(dev) != c.GGML_BACKEND_DEVICE_TYPE_CPU) try devices.append(a, dev);
    }
    io.printf("Available devices:\n", .{});

    if (devices.items.len == 0) {
        io.printf("  (none)\n", .{});
        return;
    }
    for (devices.items) |dev| {
        var free: usize = 0;
        var total: usize = 0;
        c.ggml_backend_dev_memory(dev, &free, &total);
        io.printf("  %s: %s (%zu MiB, %zu MiB free)\n", .{ c.ggml_backend_dev_name(dev), c.ggml_backend_dev_description(dev), total / MiB, free / MiB });
    }
}

/// Ports `parse_tensor_buffer_overrides` (arg.cpp:252 @c1d0e7a00).
fn parseTensorBufferOverrides(p: *Params, value: []const u8) !void {
    c.ggml_backend_load_all();

    // `std::map<std::string, buft>`: by name, sorted
    const Entry = struct { name: []const u8, buft: c.ggml_backend_buffer_type_t };
    var buft_list: std.ArrayList(Entry) = .empty;
    var i: usize = 0;
    while (i < c.ggml_backend_dev_count()) : (i += 1) {
        const dev = c.ggml_backend_dev_get(i);
        const buft = c.ggml_backend_dev_buffer_type(dev);
        if (buft != null) {
            const name = std.mem.span(c.ggml_backend_buft_name(buft));
            var replaced = false;
            for (buft_list.items) |*e| {
                if (std.mem.eql(u8, e.name, name)) {
                    e.buft = buft;
                    replaced = true;
                }
            }
            if (!replaced) try buft_list.append(p.arena, .{ .name = name, .buft = buft });
        }
    }
    std.mem.sort(Entry, buft_list.items, {}, struct {
        fn lt(_: void, x: Entry, y: Entry) bool {
            return std.mem.order(u8, x.name, y.name) == .lt;
        }
    }.lt);

    const overrides = try strings.split(p.arena, value, ',');
    for (overrides.items) |override| {
        const pos = std.mem.indexOfScalar(u8, override, '=') orelse return throwStr("invalid value");
        const tensor_name = override[0..pos];
        const buffer_type = override[pos + 1 ..];

        const found = for (buft_list.items) |e| {
            if (std.mem.eql(u8, e.name, buffer_type)) break e.buft;
        } else null;
        if (found == null) {
            io.printf("Available buffer types:\n", .{});
            for (buft_list.items) |e| io.printf("  %s\n", .{c.ggml_backend_buft_name(e.buft)});
            return throwStr("unknown buffer type");
        }
        try p.tensor_buft_overrides.append(p.arena, .{ .pattern = (try p.arena.dupeSentinel(u8, tensor_name, 0)).ptr, .buft = found });
    }
}

/// Ports `LLM_FFN_EXPS_REGEX` (common/common.h:1114 @c1d0e7a00).
const llm_ffn_exps_regex = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

/// The `[,/]+` split `-ts` and `-fitt` take, as `std::sregex_token_iterator`
/// with submatch -1 gives it: every field before a run of separators, empty
/// ones included, then what follows the last run if it is not empty.
fn splitCommaSlash(a: std.mem.Allocator, s: []const u8) !std.ArrayList([:0]const u8) {
    var out: std.ArrayList([:0]const u8) = .empty;
    var start: usize = 0;
    var i: usize = 0;
    var matched = false;
    while (i < s.len) {
        if (s[i] == ',' or s[i] == '/') {
            try out.append(a, try a.dupeSentinel(u8, s[start..i], 0));
            while (i < s.len and (s[i] == ',' or s[i] == '/')) i += 1;
            start = i;
            matched = true;
        } else {
            i += 1;
        }
    }
    if (!matched or start < s.len) try out.append(a, try a.dupeSentinel(u8, s[start..], 0));
    return out;
}

/// Ports `string_split` (common/common.h:786 @c1d0e7a00): each field as
/// `std::getline` cuts them -- no empty last field -- read by `operator>>`,
/// 0 where it reads nothing.
fn splitInts(a: std.mem.Allocator, s: []const u8) !std.ArrayList(i32) {
    var out: std.ArrayList(i32) = .empty;
    var fields = std.mem.splitScalar(u8, s, ',');
    var consumed: usize = 0;
    while (fields.next()) |field| {
        consumed += field.len + 1;
        if (field.len == 0 and consumed > s.len) break;
        var i: usize = 0;
        while (i < field.len and std.ascii.isWhitespace(field[i])) i += 1;
        const st = i;
        if (i < field.len and (field[i] == '+' or field[i] == '-')) i += 1;
        while (i < field.len and std.ascii.isDigit(field[i])) i += 1;
        try out.append(a, std.fmt.parseInt(i32, field[st..i], 10) catch 0);
    }
    return out;
}

/// The table, part two.
pub fn add(ctx: *Context) !void {
    const p = ctx.params;
    const ex = ctx.ex;
    const kv_types = try allKvCacheTypes(p.arena);

    // arg.cpp:2322
    try ctx.add((Arg{ .args = &.{"--rope-scaling"}, .value_hint = "{none,linear,yarn}", .help = "RoPE frequency scaling method, defaults to linear unless specified by the model", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (std.mem.eql(u8, value, "none")) {
                q.rope_scaling_type = c.LLAMA_ROPE_SCALING_TYPE_NONE;
            } else if (std.mem.eql(u8, value, "linear")) {
                q.rope_scaling_type = c.LLAMA_ROPE_SCALING_TYPE_LINEAR;
            } else if (std.mem.eql(u8, value, "yarn")) {
                q.rope_scaling_type = c.LLAMA_ROPE_SCALING_TYPE_YARN;
            } else return throwStr("invalid value");
        }
    }.f } }).setEnv("LLAMA_ARG_ROPE_SCALING_TYPE"));
    // arg.cpp:2332
    try ctx.add((Arg{ .args = &.{"--rope-scale"}, .value_hint = "N", .help = "RoPE context scaling factor, expands context by a factor of N", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.rope_freq_scale = 1.0 / try arg.stof(value);
        }
    }.f } }).setEnv("LLAMA_ARG_ROPE_SCALE"));
    // arg.cpp:2339
    try ctx.add((Arg{ .args = &.{"--rope-freq-base"}, .value_hint = "N", .help = "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.rope_freq_base = try arg.stof(value);
        }
    }.f } }).setEnv("LLAMA_ARG_ROPE_FREQ_BASE"));
    // arg.cpp:2346
    try ctx.add((Arg{ .args = &.{"--rope-freq-scale"}, .value_hint = "N", .help = "RoPE frequency scaling factor, expands context by a factor of 1/N", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.rope_freq_scale = try arg.stof(value);
        }
    }.f } }).setEnv("LLAMA_ARG_ROPE_FREQ_SCALE"));
    // arg.cpp:2353
    try ctx.add((Arg{ .args = &.{"--yarn-orig-ctx"}, .value_hint = "N", .help = try ctx.fmt("YaRN: original context size of model (default: %d = model training context size)", .{p.yarn_orig_ctx}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.yarn_orig_ctx = value;
        }
    }.f } }).setEnv("LLAMA_ARG_YARN_ORIG_CTX"));
    // arg.cpp:2360
    try ctx.add((Arg{ .args = &.{"--yarn-ext-factor"}, .value_hint = "N", .help = try ctx.fmt("YaRN: extrapolation mix factor (default: %.2f, 0.0 = full interpolation)", .{@as(f64, p.yarn_ext_factor)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.yarn_ext_factor = try arg.stof(value);
        }
    }.f } }).setEnv("LLAMA_ARG_YARN_EXT_FACTOR"));
    // arg.cpp:2367
    try ctx.add((Arg{ .args = &.{"--yarn-attn-factor"}, .value_hint = "N", .help = try ctx.fmt("YaRN: scale sqrt(t) or attention magnitude (default: %.2f)", .{@as(f64, p.yarn_attn_factor)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.yarn_attn_factor = try arg.stof(value);
        }
    }.f } }).setEnv("LLAMA_ARG_YARN_ATTN_FACTOR"));
    // arg.cpp:2374
    try ctx.add((Arg{ .args = &.{"--yarn-beta-slow"}, .value_hint = "N", .help = try ctx.fmt("YaRN: high correction dim or alpha (default: %.2f)", .{@as(f64, p.yarn_beta_slow)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.yarn_beta_slow = try arg.stof(value);
        }
    }.f } }).setEnv("LLAMA_ARG_YARN_BETA_SLOW"));
    // arg.cpp:2381
    try ctx.add((Arg{ .args = &.{"--yarn-beta-fast"}, .value_hint = "N", .help = try ctx.fmt("YaRN: low correction dim or beta (default: %.2f)", .{@as(f64, p.yarn_beta_fast)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.yarn_beta_fast = try arg.stof(value);
        }
    }.f } }).setEnv("LLAMA_ARG_YARN_BETA_FAST"));
    // arg.cpp:2402
    try ctx.add((Arg{ .args = &.{ "-kvo", "--kv-offload" }, .args_neg = &.{ "-nkvo", "--no-kv-offload" }, .help = try ctx.fmt("whether to enable KV cache offloading (default: %s)", .{@as([*:0]const u8, if (p.no_kv_offload) "disabled" else "enabled")}), .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.no_kv_offload = !value;
        }
    }.f } }).setEnv("LLAMA_ARG_KV_OFFLOAD"));
    // arg.cpp:2410
    try ctx.add((Arg{ .args = &.{"--repack"}, .args_neg = &.{ "-nr", "--no-repack" }, .help = try ctx.fmt("whether to enable weight repacking (default: %s)", .{@as([*:0]const u8, if (p.no_extra_bufts) "disabled" else "enabled")}), .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.no_extra_bufts = !value;
        }
    }.f } }).setEnv("LLAMA_ARG_REPACK"));
    // arg.cpp:2418
    try ctx.add((Arg{ .args = &.{"--no-host"}, .help = "bypass host buffer allowing extra buffers to be used", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.no_host = true;
        }
    }.f } }).setEnv("LLAMA_ARG_NO_HOST"));
    // arg.cpp:2425
    try ctx.add((Arg{ .args = &.{ "-ctk", "--cache-type-k" }, .value_hint = "TYPE", .help = try ctx.fmt("KV cache data type for K\nallowed values: %s\n(default: %s)", .{ kv_types.ptr, c.ggml_type_name(p.cache_type_k) }), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.cache_type_k = try kvCacheTypeFromStr(value);
        }
    }.f } }).setEnv("LLAMA_ARG_CACHE_TYPE_K"));
    // arg.cpp:2438
    try ctx.add((Arg{ .args = &.{ "-ctv", "--cache-type-v" }, .value_hint = "TYPE", .help = try ctx.fmt("KV cache data type for V\nallowed values: %s\n(default: %s)", .{ kv_types.ptr, c.ggml_type_name(p.cache_type_v) }), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.cache_type_v = try kvCacheTypeFromStr(value);
        }
    }.f } }).setEnv("LLAMA_ARG_CACHE_TYPE_V"));
    // arg.cpp:2451
    try ctx.add((Arg{ .args = &.{"--hellaswag"}, .help = "compute HellaSwag score over random tasks from datafile supplied with -f", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.hellaswag = true;
        }
    }.f } }).setExamples(&.{.perplexity}));
    // arg.cpp:2458
    try ctx.add((Arg{ .args = &.{"--hellaswag-tasks"}, .value_hint = "N", .help = try ctx.fmt("number of tasks to use when computing the HellaSwag score (default: %zu)", .{p.hellaswag_tasks}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.hellaswag_tasks = @bitCast(@as(isize, value));
        }
    }.f } }).setExamples(&.{.perplexity}));
    // arg.cpp:2465
    try ctx.add((Arg{ .args = &.{"--winogrande"}, .help = "compute Winogrande score over random tasks from datafile supplied with -f", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.winogrande = true;
        }
    }.f } }).setExamples(&.{.perplexity}));
    // arg.cpp:2472
    try ctx.add((Arg{ .args = &.{"--winogrande-tasks"}, .value_hint = "N", .help = try ctx.fmt("number of tasks to use when computing the Winogrande score (default: %zu)", .{p.winogrande_tasks}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.winogrande_tasks = @bitCast(@as(isize, value));
        }
    }.f } }).setExamples(&.{.perplexity}));
    // arg.cpp:2479
    try ctx.add((Arg{ .args = &.{"--multiple-choice"}, .help = "compute multiple choice score over random tasks from datafile supplied with -f", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.multiple_choice = true;
        }
    }.f } }).setExamples(&.{.perplexity}));
    // arg.cpp:2486
    try ctx.add((Arg{ .args = &.{"--multiple-choice-tasks"}, .value_hint = "N", .help = try ctx.fmt("number of tasks to use when computing the multiple choice score (default: %zu)", .{p.multiple_choice_tasks}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.multiple_choice_tasks = @bitCast(@as(isize, value));
        }
    }.f } }).setExamples(&.{.perplexity}));
    // arg.cpp:2493
    try ctx.add((Arg{ .args = &.{"--kl-divergence"}, .help = "computes KL-divergence to logits provided via --kl-divergence-base", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.kl_divergence = true;
        }
    }.f } }).setExamples(&.{.perplexity}));
    // arg.cpp:2500
    try ctx.add((Arg{ .args = &.{ "--save-all-logits", "--kl-divergence-base" }, .value_hint = "FNAME", .help = "set logits file", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.logits_file = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.perplexity}));
    // arg.cpp:2507
    try ctx.add((Arg{ .args = &.{"--ppl-stride"}, .value_hint = "N", .help = try ctx.fmt("stride for perplexity calculation (default: %d)", .{p.ppl_stride}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.ppl_stride = value;
        }
    }.f } }).setExamples(&.{.perplexity}));
    // arg.cpp:2514
    try ctx.add((Arg{ .args = &.{"--ppl-output-type"}, .value_hint = "<0|1>", .help = try ctx.fmt("output type for perplexity calculation (default: %d)", .{p.ppl_output_type}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.ppl_output_type = value;
        }
    }.f } }).setExamples(&.{.perplexity}));
    // arg.cpp:2521
    try ctx.add((Arg{ .args = &.{ "-dt", "--defrag-thold" }, .value_hint = "N", .help = "KV cache defragmentation threshold (DEPRECATED)", .handler = .{ .string = struct {
        fn f(_: *Params, _: S) !void {
            log.wrn("DEPRECATED: --defrag-thold is deprecated and no longer necessary to specify\n", .{});
        }
    }.f } }).setEnv("LLAMA_ARG_DEFRAG_THOLD"));
    // arg.cpp:2543, the arm for every example but the server
    if (ex != .server) {
        try ctx.add((Arg{ .args = &.{ "-np", "--parallel" }, .value_hint = "N", .help = try ctx.fmt("number of parallel sequences to decode (default: %d)", .{p.n_parallel}), .handler = .{ .int = struct {
            fn f(q: *Params, value: c_int) !void {
                q.n_parallel = value;
            }
        }.f } }).setEnv("LLAMA_ARG_N_PARALLEL"));
    }
    // arg.cpp:2647: `--rpc` is added only with an RPC backend, which this
    // build never has -- but the C asks, and asking initialises the backend
    // registry, so every device's start-up log happens here, at the
    // verbosity in force before the command line is read.
    if (c.llama_supports_rpc()) return throwStr("an RPC backend is not supported by this port");
    // arg.cpp:2657
    try ctx.add((Arg{ .args = &.{"--mlock"}, .help = "DEPRECATED in favor of `--load-mode`: force system to keep model in RAM rather than swapping or compressing", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            log.wrn("DEPRECATED: --mlock is deprecated. use --load-mode mlock instead\n", .{});
            q.load_mode = c.LLAMA_LOAD_MODE_MLOCK;
        }
    }.f } }).setEnv("LLAMA_ARG_MLOCK"));
    // arg.cpp:2665
    try ctx.add((Arg{ .args = &.{"--mmap"}, .args_neg = &.{"--no-mmap"}, .help = "DEPRECATED in favor of `--load-mode`: whether to memory-map model. (if mmap disabled, slower load but may reduce pageouts if not using mlock)", .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            log.wrn("DEPRECATED: --mmap and --no-mmap are deprecated. use --load-mode mmap instead\n", .{});
            q.load_mode = if (value) c.LLAMA_LOAD_MODE_MMAP else c.LLAMA_LOAD_MODE_NONE;
        }
    }.f } }).setEnv("LLAMA_ARG_MMAP"));
    // arg.cpp:2674
    try ctx.add((Arg{ .args = &.{ "-dio", "--direct-io" }, .args_neg = &.{ "-ndio", "--no-direct-io" }, .help = "DEPRECATED in favor of `--load-mode`: use DirectIO if available", .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            log.wrn("DEPRECATED: --direct-io and --no-direct-io are deprecated. use --load-mode dio instead\n", .{});
            q.load_mode = if (value) c.LLAMA_LOAD_MODE_DIRECT_IO else c.LLAMA_LOAD_MODE_NONE;
        }
    }.f } }).setEnv("LLAMA_ARG_DIO"));
    // arg.cpp:2683
    try ctx.add((Arg{ .args = &.{ "-lm", "--load-mode" }, .value_hint = "MODE", .help = "model loading mode (default: auto)\n- auto: mmap, unless a device does not support it\n- none: no special loading mode\n- mmap: memory-map model (if mmap disabled, slower load but may reduce pageouts if not using mlock)\n- mlock: force system to keep model in RAM rather than swapping or compressing\n- mmap+mlock: mmap + force system to keep model in RAM rather than swapping or compressing\n- dio: use DirectIO if available\n", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const modes = [_]struct { []const u8, c.llama_load_mode }{
                .{ "auto", c.LLAMA_LOAD_MODE_AUTO },         .{ "none", c.LLAMA_LOAD_MODE_NONE },
                .{ "mmap", c.LLAMA_LOAD_MODE_MMAP },         .{ "mlock", c.LLAMA_LOAD_MODE_MLOCK },
                .{ "mmap+mlock", c.LLAMA_LOAD_MODE_MMAP_MLOCK }, .{ "dio", c.LLAMA_LOAD_MODE_DIRECT_IO },
            };
            for (modes) |m| if (std.mem.eql(u8, value, m[0])) {
                q.load_mode = m[1];
                return;
            };
            return throwStr("invalid value");
        }
    }.f } }).setEnv("LLAMA_ARG_LOAD_MODE"));
    // arg.cpp:2702
    try ctx.add((Arg{ .args = &.{"--numa"}, .value_hint = "TYPE", .help = "attempt optimizations that help on some NUMA systems\n- distribute: spread execution evenly over all nodes\n- isolate: only spawn threads on CPUs on the node that execution started on\n- numactl: use the CPU map provided by numactl\nif run without this previously, it is recommended to drop the system page cache before using this\nsee https://github.com/ggml-org/llama.cpp/issues/1437", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (std.mem.eql(u8, value, "distribute") or value.len == 0) {
                q.numa = c.GGML_NUMA_STRATEGY_DISTRIBUTE;
            } else if (std.mem.eql(u8, value, "isolate")) {
                q.numa = c.GGML_NUMA_STRATEGY_ISOLATE;
            } else if (std.mem.eql(u8, value, "numactl")) {
                q.numa = c.GGML_NUMA_STRATEGY_NUMACTL;
            } else return throwStr("invalid value");
        }
    }.f } }).setEnv("LLAMA_ARG_NUMA"));
    // arg.cpp:2717
    try ctx.add((Arg{ .args = &.{ "-dev", "--device" }, .value_hint = "<dev1,dev2,..>", .help = "comma-separated list of devices to use for offloading (none = don't offload)\nuse --list-devices to see a list of available devices", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.devices = try parseDeviceList(q, value);
        }
    }.f } }).setEnv("LLAMA_ARG_DEVICE"));
    // arg.cpp:2725
    try ctx.add(.{ .args = &.{"--list-devices"}, .help = "print list of available devices and exit", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            try printAvailableDevices(q.arena);
            exit(0);
        }
    }.f } });
    // arg.cpp:2733
    try ctx.add((Arg{ .args = &.{ "-ot", "--override-tensor" }, .value_hint = "<tensor name pattern>=<buffer type>,...", .help = "override tensor buffer type", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            try parseTensorBufferOverrides(q, value);
        }
    }.f } }).setEnv("LLAMA_ARG_OVERRIDE_TENSOR"));
    // arg.cpp:2739
    try ctx.add((Arg{ .args = &.{ "-cmoe", "--cpu-moe" }, .help = "keep all Mixture of Experts (MoE) weights in the CPU", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            // `llm_ffn_exps_cpu_override` (common/common.h:1120 @c1d0e7a00)
            try q.tensor_buft_overrides.append(q.arena, .{ .pattern = llm_ffn_exps_regex, .buft = c.ggml_backend_cpu_buffer_type() });
        }
    }.f } }).setEnv("LLAMA_ARG_CPU_MOE"));
    // arg.cpp:2746
    try ctx.add((Arg{ .args = &.{ "-ncmoe", "--n-cpu-moe" }, .value_hint = "N", .help = "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU", .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            if (value < 0) return throwStr("invalid value");
            var i: c_int = 0;
            while (i < value) : (i += 1) {
                // `llm_ffn_exps_block_regex` (common/common.h:1116 @c1d0e7a00)
                const re = try std.fmt.allocPrintSentinel(q.arena, "blk\\.{d}{s}", .{ i, llm_ffn_exps_regex }, 0);
                try q.tensor_buft_overrides.append(q.arena, .{ .pattern = re.ptr, .buft = c.ggml_backend_cpu_buffer_type() });
            }
        }
    }.f } }).setEnv("LLAMA_ARG_N_CPU_MOE"));
    // arg.cpp:2762
    std.debug.assert(p.n_gpu_layers < 0); // string_format would need to be extended for a default >= 0
    try ctx.add((Arg{ .args = &.{ "-ngl", "--gpu-layers", "--n-gpu-layers" }, .value_hint = "N", .help = try ctx.fmt("max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: %s)", .{@as([*:0]const u8, if (p.n_gpu_layers == -1) "auto" else "all")}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (std.mem.eql(u8, value, "auto")) {
                q.n_gpu_layers = -1;
            } else if (std.mem.eql(u8, value, "all")) {
                q.n_gpu_layers = -2;
            } else {
                q.n_gpu_layers = try arg.stoi(value);
            }
            if (!c.llama_supports_gpu_offload()) {
                io.eprintf("warning: no usable GPU found, --gpu-layers option will be ignored\n", .{});
                io.eprintf("warning: one possible reason is that llama.cpp was compiled without GPU support\n", .{});
                io.eprintf("warning: consult docs/build.md for compilation instructions\n", .{});
            }
        }
    }.f } }).setEnv("LLAMA_ARG_N_GPU_LAYERS"));
    // arg.cpp:2780
    try ctx.add((Arg{ .args = &.{ "-sm", "--split-mode" }, .value_hint = "{none,layer,row,tensor}", .help = "how to split the model across multiple GPUs, one of:\n- none: use one GPU only\n- layer (default): split layers and KV across GPUs (pipelined)\n- row: split weight across GPUs by rows (parallelized)\n- tensor: split weights and KV across GPUs (parallelized, EXPERIMENTAL)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (std.mem.eql(u8, value, "none")) {
                q.split_mode = c.LLAMA_SPLIT_MODE_NONE;
            } else if (std.mem.eql(u8, value, "layer")) {
                q.split_mode = c.LLAMA_SPLIT_MODE_LAYER;
            } else if (std.mem.eql(u8, value, "row")) {
                q.split_mode = c.LLAMA_SPLIT_MODE_ROW;
            } else if (std.mem.eql(u8, value, "tensor")) {
                q.split_mode = c.LLAMA_SPLIT_MODE_TENSOR;
            } else return throwStr("invalid value");
            if (!c.llama_supports_gpu_offload()) {
                io.eprintf("warning: llama.cpp was compiled without support for GPU offload. Setting the split mode has no effect.\n", .{});
            }
        }
    }.f } }).setEnv("LLAMA_ARG_SPLIT_MODE"));
    // arg.cpp:2804
    try ctx.add((Arg{ .args = &.{ "-ts", "--tensor-split" }, .value_hint = "N0,N1,N2,...", .help = "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            // split string by , and /
            const split_arg = try splitCommaSlash(q.arena, value);
            const max = c.llama_max_devices();
            if (split_arg.items.len >= max) return throw("got %zu input configs, but system only has %zu devices", .{ split_arg.items.len, max });
            var i: usize = 0;
            while (i < max) : (i += 1) {
                q.tensor_split[i] = if (i < split_arg.items.len) try arg.stof(split_arg.items[i]) else 0.0;
            }
            if (!c.llama_supports_gpu_offload()) {
                io.eprintf("warning: llama.cpp was compiled without support for GPU offload. Setting a tensor split has no effect.\n", .{});
            }
        }
    }.f } }).setEnv("LLAMA_ARG_TENSOR_SPLIT"));
    // arg.cpp:2831
    try ctx.add((Arg{ .args = &.{ "-mg", "--main-gpu" }, .value_hint = "INDEX", .help = try ctx.fmt("the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with split-mode = row) (default: %d)", .{p.main_gpu}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.main_gpu = value;
            if (!c.llama_supports_gpu_offload()) {
                io.eprintf("warning: llama.cpp was compiled without support for GPU offload. Setting the main GPU has no effect.\n", .{});
            }
        }
    }.f } }).setEnv("LLAMA_ARG_MAIN_GPU"));
    // arg.cpp:2841
    try ctx.add((Arg{ .args = &.{ "-fit", "--fit" }, .value_hint = "[on|off]", .help = try ctx.fmt("whether to adjust unset arguments to fit in device memory ('on' or 'off', default: '%s')", .{@as([*:0]const u8, if (p.fit_params) "on" else "off")}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (arg.isTruthy(value)) {
                q.fit_params = true;
            } else if (arg.isFalsey(value)) {
                q.fit_params = false;
            } else return throw("error: unknown value for --fit: '%s'\n", .{value.ptr});
        }
    }.f } }).setEnv("LLAMA_ARG_FIT"));
    // arg.cpp:2855
    try ctx.add((Arg{ .args = &.{ "-fitp", "--fit-print" }, .value_hint = "[on|off]", .help = try ctx.fmt("print the estimated required memory ('on' or 'off', default: '%s')", .{@as([*:0]const u8, if (p.fit_params_print) "on" else "off")}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (arg.isTruthy(value)) {
                q.fit_params_print = true;
            } else if (arg.isFalsey(value)) {
                q.fit_params_print = false;
            } else return throw("error: unknown value for --fit-print: '%s'\n", .{value.ptr});
        }
    }.f } }).setExamples(&.{.fit_params}).setEnv("LLAMA_ARG_FIT_ESTIMATE"));
    // arg.cpp:2869
    try ctx.add((Arg{ .args = &.{ "-fitt", "--fit-target" }, .value_hint = "MiB0,MiB1,MiB2,...", .help = try ctx.fmt("target margin per device for --fit, comma-separated list of values, single value is broadcast across all devices, default: %zu", .{p.fit_params_target.items[0] / (1024 * 1024)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            // split string by , and /
            const split_arg = try splitCommaSlash(q.arena, value);
            const max = c.llama_max_devices();
            if (split_arg.items.len >= max) return throw("got %zu input configs, but system only has %zu devices", .{ split_arg.items.len, max });
            if (split_arg.items.len == 1) {
                const v = (try stoull(split_arg.items[0])) *% 1024 *% 1024;
                for (q.fit_params_target.items) |*t| t.* = v;
                return;
            }
            for (split_arg.items, 0..) |s, i| q.fit_params_target.items[i] = (try stoull(s)) *% 1024 *% 1024;
        }
    }.f } }).setEnv("LLAMA_ARG_FIT_TARGET"));
    // arg.cpp:2894
    try ctx.add((Arg{ .args = &.{ "-fitc", "--fit-ctx" }, .value_hint = "N", .help = try ctx.fmt("minimum ctx size that can be set by --fit option, default: %u", .{@as(c_uint, @bitCast(p.fit_params_min_ctx))}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.fit_params_min_ctx = value;
        }
    }.f } }).setEnv("LLAMA_ARG_FIT_CTX"));
    // arg.cpp:2901
    try ctx.add(.{ .args = &.{"--check-tensors"}, .help = try ctx.fmt("check model tensor data for invalid values (default: %s)", .{tf(p.check_tensors)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.check_tensors = true;
        }
    }.f } });
    // arg.cpp:2908
    try ctx.add(.{ .args = &.{"--override-kv"}, .value_hint = "KEY=TYPE:VALUE,...", .help = "advanced option to override model metadata by key. to specify multiple overrides, either use comma-separated values.\ntypes: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false,tokenizer.ggml.add_eos_token=bool:false", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const items = try options1.parseCsvRow(q.arena, value);
            for (items.items) |item| {
                if (!try strings.parseKvOverride(item, &q.kv_overrides, q.arena)) {
                    return throw("error: Invalid type for KV override: %s\n", .{item.ptr});
                }
            }
        }
    }.f } });
    // arg.cpp:2920
    try ctx.add(.{ .args = &.{"--op-offload"}, .args_neg = &.{"--no-op-offload"}, .help = try ctx.fmt("whether to offload host tensor operations to device (default: %s)", .{@as([*:0]const u8, if (p.no_op_offload) "false" else "true")}), .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.no_op_offload = !value;
        }
    }.f } });
    // arg.cpp:2928
    try ctx.add((Arg{ .args = &.{"--lora"}, .value_hint = "FNAME", .help = "path to LoRA adapter (use comma-separated values to load multiple adapters)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const items = try options1.parseCsvRow(q.arena, value);
            for (items.items) |item| try q.lora_adapters.append(q.arena, .{ .path = item, .scale = 1.0 });
        }
    }.f } }).setExamples(&.{ .common, .export_lora }));
    // arg.cpp:2938
    try ctx.add((Arg{ .args = &.{"--lora-scaled"}, .value_hint = "FNAME:SCALE,...", .help = "path to LoRA adapter with user defined scaling (format: FNAME:SCALE,...)\nnote: use comma-separated values", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const items = try options1.parseCsvRow(q.arena, value);
            for (items.items) |item| {
                const parts = try strings.split(q.arena, item, ':');
                if (parts.items.len != 2) return throwStr("lora-scaled format: FNAME:SCALE");
                const scale = try arg.stof(try arg.dupe(q, parts.items[1]));
                try q.lora_adapters.append(q.arena, .{ .path = try arg.dupe(q, parts.items[0]), .scale = scale });
            }
        }
    }.f } }).setExamples(&.{ .common, .export_lora }));
    // arg.cpp:2953
    try ctx.add(.{ .args = &.{"--control-vector"}, .value_hint = "FNAME", .help = "add a control vector\nnote: use comma-separated values to add multiple control vectors", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const items = try options1.parseCsvRow(q.arena, value);
            for (items.items) |item| try q.control_vectors.append(q.arena, .{ .strength = 1.0, .fname = item });
        }
    }.f } });
    // arg.cpp:2962
    try ctx.add(.{ .args = &.{"--control-vector-scaled"}, .value_hint = "FNAME:SCALE,...", .help = "add a control vector with user defined scaling SCALE\nnote: use comma-separated values (format: FNAME:SCALE,...)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const items = try options1.parseCsvRow(q.arena, value);
            for (items.items) |item| {
                const parts = try strings.split(q.arena, item, ':');
                if (parts.items.len != 2) return throwStr("control-vector-scaled format: FNAME:SCALE");
                const strength = try arg.stof(try arg.dupe(q, parts.items[1]));
                try q.control_vectors.append(q.arena, .{ .strength = strength, .fname = try arg.dupe(q, parts.items[0]) });
            }
        }
    }.f } });
    // arg.cpp:2976
    try ctx.add(.{ .args = &.{"--control-vector-layer-range"}, .value_hint = "START", .value_hint_2 = "END", .help = "layer range to apply the control vector(s) to, start and end inclusive", .handler = .{ .str_str = struct {
        fn f(q: *Params, start: S, end: S) !void {
            q.control_vector_layer_start = try arg.stoi(start);
            q.control_vector_layer_end = try arg.stoi(end);
        }
    }.f } });
    // arg.cpp:3008
    try ctx.add((Arg{ .args = &.{ "-m", "--model" }, .value_hint = "FNAME", .help = if (ex == .export_lora) "model path from which to load base model" else "model path to load", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.model.path = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{ .common, .export_lora, .download, .tokenize }).setEnv("LLAMA_ARG_MODEL"));
    // arg.cpp:3017
    try ctx.add((Arg{ .args = &.{ "-mu", "--model-url" }, .value_hint = "MODEL_URL", .help = "model download url (default: unused)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.model.url = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{ .common, .download, .tokenize }).setEnv("LLAMA_ARG_MODEL_URL"));
    // arg.cpp:3024
    try ctx.add((Arg{ .args = &.{ "-dr", "--docker-repo" }, .value_hint = "[<repo>/]<model>[:quant]", .help = "Docker Hub model repository. repo is optional, default to ai/. quant is optional, default to :latest.\nexample: gemma3\n(default: unused)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.model.docker_repo = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{ .common, .download, .tokenize }).setEnv("LLAMA_ARG_DOCKER_REPO"));
    // arg.cpp:3033
    try ctx.add((Arg{ .args = &.{ "-hf", "-hfr", "--hf-repo" }, .value_hint = "<user>/<model>[:quant]", .help = "Hugging Face model repository; quant is optional, case-insensitive, default to Q4_K_M, or falls back to the first file in the repo if Q4_K_M doesn't exist.\nmmproj is also downloaded automatically if available. to disable, add --no-mmproj\nexample: ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M\n(default: unused)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.model.hf_repo = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{ .common, .download, .tokenize }).setEnv("LLAMA_ARG_HF_REPO"));
    // arg.cpp:3043
    try ctx.add((Arg{ .args = &.{ "-hff", "--hf-file" }, .value_hint = "FILE", .help = "Hugging Face model file. If specified, it will override the quant in --hf-repo (default: unused)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.model.hf_file = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{ .common, .download, .tokenize }).setEnv("LLAMA_ARG_HF_FILE"));
    // arg.cpp:3050
    try ctx.add((Arg{ .args = &.{ "-hft", "--hf-token" }, .value_hint = "TOKEN", .help = "Hugging Face access token (default: value from HF_TOKEN environment variable)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.hf_token = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{ .common, .download, .tokenize }).setEnv("HF_TOKEN"));
    // arg.cpp:3119
    const out_file_z = try p.arena.dupeSentinel(u8, p.out_file, 0);
    try ctx.add((Arg{ .args = &.{ "-o", "--output", "--output-file" }, .value_hint = "FNAME", .help = try ctx.fmt("output file (default: '%s')", .{out_file_z.ptr}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.out_file = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{ .imatrix, .cvector_generator, .export_lora, .tts, .finetune, .results, .export_graph_ops, .cli }));
    // arg.cpp:3127
    try ctx.add((Arg{ .args = &.{ "-ofreq", "--output-frequency" }, .value_hint = "N", .help = try ctx.fmt("output the imatrix every N iterations (default: %d)", .{p.n_out_freq}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.n_out_freq = value;
        }
    }.f } }).setExamples(&.{.imatrix}));
    // arg.cpp:3134
    try ctx.add((Arg{ .args = &.{"--output-format"}, .value_hint = "{gguf,dat}", .help = try ctx.fmt("output format for imatrix file (default: %s)", .{@as([*:0]const u8, if (p.imat_dat > 0) "dat" else "gguf")}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (std.mem.eql(u8, value, "gguf")) {
                q.imat_dat = -1;
            } else if (std.mem.eql(u8, value, "dat")) {
                q.imat_dat = 1;
            } else return throwStr("invalid output format");
        }
    }.f } }).setExamples(&.{.imatrix}));
    // arg.cpp:3143
    try ctx.add((Arg{ .args = &.{"--save-frequency"}, .value_hint = "N", .help = try ctx.fmt("save an imatrix copy every N iterations (default: %d)", .{p.n_save_freq}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.n_save_freq = value;
        }
    }.f } }).setExamples(&.{.imatrix}));
    // arg.cpp:3150
    try ctx.add((Arg{ .args = &.{"--process-output"}, .help = try ctx.fmt("collect data for the output tensor (default: %s)", .{tf(p.process_output)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.process_output = true;
        }
    }.f } }).setExamples(&.{.imatrix}));
    // arg.cpp:3157
    try ctx.add((Arg{ .args = &.{"--ppl"}, .args_neg = &.{"--no-ppl"}, .help = try ctx.fmt("whether to compute perplexity (default: %s)", .{tf(p.compute_ppl)}), .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.compute_ppl = value;
        }
    }.f } }).setExamples(&.{.imatrix}));
    // arg.cpp:3165
    try ctx.add((Arg{ .args = &.{ "--chunk", "--from-chunk" }, .value_hint = "N", .help = try ctx.fmt("start processing the input from chunk N (default: %d)", .{p.i_chunk}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.i_chunk = value;
        }
    }.f } }).setExamples(&.{.imatrix}));
    // arg.cpp:3172
    try ctx.add((Arg{ .args = &.{"--show-statistics"}, .help = try ctx.fmt("show imatrix statistics and then exit (default: %s)", .{tf(p.show_statistics)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.show_statistics = true;
        }
    }.f } }).setExamples(&.{.imatrix}));
    // arg.cpp:3179
    try ctx.add((Arg{ .args = &.{"--parse-special"}, .help = try ctx.fmt("parse special tokens (chat, tool, etc) (default: %s)", .{tf(p.parse_special)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.parse_special = true;
        }
    }.f } }).setExamples(&.{.imatrix}));
    // arg.cpp:3186
    try ctx.add((Arg{ .args = &.{"--ids"}, .help = try ctx.fmt("only print the token IDs, in a Python-parseable list form like [1, 2, 3] (default: %s)", .{tf(p.tokenize_ids)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.tokenize_ids = true;
        }
    }.f } }).setExamples(&.{.tokenize}));
    // arg.cpp:3193
    try ctx.add((Arg{ .args = &.{"--stdin"}, .help = try ctx.fmt("read the prompt from stdin (takes precedence over -f/--file and -p/--prompt) (default: %s)", .{tf(p.tokenize_stdin)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.tokenize_stdin = true;
        }
    }.f } }).setExamples(&.{.tokenize}));
    // arg.cpp:3200
    try ctx.add((Arg{ .args = &.{"--no-bos"}, .help = try ctx.fmt("do not add a BOS token to the prompt, even if the model normally uses one (default: %s)", .{tf(p.tokenize_no_bos)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.tokenize_no_bos = true;
        }
    }.f } }).setExamples(&.{.tokenize}));
    // arg.cpp:3207
    try ctx.add((Arg{ .args = &.{"--no-parse-special"}, .help = try ctx.fmt("do not parse special tokens (chat, tool, etc) (default: %s)", .{tf(!p.parse_special)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.parse_special = false;
        }
    }.f } }).setExamples(&.{.tokenize}));
    // arg.cpp:3214
    try ctx.add((Arg{ .args = &.{"--show-count"}, .help = try ctx.fmt("print the total number of tokens (default: %s)", .{tf(p.tokenize_show_count)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.tokenize_show_count = true;
        }
    }.f } }).setExamples(&.{.tokenize}));
    // arg.cpp:3221
    try ctx.add((Arg{ .args = &.{"-pps"}, .help = try ctx.fmt("is the prompt shared across parallel sequences (default: %s)", .{tf(p.is_pp_shared)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.is_pp_shared = true;
        }
    }.f } }).setExamples(&.{ .bench, .parallel }));
    // arg.cpp:3228
    try ctx.add((Arg{ .args = &.{"-tgs"}, .help = try ctx.fmt("is the text generation separated across the different sequences (default: %s)", .{tf(p.is_tg_separate)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.is_tg_separate = true;
        }
    }.f } }).setExamples(&.{ .bench, .parallel }));
    // arg.cpp:3235
    try ctx.add((Arg{ .args = &.{"-npp"}, .value_hint = "n0,n1,...", .help = "number of prompt tokens", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const v = try splitInts(q.arena, value);
            try q.n_pp.appendSlice(q.arena, v.items);
        }
    }.f } }).setExamples(&.{.bench}));
    // arg.cpp:3243
    try ctx.add((Arg{ .args = &.{"-ntg"}, .value_hint = "n0,n1,...", .help = "number of text generation tokens", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const v = try splitInts(q.arena, value);
            try q.n_tg.appendSlice(q.arena, v.items);
        }
    }.f } }).setExamples(&.{.bench}));
    // arg.cpp:3251
    try ctx.add((Arg{ .args = &.{"-npl"}, .value_hint = "n0,n1,...", .help = "number of parallel prompts", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const v = try splitInts(q.arena, value);
            try q.n_pl.appendSlice(q.arena, v.items);
        }
    }.f } }).setExamples(&.{.bench}));
    // arg.cpp:3786
    const pos_z = try p.arena.dupeSentinel(u8, p.cvector_positive_file, 0);
    try ctx.add((Arg{ .args = &.{"--positive-file"}, .value_hint = "FNAME", .help = try ctx.fmt("positive prompts file, one prompt per line (default: '%s')", .{pos_z.ptr}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.cvector_positive_file = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.cvector_generator}));
    // arg.cpp:3793
    const neg_z = try p.arena.dupeSentinel(u8, p.cvector_negative_file, 0);
    try ctx.add((Arg{ .args = &.{"--negative-file"}, .value_hint = "FNAME", .help = try ctx.fmt("negative prompts file, one prompt per line (default: '%s')", .{neg_z.ptr}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.cvector_negative_file = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{.cvector_generator}));
    // arg.cpp:3800
    try ctx.add((Arg{ .args = &.{"--pca-batch"}, .value_hint = "N", .help = try ctx.fmt("batch size used for PCA. Larger batch runs faster, but uses more memory (default: %d)", .{p.n_pca_batch}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.n_pca_batch = value;
        }
    }.f } }).setExamples(&.{.cvector_generator}));
    // arg.cpp:3807
    try ctx.add((Arg{ .args = &.{"--pca-iter"}, .value_hint = "N", .help = try ctx.fmt("number of iterations used for PCA (default: %d)", .{p.n_pca_iterations}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.n_pca_iterations = value;
        }
    }.f } }).setExamples(&.{.cvector_generator}));
    // arg.cpp:3814
    try ctx.add((Arg{ .args = &.{"--method"}, .value_hint = "{pca, mean}", .help = "dimensionality reduction method to be used (default: pca)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (std.mem.eql(u8, value, "pca")) {
                q.cvector_dimre_method = .pca;
            } else if (std.mem.eql(u8, value, "mean")) {
                q.cvector_dimre_method = .mean;
            } else return throwStr("invalid value");
        }
    }.f } }).setExamples(&.{.cvector_generator}));
    // arg.cpp:3823
    try ctx.add((Arg{ .args = &.{"--output-format"}, .value_hint = "{md,jsonl}", .help = "output format for batched-bench results (default: md)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (std.mem.eql(u8, value, "jsonl")) {
                q.batched_bench_output_jsonl = true;
            } else if (std.mem.eql(u8, value, "md")) {
                q.batched_bench_output_jsonl = false;
            } else return throwStr("invalid value");
        }
    }.f } }).setExamples(&.{.bench}));
    // arg.cpp:3832
    try ctx.add(.{ .args = &.{"--log-disable"}, .help = "Log disable", .handler = .{ .void_ = struct {
        fn f(_: *Params) !void {
            log.pause();
        }
    }.f } });
    // arg.cpp:3839
    try ctx.add((Arg{ .args = &.{"--log-file"}, .value_hint = "FNAME", .help = "Log to file", .handler = .{ .string = struct {
        fn f(_: *Params, value: S) !void {
            log.setFile(value);
        }
    }.f } }).setEnv("LLAMA_ARG_LOG_FILE"));
    // arg.cpp:3858
    try ctx.add((Arg{ .args = &.{"--log-colors"}, .value_hint = "[on|off|auto]", .help = "Set colored logging ('on', 'off', or 'auto', default: 'auto')\n'auto' enables colors when output is to a terminal", .handler = .{ .string = struct {
        fn f(_: *Params, value: S) !void {
            if (arg.isTruthy(value)) {
                log.setColors(true);
            } else if (arg.isFalsey(value)) {
                log.setColors(false);
            } else if (arg.isAutoy(value)) {
                log.setColors(null);
            } else return throw("error: unknown value for --log-colors: '%s'\n", .{value.ptr});
        }
    }.f } }).setEnv("LLAMA_ARG_LOG_COLORS"));
    // arg.cpp:3875
    try ctx.add(.{ .args = &.{ "-v", "--verbose", "--log-verbose" }, .help = "Set verbosity level to infinity (i.e. log all messages, useful for debugging)", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.verbosity = std.math.maxInt(i32);
            log.verbosity_thold = std.math.maxInt(i32);
        }
    }.f } });
    // arg.cpp:3883
    try ctx.add((Arg{ .args = &.{"--offline"}, .help = "Offline mode: forces use of cache, prevents network access", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.offline = true;
        }
    }.f } }).setExamples(&.{ .common, .download, .tokenize }).setEnv("LLAMA_ARG_OFFLINE"));
    // arg.cpp:3890
    try ctx.add((Arg{ .args = &.{ "-lv", "--verbosity", "--log-verbosity" }, .value_hint = "N", .help = try ctx.fmt("Set the verbosity threshold. Messages with a higher verbosity will be ignored. Values:\n - 0: generic output\n - 1: error\n - 2: warning\n - 3: info\n - 4: trace (more info)\n - 5: debug\n(default: %d)\n", .{p.verbosity}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.verbosity = value;
            log.verbosity_thold = value;
        }
    }.f } }).setEnv("LLAMA_ARG_LOG_VERBOSITY"));
    // arg.cpp:3905
    try ctx.add((Arg{ .args = &.{"--log-prefix"}, .args_neg = &.{"--no-log-prefix"}, .help = "Enable prefix in log messages", .handler = .{ .bool_ = struct {
        fn f(_: *Params, value: bool) !void {
            log.setPrefix(value);
        }
    }.f } }).setEnv("LLAMA_ARG_LOG_PREFIX"));
    // arg.cpp:3913
    try ctx.add((Arg{ .args = &.{"--log-timestamps"}, .args_neg = &.{"--no-log-timestamps"}, .help = "Enable timestamps in log messages", .handler = .{ .bool_ = struct {
        fn f(_: *Params, value: bool) !void {
            log.setTimestamps(value);
        }
    }.f } }).setEnv("LLAMA_ARG_LOG_TIMESTAMPS"));
    // arg.cpp:4041
    try ctx.add((Arg{ .args = &.{ "--spec-draft-type-k", "-ctkd", "--cache-type-k-draft" }, .value_hint = "TYPE", .help = try ctx.fmt("KV cache data type for K for the draft model\nallowed values: %s\n(default: %s)", .{ kv_types.ptr, c.ggml_type_name(p.spec_draft_cache_type_k) }), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.spec_draft_cache_type_k = try kvCacheTypeFromStr(value);
        }
    }.f } }).setEnv("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K"));
    // arg.cpp:4054
    try ctx.add((Arg{ .args = &.{ "--spec-draft-type-v", "-ctvd", "--cache-type-v-draft" }, .value_hint = "TYPE", .help = try ctx.fmt("KV cache data type for V for the draft model\nallowed values: %s\n(default: %s)", .{ kv_types.ptr, c.ggml_type_name(p.spec_draft_cache_type_v) }), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.spec_draft_cache_type_v = try kvCacheTypeFromStr(value);
        }
    }.f } }).setEnv("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V"));
    // arg.cpp:4452
    try ctx.add((Arg{ .args = &.{"--check"}, .help = try ctx.fmt("check rather than generate results (default: %s)", .{tf(p.check)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.check = true;
        }
    }.f } }).setExamples(&.{.results}));
}

/// `std::stoull`.
fn stoull(s: [:0]const u8) !u64 {
    var end: ?[*:0]const u8 = null;
    std.c._errno().* = 0;
    const v = strtoull(s.ptr, &end, 10);
    if (std.c._errno().* == @intFromEnum(std.c.E.RANGE)) return throwStr("stoull: out of range");
    if (end == s.ptr) return throwStr("stoull: no conversion");
    return v;
}
extern fn strtoull(s: [*:0]const u8, end: *?[*:0]const u8, base: c_int) c_ulonglong;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
