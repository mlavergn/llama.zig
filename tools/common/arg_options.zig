//! The option table of `common_params_parser_init`, for the tools ported.
//!
//! # Provenance
//!
//! Ports `common_params_parser_init` (arg.cpp:1344 @cd959bae9) from
//! `llama.cpp/common/arg.cpp` (v0.9.0, `cd959bae9`): every option an example
//! this project ports admits -- `common`'s, and `tokenize`'s, `fit_params`',
//! `export_lora`'s, `bench`'s, `cvector_generator`'s, `imatrix`'s,
//! `perplexity`'s, `results`', `mtmd`'s, `tts`' and `server`'s own -- in the
//! C's order, split across `arg_options.zig` to `arg_options4.zig`, each with its
//! spellings, value hints, help text, examples, environment variable and
//! handler. The help text is built from the tool's current defaults, as the
//! C's `string_format` builds it. Options only other examples take are left
//! out: `Context.add` would drop them for these.
//!
//! Each option names the line of the C it ports in the comment above it.
//!
//! # Refused handlers
//!
//! A few options reach code this port does not have -- the model cache and
//! downloader, the RPC backend, JSON-schema grammars, multimodal projectors.
//! Their help text is ported exactly, so `--help` is upstream's; their
//! handlers throw, saying so, and the parser reports that as it reports any
//! handler's error.

const std = @import("std");
const builtin = @import("builtin");
const c = @import("tools_h");
const arg = @import("arg.zig");
const params_mod = @import("params.zig");
const log = @import("log.zig");
const io = @import("io.zig");
const build_info = @import("build_info.zig");

const Params = params_mod.Params;
const Example = params_mod.Example;
const Arg = arg.Arg;
const Context = arg.Context;
const SamplingConfig = params_mod.SamplingConfig;
const SamplerType = params_mod.SamplerType;
const throw = arg.throw;
const throwStr = arg.throwStr;

extern fn exit(code: c_int) noreturn;

/// `"true"` or `"false"`, as `%s` takes them.
fn tf(b: bool) [*:0]const u8 {
    return if (b) "true" else "false";
}

/// `"enabled"` or `"disabled"`.
fn ed(b: bool) [*:0]const u8 {
    return if (b) "enabled" else "disabled";
}

/// A value hint or a spelling.
const S = [:0]const u8;

/// Ports `read_file` (arg.cpp:43 @cd959bae9).
pub fn readFile(p: *Params, fname: [:0]const u8) ![]const u8 {
    const f = std.c.fopen(fname.ptr, "r") orelse return throw("error: failed to open file '%s'\n", .{fname.ptr});
    defer _ = std.c.fclose(f);
    var content: std.ArrayList(u8) = .empty;
    var buf: [65536]u8 = undefined;
    while (true) {
        const n = std.c.fread(&buf, 1, buf.len, f);
        if (n == 0) break;
        try content.appendSlice(p.arena, buf[0..n]);
    }
    return content.items;
}

/// The body of the `-j` and `-jf` handlers (arg.cpp lines 2281 and 2288): the text as
/// JSON, through `json_schema_to_grammar`, as an output-format grammar. A
/// parse error throws nlohmann's `parse_error` text, a schema outside the
/// subset `std::invalid_argument`'s; both reach the parser as the handler's
/// exception.
fn jsonSchemaGrammar(q: *Params, text: []const u8) !void {
    const json = @import("cli_fit").json;
    const value = json.parse(q.arena, text) catch |e| switch (e) {
        error.JsonError => return throwStr(json.what()),
        error.OutOfMemory => return error.OutOfMemory,
    };
    const grammar = json.schema_to_grammar.jsonSchemaToGrammar(q.arena, &value, false) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return throwStr(json.schema.what()),
    };
    q.sampling.grammar_type = .output_format;
    q.sampling.grammar = grammar;
}

/// Ports `parse_csv_row` (arg.cpp:1300 @cd959bae9).
pub fn parseCsvRow(a: std.mem.Allocator, input: []const u8) !std.ArrayList([:0]const u8) {
    var fields: std.ArrayList([:0]const u8) = .empty;
    var field: std.ArrayList(u8) = .empty;
    var in_quotes = false;

    var i: usize = 0;
    while (i < input.len) : (i += 1) {
        const ch = input[i];
        if (ch == '"') {
            if (!in_quotes) {
                // start of quoted field (only valid if at beginning of field)
                if (field.items.len != 0) {
                    // quote appeared in middle of unquoted field, treat as literal
                    try field.append(a, '"');
                } else {
                    in_quotes = true; // start
                }
            } else {
                if (i + 1 < input.len and input[i + 1] == '"') {
                    // escaped quote: ""
                    try field.append(a, '"');
                    i += 1; // skip the next quote
                } else {
                    in_quotes = false; // end
                }
            }
        } else if (ch == ',') {
            if (in_quotes) {
                try field.append(a, ',');
            } else {
                try fields.append(a, try a.dupeSentinel(u8, field.items, 0));
                field = .empty;
            }
        } else {
            try field.append(a, ch);
        }
    }

    // Add the last field
    try fields.append(a, try a.dupeSentinel(u8, field.items, 0));
    return fields;
}

/// `std::stoull` on part of a range.
fn stoull(a: std.mem.Allocator, s: []const u8) !u64 {
    const z = try a.dupeSentinel(u8, s, 0);
    var end: ?[*:0]const u8 = null;
    std.c._errno().* = 0;
    const v = strtoull(z.ptr, &end, 10);
    if (std.c._errno().* == @backingInt(std.c.E.RANGE)) return throwStr("stoull: out of range");
    if (end == z.ptr) return throwStr("stoull: no conversion");
    return v;
}
extern fn strtoull(s: [*:0]const u8, end: *?[*:0]const u8, base: c_int) c_ulonglong;

/// Ports `parse_cpu_range` (common/common.cpp:333 @cd959bae9).
pub fn parseCpuRange(a: std.mem.Allocator, range: []const u8, boolmask: *[c.GGML_MAX_N_THREADS]bool) !bool {
    const func = "parse_cpu_range";
    const dash_loc = std.mem.indexOfScalar(u8, range, '-') orelse {
        log.comErr(func, "%s", .{@as([*:0]const u8, "Format of CPU range is invalid! Expected [<start>]-[<end>].\n")});
        return false;
    };

    var start_i: u64 = 0;
    var end_i: u64 = 0;

    if (dash_loc != 0) {
        start_i = try stoull(a, range[0..dash_loc]);
        if (start_i >= c.GGML_MAX_N_THREADS) {
            log.comErr(func, "%s", .{@as([*:0]const u8, "Start index out of bounds!\n")});
            return false;
        }
    }

    if (dash_loc == range.len - 1) {
        end_i = c.GGML_MAX_N_THREADS - 1;
    } else {
        end_i = try stoull(a, range[dash_loc + 1 ..]);
        if (end_i >= c.GGML_MAX_N_THREADS) {
            log.comErr(func, "%s", .{@as([*:0]const u8, "End index out of bounds!\n")});
            return false;
        }
    }

    var i = start_i;
    while (i <= end_i) : (i += 1) boolmask[@intCast(i)] = true;
    return true;
}

/// Ports `parse_cpu_mask` (common/common.cpp:370 @cd959bae9).
pub fn parseCpuMask(mask: []const u8, boolmask: *[c.GGML_MAX_N_THREADS]bool) bool {
    // Discard potential 0x prefix
    var start_i: usize = 0;
    if (mask.len >= 2 and std.mem.eql(u8, mask[0..2], "0x")) start_i = 2;

    var num_digits = mask.len - start_i;
    num_digits = @min(num_digits, 128);

    const end_i = num_digits + start_i;

    var i = start_i;
    var n: usize = num_digits *% 4 -% 1;
    while (i < end_i) : ({
        i += 1;
        n -%= 4;
    }) {
        const ch = mask[i];
        var id: i8 = @bitCast(ch);
        if (ch >= '0' and ch <= '9') {
            id -= '0';
        } else if (ch >= 'a' and ch <= 'f') {
            id -= 'a' - 10;
        } else if (ch >= 'A' and ch <= 'F') {
            id -= 'A' - 10;
        } else {
            log.comErr("parse_cpu_mask", "Invalid hex character '%c' at position %d\n", .{ @as(c_int, ch), @as(i32, @intCast(i)) });
            return false;
        }

        boolmask[n] = boolmask[n] or ((id & 8) != 0);
        boolmask[n - 1] = boolmask[n - 1] or ((id & 4) != 0);
        boolmask[n - 2] = boolmask[n - 2] or ((id & 2) != 0);
        boolmask[n - 3] = boolmask[n - 3] or ((id & 1) != 0);
    }

    return true;
}

/// `common_sampler_types_from_names`, ported in `sampling.zig`, into the
/// parameters' arena.
fn samplerTypesFromNames(p: *Params, names: []const []const u8) !std.ArrayList(SamplerType) {
    return std.ArrayList(SamplerType).fromOwnedSlice(try @import("sampling.zig").typesFromNames(p.arena, names));
}

/// `samplerTypesFromNames`, for `common_init_sampler_from_model`.
pub const samplerTypesFromNamesPub = samplerTypesFromNames;

/// `common_sampler_types_from_chars`, ported in `sampling.zig`, into the
/// parameters' arena.
fn samplerTypesFromChars(p: *Params, chars: []const u8) !std.ArrayList(SamplerType) {
    return std.ArrayList(SamplerType).fromOwnedSlice(try @import("sampling.zig").typesFromChars(p.arena, chars));
}

/// `params.sampling.dry_sequence_breakers` as option 54's help lists them:
/// each quoted, a newline shown as `\n`.
fn dryBreakersText(p: *Params) ![]const u8 {
    const br = p.sampling.dry_sequence_breakers.items;
    if (br.len == 0) return "none";
    var s: std.ArrayList(u8) = .empty;
    for (br, 0..) |b, i| {
        try s.appendSlice(p.arena, if (i == 0) "'" else ", '");
        try s.appendSlice(p.arena, if (std.mem.eql(u8, b, "\n")) "\\n" else b);
        try s.append(p.arena, '\'');
    }
    return s.items;
}

/// Option 54's `static bool defaults_cleared`, once per process.
var dry_defaults_cleared = false;

/// The table, part one: `-h` to `--pooling`, arg.cpp's lines 1448 to 2317.
fn addPart1(ctx: *Context) !void {
    const p = ctx.params;
    const ex = ctx.ex;

    var sampler_type_chars: std.ArrayList(u8) = .empty;
    var sampler_type_names: std.ArrayList(u8) = .empty;
    for (p.sampling.samplers.items) |sampler| {
        try sampler_type_chars.append(p.arena, sampler.toChr());
        try sampler_type_names.appendSlice(p.arena, sampler.toStr());
        try sampler_type_names.append(p.arena, ';');
    }
    if (sampler_type_names.items.len != 0) _ = sampler_type_names.pop(); // remove last semicolon
    try sampler_type_chars.append(p.arena, 0);
    try sampler_type_names.append(p.arena, 0);
    const chars_z: [*:0]const u8 = @ptrCast(sampler_type_chars.items.ptr);
    const names_z: [*:0]const u8 = @ptrCast(sampler_type_names.items.ptr);

    // arg.cpp:1454
    try ctx.add((Arg{ .args = &.{ "-h", "--help", "--usage" }, .help = "print usage and exit", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.usage = true;
        }
    }.f } }).setExamples(&.{ .common, .download }));
    // arg.cpp:1461
    try ctx.add(.{ .args = &.{"--version"}, .help = "show version and build info", .handler = .{ .void_ = struct {
        fn f(_: *Params) !void {
            build_info.print(c.llama_version(), io.cStderr());
            exit(0);
        }
    }.f } });
    // arg.cpp:1469
    try ctx.add(.{ .args = &.{ "-cl", "--cache-list" }, .help = "show list of models in cache", .handler = .{ .void_ = struct {
        fn f(_: *Params) !void {
            return throwStr("the model cache is not supported by this port");
        }
    }.f } });
    // arg.cpp:1481
    try ctx.add(.{ .args = &.{"--completion-bash"}, .help = "print source-able bash completion script for llama.cpp", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.completion = true;
        }
    }.f } });
    // arg.cpp:1527
    try ctx.add((Arg{ .args = &.{ "-t", "--threads" }, .value_hint = "N", .help = try ctx.fmt("number of CPU threads to use during generation (default: %d)", .{p.cpuparams.n_threads}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.cpuparams.n_threads = value;
            if (q.cpuparams.n_threads <= 0) q.cpuparams.n_threads = @intCast(std.Thread.getCpuCount() catch 0);
        }
    }.f } }).setEnv("LLAMA_ARG_THREADS"));
    // arg.cpp:1537
    try ctx.add(.{ .args = &.{ "-tb", "--threads-batch" }, .value_hint = "N", .help = "number of threads to use during batch and prompt processing (default: same as --threads)", .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.cpuparams_batch.n_threads = value;
            if (q.cpuparams_batch.n_threads <= 0) q.cpuparams_batch.n_threads = @intCast(std.Thread.getCpuCount() catch 0);
        }
    }.f } });
    // arg.cpp:1547
    try ctx.add(.{ .args = &.{ "-C", "--cpu-mask" }, .value_hint = "M", .help = "CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")", .handler = .{ .string = struct {
        fn f(q: *Params, mask: S) !void {
            q.cpuparams.mask_valid = true;
            if (!parseCpuMask(mask, &q.cpuparams.cpumask)) return throwStr("invalid cpumask");
        }
    }.f } });
    // arg.cpp:1557
    try ctx.add(.{ .args = &.{ "-Cr", "--cpu-range" }, .value_hint = "lo-hi", .help = "range of CPUs for affinity. Complements --cpu-mask", .handler = .{ .string = struct {
        fn f(q: *Params, range: S) !void {
            q.cpuparams.mask_valid = true;
            if (!try parseCpuRange(q.arena, range, &q.cpuparams.cpumask)) return throwStr("invalid range");
        }
    }.f } });
    // arg.cpp:1567
    try ctx.add(.{ .args = &.{"--cpu-strict"}, .value_hint = "<0|1>", .help = try ctx.fmt("use strict CPU placement (default: %u)\n", .{@as(c_uint, @intFromBool(p.cpuparams.strict_cpu))}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.cpuparams.strict_cpu = try arg.stoul(value) != 0;
        }
    }.f } });
    // arg.cpp:1574
    try ctx.add(.{ .args = &.{"--prio"}, .value_hint = "N", .help = try ctx.fmt("set process/thread priority : low(-1), normal(0), medium(1), high(2), realtime(3) (default: %d)\n", .{@as(c_int, @intCast(p.cpuparams.priority))}), .handler = .{ .int = struct {
        fn f(q: *Params, prio: c_int) !void {
            if (prio < c.GGML_SCHED_PRIO_LOW or prio > c.GGML_SCHED_PRIO_REALTIME) return throwStr("invalid value");
            q.cpuparams.priority = @intCast(prio);
        }
    }.f } });
    // arg.cpp:1584
    try ctx.add(.{ .args = &.{"--poll"}, .value_hint = "<0...100>", .help = try ctx.fmt("use polling level to wait for work (0 - no polling, default: %u)\n", .{@as(c_uint, p.cpuparams.poll)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.cpuparams.poll = @truncate(try arg.stoul(value));
        }
    }.f } });
    // arg.cpp:1591
    try ctx.add(.{ .args = &.{ "-Cb", "--cpu-mask-batch" }, .value_hint = "M", .help = "CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)", .handler = .{ .string = struct {
        fn f(q: *Params, mask: S) !void {
            q.cpuparams_batch.mask_valid = true;
            if (!parseCpuMask(mask, &q.cpuparams_batch.cpumask)) return throwStr("invalid cpumask");
        }
    }.f } });
    // arg.cpp:1601
    try ctx.add(.{ .args = &.{ "-Crb", "--cpu-range-batch" }, .value_hint = "lo-hi", .help = "ranges of CPUs for affinity. Complements --cpu-mask-batch", .handler = .{ .string = struct {
        fn f(q: *Params, range: S) !void {
            q.cpuparams_batch.mask_valid = true;
            if (!try parseCpuRange(q.arena, range, &q.cpuparams_batch.cpumask)) return throwStr("invalid range");
        }
    }.f } });
    // arg.cpp:1611
    try ctx.add(.{ .args = &.{"--cpu-strict-batch"}, .value_hint = "<0|1>", .help = "use strict CPU placement (default: same as --cpu-strict)", .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.cpuparams_batch.strict_cpu = value != 0;
        }
    }.f } });
    // arg.cpp:1618
    try ctx.add(.{ .args = &.{"--prio-batch"}, .value_hint = "N", .help = try ctx.fmt("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", .{@as(c_int, @intCast(p.cpuparams_batch.priority))}), .handler = .{ .int = struct {
        fn f(q: *Params, prio: c_int) !void {
            if (prio < 0 or prio > 3) return throwStr("invalid value");
            q.cpuparams_batch.priority = @intCast(prio);
        }
    }.f } });
    // arg.cpp:1628
    try ctx.add(.{ .args = &.{"--poll-batch"}, .value_hint = "<0|1>", .help = "use polling to wait for work (default: same as --poll)", .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.cpuparams_batch.poll = @bitCast(value);
        }
    }.f } });
    // arg.cpp:1635
    try ctx.add((Arg{ .args = &.{ "-lcs", "--lookup-cache-static" }, .value_hint = "FNAME", .help = "path to static lookup cache to use for lookup decoding (not updated by generation)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.speculative.ngram_cache.lookup_cache_static = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{ .lookup, .server }));
    // arg.cpp:1642
    try ctx.add((Arg{ .args = &.{ "-lcd", "--lookup-cache-dynamic" }, .value_hint = "FNAME", .help = "path to dynamic lookup cache to use for lookup decoding (updated by generation)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.speculative.ngram_cache.lookup_cache_dynamic = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{ .lookup, .server }));
    // arg.cpp:1649
    try ctx.add((Arg{
        .args = &.{ "-c", "--ctx-size" },
        .value_hint = "N",
        .help = try ctx.fmt("size of the prompt context (default: %d, 0 = loaded from model)", .{p.n_ctx}),
        .handler = .{
            .int = struct {
                fn f(q: *Params, value: c_int) !void {
                    q.n_ctx = value;
                    // disable context reduction in llama_params_fit if the user explicitly requests the full context size:
                    if (value == 0) q.fit_params_min_ctx = @bitCast(@as(u32, std.math.maxInt(u32)));
                }
            }.f,
        },
    }).setEnv("LLAMA_ARG_CTX_SIZE"));
    // arg.cpp:1660
    try ctx.add((Arg{ .args = &.{"--kv-unified-per-slot"}, .value_hint = "N", .help = "context limit per parallel slot (default: unset, behavior unchanged).\nwhen set without -c/--ctx-size, the shared KV pool is sized to n_parallel*N", .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.kv_unified_per_slot = value;
        }
    }.f } }).setEnv("LLAMA_ARG_KV_UNIFIED_PER_SLOT").setExamples(&.{.server}));
    // arg.cpp:1668
    try ctx.add((Arg{ .args = &.{ "-n", "--predict", "--n-predict" }, .value_hint = "N", .help = if (ex == .completion)
        try ctx.fmt("number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)", .{p.n_predict})
    else
        try ctx.fmt("number of tokens to predict (default: %d, -1 = infinity)", .{p.n_predict}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.n_predict = value;
        }
    }.f } }).setEnv("LLAMA_ARG_N_PREDICT"));
    // arg.cpp:1679
    try ctx.add((Arg{ .args = &.{ "-b", "--batch-size" }, .value_hint = "N", .help = try ctx.fmt("logical maximum batch size (default: %d)", .{p.n_batch}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.n_batch = value;
        }
    }.f } }).setEnv("LLAMA_ARG_BATCH"));
    // arg.cpp:1686
    try ctx.add((Arg{ .args = &.{ "-ub", "--ubatch-size" }, .value_hint = "N", .help = try ctx.fmt("physical maximum batch size (default: %d)", .{p.n_ubatch}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.n_ubatch = value;
        }
    }.f } }).setEnv("LLAMA_ARG_UBATCH"));
    // arg.cpp:1693
    try ctx.add(.{ .args = &.{"--keep"}, .value_hint = "N", .help = try ctx.fmt("number of tokens to keep from the initial prompt (default: %d, -1 = all)", .{p.n_keep}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.n_keep = value;
        }
    }.f } });
    // arg.cpp:1700
    try ctx.add((Arg{ .args = &.{"--swa-full"}, .help = try ctx.fmt("use full-size SWA cache (default: %s)\n[(more info)](https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)", .{tf(p.swa_full)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.swa_full = true;
        }
    }.f } }).setEnv("LLAMA_ARG_SWA_FULL"));
    // arg.cpp:1708
    try ctx.add((Arg{ .args = &.{ "-ctxcp", "--ctx-checkpoints", "--swa-checkpoints" }, .value_hint = "N", .help = try ctx.fmt("max number of context checkpoints to create per slot (default: %d)[(more info)](https://github.com/ggml-org/llama.cpp/pull/15293)", .{p.n_ctx_checkpoints}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.n_ctx_checkpoints = value;
        }
    }.f } }).setEnv("LLAMA_ARG_CTX_CHECKPOINTS").setExamples(&.{ .server, .cli }));
    // arg.cpp:1716
    try ctx.add((Arg{ .args = &.{ "-cms", "--checkpoint-min-step" }, .value_hint = "N", .help = try ctx.fmt("minimum spacing between context checkpoints in tokens (default: %d, 0 = no minimum)", .{p.checkpoint_min_step}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            if (value < 0) return throwStr("checkpoint-min-step must be non-negative");
            q.checkpoint_min_step = value;
        }
    }.f } }).setEnv("LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT").setExamples(&.{.server}));
    // arg.cpp:1726
    try ctx.add((Arg{ .args = &.{ "-cram", "--cache-ram" }, .value_hint = "N", .help = try ctx.fmt("set the maximum cache size in MiB (default: %d, 0 for decision models, -1 - no limit, 0 - disable)[(more info)](https://github.com/ggml-org/llama.cpp/pull/16391)", .{p.cache_ram_mib}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.cache_ram_mib = value;
            q.cache_ram_explicit = true;
        }
    }.f } }).setEnv("LLAMA_ARG_CACHE_RAM").setExamples(&.{ .server, .cli }));
    // arg.cpp:1735
    try ctx.add((Arg{ .args = &.{ "-kvu", "--kv-unified" }, .args_neg = &.{ "-no-kvu", "--no-kv-unified" }, .help = "use single unified KV buffer shared across all sequences (default: enabled if number of slots is auto)", .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.kv_unified = value;
        }
    }.f } }).setEnv("LLAMA_ARG_KV_UNIFIED").setExamples(&.{ .server, .perplexity, .batched, .bench, .parallel, .speculative }));
    // arg.cpp:1743
    try ctx.add((Arg{ .args = &.{"--cache-idle-slots"}, .args_neg = &.{"--no-cache-idle-slots"}, .help = "save idle slots to the prompt cache on new task, and clear them when using unified KV (default: enabled, requires cache-ram)", .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.cache_idle_slots = value;
        }
    }.f } }).setEnv("LLAMA_ARG_CACHE_IDLE_SLOTS").setExamples(&.{.server}));
    // arg.cpp:1751
    try ctx.add((Arg{ .args = &.{"--context-shift"}, .args_neg = &.{"--no-context-shift"}, .help = try ctx.fmt("whether to use context shift on infinite text generation (default: %s)", .{ed(p.ctx_shift)}), .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.ctx_shift = value;
        }
    }.f } }).setExamples(&.{ .completion, .cli, .server, .imatrix, .perplexity }).setEnv("LLAMA_ARG_CONTEXT_SHIFT"));
    // arg.cpp:1759
    try ctx.add((Arg{ .args = &.{"--chunks"}, .value_hint = "N", .help = try ctx.fmt("max number of chunks to process (default: %d, -1 = all)", .{p.n_chunks}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.n_chunks = value;
        }
    }.f } }).setExamples(&.{ .imatrix, .perplexity, .retrieval }));
    // arg.cpp:1766
    try ctx.add((Arg{ .args = &.{ "-fa", "--flash-attn" }, .value_hint = "[on|off|auto]", .help = try ctx.fmt("set Flash Attention use ('on', 'off', or 'auto', default: '%s')", .{c.llama_flash_attn_type_name(p.flash_attn_type)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (arg.isTruthy(value)) {
                q.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_ENABLED;
            } else if (arg.isFalsey(value)) {
                q.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_DISABLED;
            } else if (arg.isAutoy(value)) {
                q.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_AUTO;
            } else {
                return throw("error: unknown value for --flash-attn: '%s'\n", .{value.ptr});
            }
        }
    }.f } }).setEnv("LLAMA_ARG_FLASH_ATTN"));
    // arg.cpp:1781
    try ctx.add((Arg{ .args = &.{ "-p", "--prompt" }, .value_hint = "PROMPT", .help = "prompt to start generation with; for system message, use -sys", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.prompt = try arg.dupe(q, value);
        }
    }.f } }).setExcludes(&.{.server}));
    // arg.cpp:1788
    try ctx.add((Arg{ .args = &.{ "-sys", "--system-prompt" }, .value_hint = "PROMPT", .help = "system prompt to use with model (if applicable, depending on chat template)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.system_prompt = try arg.dupe(q, value);
        }
    }.f } }).setExamples(&.{ .completion, .cli, .diffusion, .mtmd }));
    // arg.cpp:1795
    try ctx.add((Arg{ .args = &.{"--perf"}, .args_neg = &.{"--no-perf"}, .help = try ctx.fmt("whether to enable internal libllama performance timings (default: %s)", .{tf(p.no_perf)}), .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.no_perf = !value;
            q.sampling.no_perf = !value;
        }
    }.f } }).setEnv("LLAMA_ARG_PERF"));
    // arg.cpp:1812
    try ctx.add((Arg{
        .args = &.{ "-f", "--file" },
        .value_hint = "FNAME",
        .help = "a file containing the prompt (default: none)",
        .handler = .{
            .string = struct {
                fn f(q: *Params, value: S) !void {
                    var prompt = try readFile(q, value);
                    // store the external file name in params
                    q.prompt_file = try arg.dupe(q, value);
                    if (prompt.len != 0 and prompt[prompt.len - 1] == '\n') prompt = prompt[0 .. prompt.len - 1];
                    q.prompt = prompt;
                }
            }.f,
        },
    }).setExcludes(&.{.server}));
    // arg.cpp:1834
    try ctx.add((Arg{ .args = &.{"--in-file"}, .value_hint = "FNAME", .help = "an input file (use comma-separated values to specify multiple files)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const items = try parseCsvRow(q.arena, value);
            for (items.items) |item| {
                const file = std.c.fopen(item.ptr, "r") orelse return throw("error: failed to open file '%s'\n", .{item.ptr});
                _ = std.c.fclose(file);
                try q.in_files.append(q.arena, item);
            }
        }
    }.f } }).setExamples(&.{.imatrix}));
    // arg.cpp:1847
    try ctx.add((Arg{
        .args = &.{ "-bf", "--binary-file" },
        .value_hint = "FNAME",
        .help = "binary file containing the prompt (default: none)",
        .handler = .{
            .string = struct {
                fn f(q: *Params, value: S) !void {
                    const file = std.c.fopen(value.ptr, "rb") orelse return throw("error: failed to open file '%s'\n", .{value.ptr});
                    _ = std.c.fclose(file);
                    // store the external file name in params
                    q.prompt_file = try arg.dupe(q, value);
                    q.prompt = try readFile(q, value);
                    io.eprintf("Read %zu bytes from binary file %s\n", .{ q.prompt.len, value.ptr });
                }
            }.f,
        },
    }).setExcludes(&.{.server}));
    // arg.cpp:1863
    try ctx.add(.{ .args = &.{ "-e", "--escape" }, .args_neg = &.{"--no-escape"}, .help = try ctx.fmt("whether to process escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\) (default: %s)", .{tf(p.escape)}), .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.escape = value;
        }
    }.f } });
    // arg.cpp:1899
    try ctx.add((Arg{ .args = &.{ "-r", "--reverse-prompt" }, .value_hint = "PROMPT", .help = "halt generation at PROMPT, return control in interactive mode\n", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            try q.antiprompt.append(q.arena, try arg.dupe(q, value));
        }
    }.f } }).setExamples(&.{ .completion, .cli, .server }));
    // arg.cpp:1906
    try ctx.add((Arg{ .args = &.{ "-sp", "--special" }, .help = try ctx.fmt("special tokens output enabled (default: %s)", .{tf(p.special)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.special = true;
        }
    }.f } }).setExamples(&.{ .completion, .cli, .server }));
    // arg.cpp:1978
    try ctx.add((Arg{ .args = &.{"--warmup"}, .args_neg = &.{"--no-warmup"}, .help = try ctx.fmt("whether to perform warmup with an empty run (default: %s)", .{ed(p.warmup)}), .handler = .{ .bool_ = struct {
        fn f(q: *Params, value: bool) !void {
            q.warmup = value;
        }
    }.f } }).setExamples(&.{ .completion, .cli, .server, .mtmd, .embedding, .retrieval, .perplexity, .debug }));
    // arg.cpp:1986
    try ctx.add((Arg{ .args = &.{"--spm-infill"}, .help = try ctx.fmt("use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this. (default: %s)", .{ed(p.spm_infill)}), .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.spm_infill = true;
        }
    }.f } }).setExamples(&.{.server}));
    // arg.cpp:1996
    try ctx.add((Arg{ .args = &.{"--samplers"}, .value_hint = "SAMPLERS", .help = try ctx.fmt("samplers that will be used for generation in the order, separated by ';'\n(default: %s)", .{names_z}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            var names: std.ArrayList([]const u8) = .empty;
            var it = std.mem.splitScalar(u8, value, ';');
            while (it.next()) |n| try names.append(q.arena, n);
            q.sampling.samplers = try samplerTypesFromNames(q, names.items);
            q.sampling.user_sampling_config |= SamplingConfig.samplers;
        }
    }.f } }).setSampling());
    // arg.cpp:2005
    try ctx.add((Arg{ .args = &.{ "-s", "--seed" }, .value_hint = "SEED", .help = try ctx.fmt("RNG seed (default: %d, use random seed for %d)", .{ @as(c_uint, p.sampling.seed), @as(c_uint, c.LLAMA_DEFAULT_SEED) }), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.seed = @truncate(try arg.stoul(value));
        }
    }.f } }).setSampling());
    // arg.cpp:2012
    try ctx.add((Arg{ .args = &.{ "--sampler-seq", "--sampling-seq" }, .value_hint = "SEQUENCE", .help = try ctx.fmt("simplified sequence for samplers that will be used (default: %s)", .{chars_z}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.samplers = try samplerTypesFromChars(q, value);
        }
    }.f } }).setSampling());
    // arg.cpp:2019
    try ctx.add((Arg{ .args = &.{"--ignore-eos"}, .help = "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.sampling.ignore_eos = true;
        }
    }.f } }).setSampling());
    // arg.cpp:2026
    try ctx.add((Arg{ .args = &.{ "--temp", "--temperature" }, .value_hint = "N", .help = try ctx.fmt("temperature (default: %.2f)", .{@as(f64, p.sampling.temp)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.temp = @max(try arg.stof(value), 0.0);
            q.sampling.user_sampling_config |= SamplingConfig.temp;
        }
    }.f } }).setSampling().setEnv("LLAMA_ARG_TEMPERATURE"));
    // arg.cpp:2035
    try ctx.add((Arg{ .args = &.{"--top-k"}, .value_hint = "N", .help = try ctx.fmt("top-k sampling (default: %d, 0 = disabled)", .{p.sampling.top_k}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.sampling.top_k = value;
            q.sampling.user_sampling_config |= SamplingConfig.top_k;
        }
    }.f } }).setSampling().setEnv("LLAMA_ARG_TOP_K"));
    // arg.cpp:2043
    try ctx.add((Arg{ .args = &.{"--top-p"}, .value_hint = "N", .help = try ctx.fmt("top-p sampling (default: %.2f, 1.0 = disabled)", .{@as(f64, p.sampling.top_p)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.top_p = try arg.stof(value);
            q.sampling.user_sampling_config |= SamplingConfig.top_p;
        }
    }.f } }).setSampling().setEnv("LLAMA_ARG_TOP_P"));
    // arg.cpp:2051
    try ctx.add((Arg{ .args = &.{"--min-p"}, .value_hint = "N", .help = try ctx.fmt("min-p sampling (default: %.2f, 0.0 = disabled)", .{@as(f64, p.sampling.min_p)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.min_p = try arg.stof(value);
            q.sampling.user_sampling_config |= SamplingConfig.min_p;
        }
    }.f } }).setSampling().setEnv("LLAMA_ARG_MIN_P"));
    // arg.cpp:2059
    try ctx.add((Arg{ .args = &.{ "--top-nsigma", "--top-n-sigma" }, .value_hint = "N", .help = try ctx.fmt("top-n-sigma sampling (default: %.2f, -1.0 = disabled)", .{@as(f64, p.sampling.top_n_sigma)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.top_n_sigma = try arg.stof(value);
        }
    }.f } }).setSampling());
    // arg.cpp:2066
    try ctx.add((Arg{ .args = &.{"--xtc-probability"}, .value_hint = "N", .help = try ctx.fmt("xtc probability (default: %.2f, 0.0 = disabled)", .{@as(f64, p.sampling.xtc_probability)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.xtc_probability = try arg.stof(value);
            q.sampling.user_sampling_config |= SamplingConfig.xtc_probability;
        }
    }.f } }).setSampling());
    // arg.cpp:2074
    try ctx.add((Arg{ .args = &.{"--xtc-threshold"}, .value_hint = "N", .help = try ctx.fmt("xtc threshold (default: %.2f, 1.0 = disabled)", .{@as(f64, p.sampling.xtc_threshold)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.xtc_threshold = try arg.stof(value);
            q.sampling.user_sampling_config |= SamplingConfig.xtc_threshold;
        }
    }.f } }).setSampling());
    // arg.cpp:2082
    try ctx.add((Arg{ .args = &.{ "--typical", "--typical-p" }, .value_hint = "N", .help = try ctx.fmt("locally typical sampling, parameter p (default: %.2f, 1.0 = disabled)", .{@as(f64, p.sampling.typ_p)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.typ_p = try arg.stof(value);
        }
    }.f } }).setSampling());
    // arg.cpp:2089
    try ctx.add((Arg{ .args = &.{"--repeat-last-n"}, .value_hint = "N", .help = try ctx.fmt("last n tokens to consider for penalize (default: %d, 0 = disabled)", .{p.sampling.penalty_last_n}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            if (value < 0) return throw("error: invalid repeat-last-n = %d\n", .{value});
            q.sampling.penalty_last_n = value;
            q.sampling.n_prev = @max(q.sampling.n_prev, q.sampling.penalty_last_n);
            q.sampling.user_sampling_config |= SamplingConfig.penalty_last_n;
        }
    }.f } }).setSampling());
    // arg.cpp:2101
    try ctx.add((Arg{ .args = &.{"--repeat-penalty"}, .value_hint = "N", .help = try ctx.fmt("penalize repeat sequence of tokens (default: %.2f, 1.0 = disabled)", .{@as(f64, p.sampling.penalty_repeat)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const penalty_repeat = try arg.stof(value);
            if (!std.math.isFinite(penalty_repeat) or penalty_repeat <= 0.0 or !std.math.isFinite(1.0 / penalty_repeat)) {
                return throwStr("error: repeat-penalty must be finite and greater than 0\n");
            }
            q.sampling.penalty_repeat = penalty_repeat;
            q.sampling.user_sampling_config |= SamplingConfig.penalty_repeat;
        }
    }.f } }).setSampling().setEnv("LLAMA_ARG_REPEAT_PENALTY"));
    // arg.cpp:2115
    try ctx.add((Arg{ .args = &.{"--presence-penalty"}, .value_hint = "N", .help = try ctx.fmt("repeat alpha presence penalty (default: %.2f, 0.0 = disabled)", .{@as(f64, p.sampling.penalty_present)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const penalty_present = try arg.stof(value);
            if (!std.math.isFinite(penalty_present)) return throwStr("error: presence-penalty must be finite\n");
            q.sampling.penalty_present = penalty_present;
        }
    }.f } }).setSampling().setEnv("LLAMA_ARG_PRESENCE_PENALTY"));
    // arg.cpp:2126
    try ctx.add((Arg{ .args = &.{"--frequency-penalty"}, .value_hint = "N", .help = try ctx.fmt("repeat alpha frequency penalty (default: %.2f, 0.0 = disabled)", .{@as(f64, p.sampling.penalty_freq)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const penalty_freq = try arg.stof(value);
            if (!std.math.isFinite(penalty_freq)) return throwStr("error: frequency-penalty must be finite\n");
            q.sampling.penalty_freq = penalty_freq;
        }
    }.f } }).setSampling().setEnv("LLAMA_ARG_FREQUENCY_PENALTY"));
    // arg.cpp:2137
    try ctx.add((Arg{ .args = &.{"--dry-multiplier"}, .value_hint = "N", .help = try ctx.fmt("set DRY sampling multiplier (default: %.2f, 0.0 = disabled)", .{@as(f64, p.sampling.dry_multiplier)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.dry_multiplier = try arg.stof(value);
        }
    }.f } }).setSampling());
    // arg.cpp:2144
    try ctx.add((Arg{ .args = &.{"--dry-base"}, .value_hint = "N", .help = try ctx.fmt("set DRY sampling base value (default: %.2f)", .{@as(f64, p.sampling.dry_base)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const potential_base = try arg.stof(value);
            if (potential_base >= 1.0) q.sampling.dry_base = potential_base;
        }
    }.f } }).setSampling());
    // arg.cpp:2155
    try ctx.add((Arg{ .args = &.{"--dry-allowed-length"}, .value_hint = "N", .help = try ctx.fmt("set allowed length for DRY sampling (default: %d)", .{p.sampling.dry_allowed_length}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.sampling.dry_allowed_length = value;
        }
    }.f } }).setSampling());
    // arg.cpp:2162
    try ctx.add((Arg{ .args = &.{"--dry-penalty-last-n"}, .value_hint = "N", .help = try ctx.fmt("set DRY penalty for the last n tokens (default: %d, 0 = disable)", .{p.sampling.dry_penalty_last_n}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            if (value < 0) return throw("error: invalid dry-penalty-last-n = %d\n", .{value});
            q.sampling.dry_penalty_last_n = value;
        }
    }.f } }).setSampling());
    // arg.cpp:2172
    const breakers = try dryBreakersText(p);
    try ctx.add((Arg{ .args = &.{"--dry-sequence-breaker"}, .value_hint = "STRING", .help = try ctx.fmt("add sequence breaker for DRY sampling, clearing out default breakers (%.*s) in the process; use \"none\" to not use any sequence breakers\n", .{ @as(c_int, @intCast(breakers.len)), breakers.ptr }), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (!dry_defaults_cleared) {
                q.sampling.dry_sequence_breakers.clearRetainingCapacity();
                dry_defaults_cleared = true;
            }
            if (std.mem.eql(u8, value, "none")) {
                q.sampling.dry_sequence_breakers.clearRetainingCapacity();
            } else {
                try q.sampling.dry_sequence_breakers.append(q.arena, try arg.dupe(q, value));
            }
        }
    }.f } }).setSampling());
    // arg.cpp:2198
    try ctx.add((Arg{ .args = &.{"--adaptive-target"}, .value_hint = "N", .help = try ctx.fmt("adaptive-p: select tokens near this probability (valid range 0.0 to 1.0; negative = disabled) (default: %.2f)\n[(more info)](https://github.com/ggml-org/llama.cpp/pull/17927)", .{@as(f64, p.sampling.adaptive_target)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.adaptive_target = try arg.stof(value);
        }
    }.f } }).setSampling());
    // arg.cpp:2208
    try ctx.add((Arg{ .args = &.{"--adaptive-decay"}, .value_hint = "N", .help = try ctx.fmt("adaptive-p: decay rate for target adaptation over time. lower values are more reactive, higher values are more stable.\n(valid range 0.0 to 0.99) (default: %.2f)", .{@as(f64, p.sampling.adaptive_decay)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.adaptive_decay = try arg.stof(value);
        }
    }.f } }).setSampling());
    // arg.cpp:2218
    try ctx.add((Arg{ .args = &.{"--dynatemp-range"}, .value_hint = "N", .help = try ctx.fmt("dynamic temperature range (default: %.2f, 0.0 = disabled)", .{@as(f64, p.sampling.dynatemp_range)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.dynatemp_range = try arg.stof(value);
        }
    }.f } }).setSampling());
    // arg.cpp:2225
    try ctx.add((Arg{ .args = &.{"--dynatemp-exp"}, .value_hint = "N", .help = try ctx.fmt("dynamic temperature exponent (default: %.2f)", .{@as(f64, p.sampling.dynatemp_exponent)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.dynatemp_exponent = try arg.stof(value);
        }
    }.f } }).setSampling());
    // arg.cpp:2232
    try ctx.add((Arg{ .args = &.{"--mirostat"}, .value_hint = "N", .help = try ctx.fmt("use Mirostat sampling.\nTop K, Nucleus and Locally Typical samplers are ignored if used.\n(default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", .{p.sampling.mirostat}), .handler = .{ .int = struct {
        fn f(q: *Params, value: c_int) !void {
            q.sampling.mirostat = value;
            q.sampling.user_sampling_config |= SamplingConfig.mirostat;
        }
    }.f } }).setSampling());
    // arg.cpp:2241
    try ctx.add((Arg{ .args = &.{"--mirostat-lr"}, .value_hint = "N", .help = try ctx.fmt("Mirostat learning rate, parameter eta (default: %.2f)", .{@as(f64, p.sampling.mirostat_eta)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.mirostat_eta = try arg.stof(value);
            q.sampling.user_sampling_config |= SamplingConfig.mirostat_eta;
        }
    }.f } }).setSampling());
    // arg.cpp:2249
    try ctx.add((Arg{ .args = &.{"--mirostat-ent"}, .value_hint = "N", .help = try ctx.fmt("Mirostat target entropy, parameter tau (default: %.2f)", .{@as(f64, p.sampling.mirostat_tau)}), .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.mirostat_tau = try arg.stof(value);
            q.sampling.user_sampling_config |= SamplingConfig.mirostat_tau;
        }
    }.f } }).setSampling());
    // arg.cpp:2257
    try ctx.add((Arg{ .args = &.{ "-l", "--logit-bias" }, .value_hint = "TOKEN_ID(+/-)BIAS", .help = "modifies the likelihood of token appearing in the completion,\ni.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\nor `--logit-bias 15043-1` to decrease likelihood of token ' Hello'", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            const bias = parseLogitBias(value) orelse return throwStr("invalid input format");
            try q.sampling.logit_bias.append(q.arena, bias);
        }
    }.f } }).setSampling());
    // arg.cpp:2279
    try ctx.add((Arg{ .args = &.{"--grammar"}, .value_hint = "GRAMMAR", .help = "BNF-like grammar to constrain generations (see samples in grammars/ dir)", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.grammar_type = .user;
            q.sampling.grammar = try arg.dupe(q, value);
        }
    }.f } }).setSampling());
    // arg.cpp:2286
    try ctx.add((Arg{ .args = &.{"--grammar-file"}, .value_hint = "FNAME", .help = "file to read grammar from", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            q.sampling.grammar = try readFile(q, value);
            q.sampling.grammar_type = .user;
        }
    }.f } }).setSampling());
    // arg.cpp:2293
    try ctx.add((Arg{ .args = &.{ "-j", "--json-schema" }, .value_hint = "SCHEMA", .help = "JSON schema to constrain generations (https://json-schema.org/), e.g. `{\"type\": \"object\"}` for any JSON object", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            try jsonSchemaGrammar(q, value);
        }
    }.f } }).setSampling());
    // arg.cpp:2300
    try ctx.add((Arg{ .args = &.{ "-jf", "--json-schema-file" }, .value_hint = "FILE", .help = "File containing a JSON schema to constrain generations (https://json-schema.org/), e.g. `{\"type\": \"object\"}` for any JSON object", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            try jsonSchemaGrammar(q, try readFile(q, value));
        }
    }.f } }).setSampling());
    // arg.cpp:2317
    try ctx.add((Arg{ .args = &.{ "-bs", "--backend-sampling" }, .help = "enable backend sampling (experimental) (default: disabled)", .handler = .{ .void_ = struct {
        fn f(q: *Params) !void {
            q.sampling.backend_sampling = true;
        }
    }.f } }).setSampling().setEnv("LLAMA_ARG_BACKEND_SAMPLING"));
    // arg.cpp:2324
    try ctx.add((Arg{ .args = &.{"--pooling"}, .value_hint = "{none,mean,cls,last,rank}", .help = "pooling type for embeddings, use model default if unspecified", .handler = .{ .string = struct {
        fn f(q: *Params, value: S) !void {
            if (std.mem.eql(u8, value, "none")) {
                q.pooling_type = c.LLAMA_POOLING_TYPE_NONE;
            } else if (std.mem.eql(u8, value, "mean")) {
                q.pooling_type = c.LLAMA_POOLING_TYPE_MEAN;
            } else if (std.mem.eql(u8, value, "cls")) {
                q.pooling_type = c.LLAMA_POOLING_TYPE_CLS;
            } else if (std.mem.eql(u8, value, "last")) {
                q.pooling_type = c.LLAMA_POOLING_TYPE_LAST;
            } else if (std.mem.eql(u8, value, "rank")) {
                q.pooling_type = c.LLAMA_POOLING_TYPE_RANK;
            } else return throwStr("invalid value");
        }
    }.f } }).setExamples(&.{ .embedding, .retrieval, .server, .debug }).setEnv("LLAMA_ARG_POOLING"));
}

/// Option 62's parse: `ss >> key >> sign` then the rest of the line as the
/// bias, `+` or `-` only.
fn parseLogitBias(value: []const u8) ?c.llama_logit_bias {
    var i: usize = 0;
    while (i < value.len and std.ascii.isWhitespace(value[i])) i += 1;
    const key_start = i;
    if (i < value.len and (value[i] == '+' or value[i] == '-')) i += 1;
    while (i < value.len and std.ascii.isDigit(value[i])) i += 1;
    const key = std.fmt.parseInt(i32, value[key_start..i], 10) catch return null;
    while (i < value.len and std.ascii.isWhitespace(value[i])) i += 1;
    if (i >= value.len) return null;
    const sign = value[i];
    i += 1;
    // `std::getline` fails on an empty remainder
    if (i >= value.len) return null;
    if (sign != '+' and sign != '-') return null;
    var buf: [256]u8 = undefined;
    const rest = std.fmt.bufPrintSentinel(&buf, "{s}", .{value[i..]}, 0) catch return null;
    const v = arg.stof(rest) catch return null;
    return .{ .token = key, .bias = v * (if (sign == '-') @as(f32, -1.0) else 1.0) };
}

/// Adds every option, in the C's order.
pub fn addAll(ctx: *Context) !void {
    try addPart1(ctx);
    try @import("arg_options2.zig").add(ctx);
    try @import("arg_options3.zig").add(ctx);
    try @import("arg_options4.zig").add(ctx);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
