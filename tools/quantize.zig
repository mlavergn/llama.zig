//! `llama-quantize`: quantize a GGUF model to another type.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/quantize/quantize.cpp` at v0.4.1
//! (`b29c606e2`). Each declaration names the C++ it replaces and the line it
//! began at. The quantizing itself is libllama's `llama_model_quantize`,
//! ported and gated by `scripts/quant-diff`; this is the command line
//! around it.
//!
//! # Map order
//!
//! `imatrix_data` is an `std::unordered_map` in the C, and only its size and
//! its contents by name reach libllama -- which looks entries up by name --
//! so its order is not reproduced. What is printed walks
//! `common_imatrix::entries`, a `std::map`, in name order.

const std = @import("std");
const builtin = @import("builtin");
const common = @import("tools_common");
const c = common.c;
const io = common.io;
const log = common.log;

comptime {
    // libllama, compiled in (see `build/tools.zig`).
    _ = @import("llama");
}

/// Mirrors `tensor_type_option` (quantize.cpp:23 @b29c606e2).
const TensorTypeOption = struct {
    name: [:0]u8,
    type: c.ggml_type = c.GGML_TYPE_COUNT,
};

/// Mirrors `quant_option` (quantize.cpp:28 @b29c606e2).
const QuantOption = struct {
    name: []const u8,
    ftype: c.llama_ftype,
    desc: [*:0]const u8,
};

/// Ports `QUANT_OPTIONS` (quantize.cpp:34 @b29c606e2).
const quant_options = [_]QuantOption{
    .{ .name = "Q1_0", .ftype = c.LLAMA_FTYPE_MOSTLY_Q1_0, .desc = " 1.125 bpw quantization" },
    .{ .name = "Q2_0", .ftype = c.LLAMA_FTYPE_MOSTLY_Q2_0, .desc = " 2.25 bpw quantization (group 64)" },
    .{ .name = "Q4_0", .ftype = c.LLAMA_FTYPE_MOSTLY_Q4_0, .desc = " 4.34G, +0.4685 ppl @ Llama-3-8B" },
    .{ .name = "Q4_1", .ftype = c.LLAMA_FTYPE_MOSTLY_Q4_1, .desc = " 4.78G, +0.4511 ppl @ Llama-3-8B" },
    .{ .name = "MXFP4_MOE", .ftype = c.LLAMA_FTYPE_MOSTLY_MXFP4_MOE, .desc = " MXFP4 MoE" },
    .{ .name = "Q5_0", .ftype = c.LLAMA_FTYPE_MOSTLY_Q5_0, .desc = " 5.21G, +0.1316 ppl @ Llama-3-8B" },
    .{ .name = "Q5_1", .ftype = c.LLAMA_FTYPE_MOSTLY_Q5_1, .desc = " 5.65G, +0.1062 ppl @ Llama-3-8B" },
    .{ .name = "IQ2_XXS", .ftype = c.LLAMA_FTYPE_MOSTLY_IQ2_XXS, .desc = " 2.06 bpw quantization" },
    .{ .name = "IQ2_XS", .ftype = c.LLAMA_FTYPE_MOSTLY_IQ2_XS, .desc = " 2.31 bpw quantization" },
    .{ .name = "IQ2_S", .ftype = c.LLAMA_FTYPE_MOSTLY_IQ2_S, .desc = " 2.5  bpw quantization" },
    .{ .name = "IQ2_M", .ftype = c.LLAMA_FTYPE_MOSTLY_IQ2_M, .desc = " 2.7  bpw quantization" },
    .{ .name = "IQ1_S", .ftype = c.LLAMA_FTYPE_MOSTLY_IQ1_S, .desc = " 1.56 bpw quantization" },
    .{ .name = "IQ1_M", .ftype = c.LLAMA_FTYPE_MOSTLY_IQ1_M, .desc = " 1.75 bpw quantization" },
    .{ .name = "TQ1_0", .ftype = c.LLAMA_FTYPE_MOSTLY_TQ1_0, .desc = " 1.69 bpw ternarization" },
    .{ .name = "TQ2_0", .ftype = c.LLAMA_FTYPE_MOSTLY_TQ2_0, .desc = " 2.06 bpw ternarization" },
    .{ .name = "Q2_K", .ftype = c.LLAMA_FTYPE_MOSTLY_Q2_K, .desc = " 2.96G, +3.5199 ppl @ Llama-3-8B" },
    .{ .name = "Q2_K_S", .ftype = c.LLAMA_FTYPE_MOSTLY_Q2_K_S, .desc = " 2.96G, +3.1836 ppl @ Llama-3-8B" },
    .{ .name = "IQ3_XXS", .ftype = c.LLAMA_FTYPE_MOSTLY_IQ3_XXS, .desc = " 3.06 bpw quantization" },
    .{ .name = "IQ3_S", .ftype = c.LLAMA_FTYPE_MOSTLY_IQ3_S, .desc = " 3.44 bpw quantization" },
    .{ .name = "IQ3_M", .ftype = c.LLAMA_FTYPE_MOSTLY_IQ3_M, .desc = " 3.66 bpw quantization mix" },
    .{ .name = "Q3_K", .ftype = c.LLAMA_FTYPE_MOSTLY_Q3_K_M, .desc = "alias for Q3_K_M" },
    .{ .name = "IQ3_XS", .ftype = c.LLAMA_FTYPE_MOSTLY_IQ3_XS, .desc = " 3.3 bpw quantization" },
    .{ .name = "Q3_K_S", .ftype = c.LLAMA_FTYPE_MOSTLY_Q3_K_S, .desc = " 3.41G, +1.6321 ppl @ Llama-3-8B" },
    .{ .name = "Q3_K_M", .ftype = c.LLAMA_FTYPE_MOSTLY_Q3_K_M, .desc = " 3.74G, +0.6569 ppl @ Llama-3-8B" },
    .{ .name = "Q3_K_L", .ftype = c.LLAMA_FTYPE_MOSTLY_Q3_K_L, .desc = " 4.03G, +0.5562 ppl @ Llama-3-8B" },
    .{ .name = "IQ4_NL", .ftype = c.LLAMA_FTYPE_MOSTLY_IQ4_NL, .desc = " 4.50 bpw non-linear quantization" },
    .{ .name = "IQ4_XS", .ftype = c.LLAMA_FTYPE_MOSTLY_IQ4_XS, .desc = " 4.25 bpw non-linear quantization" },
    .{ .name = "Q4_K", .ftype = c.LLAMA_FTYPE_MOSTLY_Q4_K_M, .desc = "alias for Q4_K_M" },
    .{ .name = "Q4_K_S", .ftype = c.LLAMA_FTYPE_MOSTLY_Q4_K_S, .desc = " 4.37G, +0.2689 ppl @ Llama-3-8B" },
    .{ .name = "Q4_K_M", .ftype = c.LLAMA_FTYPE_MOSTLY_Q4_K_M, .desc = " 4.58G, +0.1754 ppl @ Llama-3-8B" },
    .{ .name = "Q5_K", .ftype = c.LLAMA_FTYPE_MOSTLY_Q5_K_M, .desc = "alias for Q5_K_M" },
    .{ .name = "Q5_K_S", .ftype = c.LLAMA_FTYPE_MOSTLY_Q5_K_S, .desc = " 5.21G, +0.1049 ppl @ Llama-3-8B" },
    .{ .name = "Q5_K_M", .ftype = c.LLAMA_FTYPE_MOSTLY_Q5_K_M, .desc = " 5.33G, +0.0569 ppl @ Llama-3-8B" },
    .{ .name = "Q6_K", .ftype = c.LLAMA_FTYPE_MOSTLY_Q6_K, .desc = " 6.14G, +0.0217 ppl @ Llama-3-8B" },
    .{ .name = "Q8_0", .ftype = c.LLAMA_FTYPE_MOSTLY_Q8_0, .desc = " 7.96G, +0.0026 ppl @ Llama-3-8B" },
    .{ .name = "F16", .ftype = c.LLAMA_FTYPE_MOSTLY_F16, .desc = "14.00G, +0.0020 ppl @ Mistral-7B" },
    .{ .name = "BF16", .ftype = c.LLAMA_FTYPE_MOSTLY_BF16, .desc = "14.00G, -0.0050 ppl @ Mistral-7B" },
    .{ .name = "F32", .ftype = c.LLAMA_FTYPE_ALL_F32, .desc = "26.00G              @ 7B" },
    // Note: Ensure COPY comes after F32 to avoid ftype 0 from matching.
    .{ .name = "COPY", .ftype = c.LLAMA_FTYPE_ALL_F32, .desc = "only copy tensors, no quantizing" },
};

/// One line of the type list `usage` prints, named so its `:` is not read
/// as a citation.
const type_line = "%-7.*s : %s\n";

/// `LLM_KV_QUANTIZE_IMATRIX_FILE`, `LLM_KV_QUANTIZE_IMATRIX_DATASET`,
/// `LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES`, `LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS`
/// (quantize.cpp:77, 78, 79, 80 @b29c606e2).
const kv_imatrix_file = "quantize.imatrix.file";
const kv_imatrix_dataset = "quantize.imatrix.dataset";
const kv_imatrix_n_entries = "quantize.imatrix.entries_count";
const kv_imatrix_n_chunks = "quantize.imatrix.chunks_count";

/// Ports `striequals` (quantize.cpp:82 @b29c606e2).
fn striequals(a: []const u8, b: []const u8) bool {
    return std.ascii.eqlIgnoreCase(a, b);
}

/// Ports `try_parse_ftype` (quantize.cpp:92 @b29c606e2): a name, case
/// folded, or the ftype's number, the first option with it.
fn tryParseFtype(arena: std.mem.Allocator, ftype_str_in: [:0]const u8, ftype: *c.llama_ftype, ftype_str_out: *[]const u8) !bool {
    const ftype_str = try arena.dupeSentinel(u8, ftype_str_in, 0);
    for (ftype_str) |*ch| ch.* = std.ascii.toUpper(ch.*);
    for (quant_options) |it| {
        if (striequals(it.name, ftype_str)) {
            ftype.* = it.ftype;
            ftype_str_out.* = it.name;
            return true;
        }
    }
    switch (common.cxx.stoi(ftype_str)) {
        .ok => |ftype_int| for (quant_options) |it| {
            if (it.ftype == ftype_int) {
                ftype.* = it.ftype;
                ftype_str_out.* = it.name;
                return true;
            }
        },
        // stoi failed
        else => {},
    }
    return false;
}

/// Ports `usage` (quantize.cpp:122 @b29c606e2).
fn usage(executable: [*:0]const u8) noreturn {
    io.printf("usage: %s [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--include-weights]\n", .{executable});
    io.printf("       [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--tensor-type] [--tensor-type-file]\n", .{});
    io.printf("       [--prune-layers] [--keep-split] [--override-kv] [--dry-run] [--max-buffer-size]\n", .{});
    io.printf("       model-f32.gguf [model-quant.gguf] type [nthreads]\n\n", .{});
    io.printf("  --allow-requantize\n", .{});
    io.printf("                                      allow requantizing tensors that have already been quantized\n", .{});
    io.printf("                                      WARNING: this can severely reduce quality compared to quantizing\n", .{});
    io.printf("                                               from 16bit or 32bit!\n", .{});
    io.printf("  --leave-output-tensor\n", .{});
    io.printf("                                      leave output.weight un(re)quantized\n", .{});
    io.printf("                                      increases model size but may also increase quality, especially when requantizing\n", .{});
    io.printf("  --pure\n", .{});
    io.printf("                                      disable k-quant mixtures and quantize all tensors to the same type\n", .{});
    io.printf("  --imatrix file_name\n", .{});
    io.printf("                                      use data in file_name as importance matrix for quant optimizations\n", .{});
    io.printf("  --include-weights tensor_name\n", .{});
    io.printf("                                      use importance matrix for this/these tensor(s)\n", .{});
    io.printf("  --exclude-weights tensor_name\n", .{});
    io.printf("                                      do not use importance matrix for this/these tensor(s)\n", .{});
    io.printf("  --output-tensor-type ggml_type\n", .{});
    io.printf("                                      use this ggml_type for the output.weight tensor\n", .{});
    io.printf("  --token-embedding-type ggml_type\n", .{});
    io.printf("                                      use this ggml_type for the token embeddings tensor\n", .{});
    io.printf("  --tensor-type tensor_name=ggml_type\n", .{});
    io.printf("                                      quantize this tensor to this ggml_type\n", .{});
    io.printf("                                      this is an advanced option to selectively quantize tensors. may be specified multiple times.\n", .{});
    io.printf("                                      example: --tensor-type attn_q=q8_0\n", .{});
    io.printf("  --tensor-type-file tensor_types.txt\n", .{});
    io.printf("                                      list of tensors to quantize to a specific ggml_type\n", .{});
    io.printf("                                      this is an advanced option to selectively quantize a long list of tensors.\n", .{});
    io.printf("                                      the file should use the same format as above, separated by spaces or newlines.\n", .{});
    io.printf("  --prune-layers L0,L1,L2...\n", .{});
    io.printf("                                      comma-separated list of layer numbers to prune from the model\n", .{});
    io.printf("                                      WARNING: this is an advanced option, use with care.\n", .{});
    io.printf("  --keep-split\n", .{});
    io.printf("                                      generate quantized model in the same shards as input\n", .{});
    io.printf("  --override-kv KEY=TYPE:VALUE\n", .{});
    io.printf("                                      override model metadata by key in the quantized model. may be specified multiple times.\n", .{});
    io.printf("                                      WARNING: this is an advanced option, use with care.\n", .{});
    io.printf("  --dry-run\n", .{});
    io.printf("                                      calculate and show the final quantization size without performing quantization\n", .{});
    io.printf("                                      example: llama-quantize --dry-run model-f32.gguf Q4_K\n", .{});
    io.printf("  --max-buffer-size MiB\n", .{});
    io.printf("                                      max amount of tensor rows kept in memory while quantizing one tensor (default: 8192)\n", .{});
    io.printf("                                      lower it to quantize models with very large tensors on a machine with little RAM\n\n", .{});
    io.printf("note: --include-weights and --exclude-weights cannot be used together\n\n", .{});
    io.printf("-----------------------------------------------------------------------------\n", .{});
    io.printf(" allowed quantization types\n", .{});
    io.printf("-----------------------------------------------------------------------------\n\n", .{});
    for (quant_options) |it| {
        if (!std.mem.eql(u8, it.name, "COPY")) {
            io.printf("  %2d  or  ", .{@as(c_int, @intCast(it.ftype))});
        } else {
            io.printf("          ", .{});
        }
        io.printf(type_line, .{ @as(c_int, @intCast(it.name.len)), it.name.ptr, it.desc });
    }
    common.cExit(1);
}

/// `imatrix_data`: name to the values handed to libllama.
const ImatrixData = std.StringArrayHashMapUnmanaged([]f32);

/// Ports `load_imatrix` (quantize.cpp:183 @b29c606e2).
fn loadImatrix(a: std.mem.Allocator, imatrix_file: [:0]const u8, imatrix_datasets: *std.ArrayList([]const u8), imatrix_data: *ImatrixData) !c_int {
    const func = "load_imatrix";
    var loaded = try common.imatrix_loader.Imatrix.init(a);
    if (!try common.imatrix_loader.load(imatrix_file, &loaded)) {
        io.eprintf("%s: failed to load imatrix from '%s'\n", .{ func, imatrix_file.ptr });
        common.cExit(1);
    }

    if (!loaded.is_legacy and !loaded.has_metadata) {
        io.eprintf("%s: missing imatrix metadata in file %s\n", .{ func, imatrix_file.ptr });
        common.cExit(1);
    }

    const trace = std.c.getenv("LLAMA_TRACE") != null;
    const names = try loaded.sortedNames(a);
    for (names) |name| {
        const entry = loaded.entries.get(name).?;
        const gop = try imatrix_data.getOrPut(a, name);
        if (!gop.found_existing) gop.key_ptr.* = try a.dupe(u8, name);
        const e = try a.alloc(f32, entry.sums.items.len);
        gop.value_ptr.* = e;

        if (!loaded.is_legacy) {
            // GGUF format: normalize by per-expert counts
            const ncounts: i64 = @intCast(entry.counts.items.len);
            const ne0: i64 = @divTrunc(@as(i64, @intCast(entry.sums.items.len)), ncounts);

            var j: i64 = 0;
            while (j < ncounts) : (j += 1) {
                const count: f32 = @floatFromInt(entry.counts.items[@intCast(j)]);
                var i: i64 = 0;
                while (i < ne0) : (i += 1) {
                    const k: usize = @intCast(j * ne0 + i);
                    e[k] = if (count > 0.0) entry.sums.items[k] / count else 1;
                }
            }

            if (trace) {
                var max_count: f32 = 0.0;
                for (entry.counts.items) |cnt| {
                    const count: f32 = @floatFromInt(cnt);
                    if (count > max_count) max_count = count;
                }
                io.printf("%s: loaded data (size = %6d, n_tokens = %6d, n_chunks = %6d) for '%.*s'\n", .{
                    func,
                    @as(c_int, @intCast(e.len)),
                    @as(c_int, @intFromFloat(max_count)),
                    @as(c_int, @intFromFloat(max_count / @as(f32, @floatFromInt(loaded.chunk_size)))),
                    @as(c_int, @intCast(name.len)),
                    name.ptr,
                });
            }
        } else {
            // Legacy format: sums contain (raw/count)*ncall, divide by ncall
            const ncall: i64 = if (entry.counts.items.len == 0) 0 else entry.counts.items[0];
            for (e, entry.sums.items) |*dst, s| dst.* = if (ncall > 0) s / @as(f32, @floatFromInt(ncall)) else s;

            if (trace) {
                io.printf("%s: loaded data (size = %6d, ncall = %6d) for '%.*s'\n", .{
                    func, @as(c_int, @intCast(e.len)), @as(c_int, @truncate(ncall)), @as(c_int, @intCast(name.len)), name.ptr,
                });
            }
        }
    }

    imatrix_datasets.* = loaded.datasets;
    loaded.datasets = .empty;

    if (imatrix_datasets.items.len != 0) {
        const d0 = imatrix_datasets.items[0];
        io.printf("%s: imatrix datasets=['%.*s'", .{ func, @as(c_int, @intCast(d0.len)), d0.ptr });
        for (imatrix_datasets.items[1..]) |d| io.printf(", '%.*s'", .{ @as(c_int, @intCast(d.len)), d.ptr });
        io.printf("]\n", .{});
    }

    io.printf("%s: loaded %d importance matrix entries from %s computed on %d chunks\n", .{
        func, @as(c_int, @intCast(imatrix_data.count())), imatrix_file.ptr, loaded.chunk_count,
    });

    return loaded.chunk_count;
}

/// Ports `prepare_imatrix` (quantize.cpp:263 @b29c606e2).
fn prepareImatrix(a: std.mem.Allocator, imatrix_file: [:0]const u8, imatrix_dataset: *std.ArrayList([]const u8), included_weights: []const [:0]const u8, excluded_weights: []const [:0]const u8, imatrix_data: *ImatrixData) !c_int {
    var m_last_call: c_int = -1;
    if (imatrix_file.len != 0) m_last_call = try loadImatrix(a, imatrix_file, imatrix_dataset, imatrix_data);
    if (imatrix_data.count() == 0) return m_last_call;
    for (excluded_weights) |name| {
        var i: usize = 0;
        while (i < imatrix_data.count()) {
            if (std.mem.indexOf(u8, imatrix_data.keys()[i], name) != null) {
                imatrix_data.orderedRemoveAt(i);
            } else {
                i += 1;
            }
        }
    }
    if (included_weights.len != 0) {
        var tmp: ImatrixData = .empty;
        for (included_weights) |name| {
            for (imatrix_data.keys(), imatrix_data.values()) |k, v| {
                if (std.mem.indexOf(u8, k, name) != null) {
                    // `tmp.emplace(std::move(e))`: the first insertion wins
                    const gop = try tmp.getOrPut(a, k);
                    if (!gop.found_existing) gop.value_ptr.* = v;
                }
            }
        }
        imatrix_data.* = tmp;
    }
    if (imatrix_data.count() != 0) {
        io.printf("%s: have %d importance matrix entries\n", .{ "prepare_imatrix", @as(c_int, @intCast(imatrix_data.count())) });
    }
    return m_last_call;
}

/// Ports `parse_ggml_type` (quantize.cpp:305 @b29c606e2).
fn parseGgmlType(arg: [*:0]const u8) c.ggml_type {
    var i: c_int = 0;
    while (i < c.GGML_TYPE_COUNT) : (i += 1) {
        const t: c.ggml_type = @intCast(i);
        const name = c.ggml_type_name(t);
        if (name != null and striequals(std.mem.span(name), std.mem.span(arg))) return t;
    }
    io.eprintf("\n%s: invalid ggml_type '%s'\n\n", .{ "parse_ggml_type", arg });
    return c.GGML_TYPE_COUNT;
}

/// Ports `parse_tensor_type` (quantize.cpp:317 @b29c606e2).
fn parseTensorType(a: std.mem.Allocator, data: [:0]const u8, tensor_type: *std.ArrayList(TensorTypeOption)) !bool {
    const func = "parse_tensor_type";
    const sep_idx = std.mem.indexOfScalar(u8, data, '=') orelse {
        io.printf("\n%s: malformed tensor type '%s'\n\n", .{ func, data.ptr });
        return false;
    };

    const tn_len = sep_idx;
    if (tn_len == 0) {
        io.printf("\n%s: missing tensor name\n\n", .{func});
        return false;
    }
    if (data.len - sep_idx == 1) {
        io.printf("\n%s: missing quantization type\n\n", .{func});
        return false;
    }

    const tn = try a.dupeSentinel(u8, data[0..tn_len], 0);
    for (tn) |*ch| ch.* = std.ascii.toLower(ch.*);
    const qt = data[sep_idx + 1 ..];
    const opt: TensorTypeOption = .{ .name = tn, .type = parseGgmlType(qt.ptr) };
    try tensor_type.append(a, opt);
    if (opt.type == c.GGML_TYPE_COUNT) {
        io.printf("\n%s: invalid quantization type '%s'\n\n", .{ func, qt.ptr });
        return false;
    }

    return true;
}

extern fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern fn fclose(f: *anyopaque) c_int;
extern fn fgetc(f: *anyopaque) c_int;
extern fn strerror(errnum: c_int) [*:0]const u8;
extern fn atoi(s: [*:0]const u8) c_int;

/// Ports `parse_tensor_type_file` (quantize.cpp:349 @b29c606e2): each
/// white-space-separated word of the file, as `file >> arg` reads them.
fn parseTensorTypeFile(a: std.mem.Allocator, filename: [:0]const u8, tensor_type: *std.ArrayList(TensorTypeOption)) !bool {
    const f = fopen(filename.ptr, "r") orelse {
        io.printf("\n%s: failed to open file '%s': %s\n\n", .{ "parse_tensor_type_file", filename.ptr, strerror(std.c._errno().*) });
        return false;
    };
    defer _ = fclose(f);

    var word: std.ArrayList(u8) = .empty;
    defer word.deinit(a);
    while (true) {
        const ch = fgetc(f);
        const ws = ch == -1 or std.ascii.isWhitespace(@intCast(ch));
        if (!ws) {
            try word.append(a, @intCast(ch));
            continue;
        }
        if (word.items.len != 0) {
            const arg = try a.dupeSentinel(u8, word.items, 0);
            word.clearRetainingCapacity();
            if (!try parseTensorType(a, arg, tensor_type)) return false;
        }
        if (ch == -1) break;
    }

    return true;
}

/// Ports `parse_layer_prune` (quantize.cpp:366 @b29c606e2).
fn parseLayerPrune(a: std.mem.Allocator, data: [:0]const u8, prune_layers: *std.ArrayList(i32)) !bool {
    const func = "parse_layer_prune";
    var block_ids = try common.strings.split(a, data, ',');
    defer block_ids.deinit(a);
    for (block_ids.items) |block_id| {
        const z = try a.dupeSentinel(u8, block_id, 0);
        const id: c_int = switch (common.cxx.stoi(z)) {
            .ok => |v| v,
            else => -1,
        };
        if (id < 0) {
            io.printf("\n%s: invalid layer id '%s'\n\n", .{ func, z.ptr });
            return false;
        }
        try prune_layers.append(a, id);
    }

    std.mem.sort(i32, prune_layers.items, {}, std.sort.asc(i32));
    // `std::unique`, then erase the tail
    var w: usize = 0;
    for (prune_layers.items, 0..) |v, i| {
        if (i == 0 or v != prune_layers.items[w - 1]) {
            prune_layers.items[w] = v;
            w += 1;
        }
    }
    prune_layers.shrinkRetainingCapacity(w);
    return true;
}

/// `std::filesystem::equivalent(a, b, ec)`: both exist and are one file.
fn equivalent(a: [:0]const u8, b: [:0]const u8) bool {
    if (comptime builtin.os.tag == .windows) {
        // Windows compares by volume serial and file index; resolving both
        // paths is the closest the C library offers.
        var ra: [260:0]u8 = undefined;
        var rb: [260:0]u8 = undefined;
        if (_fullpath(&ra, a.ptr, ra.len) == null or _fullpath(&rb, b.ptr, rb.len) == null) return false;
        if (fopen(b.ptr, "rb") == null) return false;
        return std.ascii.eqlIgnoreCase(std.mem.sliceTo(&ra, 0), std.mem.sliceTo(&rb, 0));
    }
    if (comptime builtin.os.tag == .linux) {
        // `std.c.Stat` is void on Linux; `statx` gives the same device and
        // inode, following symlinks as `stat` does.
        const linux = std.os.linux;
        var sa: linux.Statx = undefined;
        var sb: linux.Statx = undefined;
        const mask: linux.STATX = .{ .INO = true };
        if (linux.errno(linux.statx(linux.AT.FDCWD, a.ptr, 0, mask, &sa)) != .SUCCESS or
            linux.errno(linux.statx(linux.AT.FDCWD, b.ptr, 0, mask, &sb)) != .SUCCESS) return false;
        return sa.dev_major == sb.dev_major and sa.dev_minor == sb.dev_minor and sa.ino == sb.ino;
    }
    var sa: std.c.Stat = undefined;
    var sb: std.c.Stat = undefined;
    if (std.c.stat(a.ptr, &sa) != 0 or std.c.stat(b.ptr, &sb) != 0) return false;
    return sa.dev == sb.dev and sa.ino == sb.ino;
}

extern fn _fullpath(abs: [*]u8, rel: [*:0]const u8, max: usize) ?[*:0]u8;

/// The C's `strncpy(dst, src, 127); dst[127] = 0` into a `char[128]`.
fn copyStr128(dst: *[128]u8, src: []const u8) void {
    @memset(dst, 0);
    const n = @min(src.len, 127);
    @memcpy(dst[0..n], src[0..n]);
}

/// Ports `llama_quantize` (quantize.cpp:395 @b29c606e2).
fn quantize(a: std.mem.Allocator, argv: []const [:0]const u8) !u8 {
    const func = "llama_quantize";
    const exe = argv[0].ptr;
    if (argv.len < 3) usage(exe);

    var params = c.llama_model_quantize_default_params();

    var arg_idx: usize = 1;
    var imatrix_file: [:0]const u8 = "";
    var included_weights: std.ArrayList([:0]const u8) = .empty;
    var excluded_weights: std.ArrayList([:0]const u8) = .empty;
    var kv_overrides: std.ArrayList(c.llama_model_kv_override) = .empty;
    var tensor_type_opts: std.ArrayList(TensorTypeOption) = .empty;
    var prune_layers: std.ArrayList(i32) = .empty;

    while (arg_idx < argv.len and std.mem.startsWith(u8, argv[arg_idx], "--")) : (arg_idx += 1) {
        const arg = argv[arg_idx];
        const last = arg_idx == argv.len - 1;
        if (std.mem.eql(u8, arg, "--leave-output-tensor")) {
            params.quantize_output_tensor = false;
        } else if (std.mem.eql(u8, arg, "--output-tensor-type")) {
            if (last) usage(exe);
            arg_idx += 1;
            params.output_tensor_type = parseGgmlType(argv[arg_idx].ptr);
            if (params.output_tensor_type == c.GGML_TYPE_COUNT) usage(exe);
        } else if (std.mem.eql(u8, arg, "--token-embedding-type")) {
            if (last) usage(exe);
            arg_idx += 1;
            params.token_embedding_type = parseGgmlType(argv[arg_idx].ptr);
            if (params.token_embedding_type == c.GGML_TYPE_COUNT) usage(exe);
        } else if (std.mem.eql(u8, arg, "--tensor-type")) {
            if (last) usage(exe);
            arg_idx += 1;
            if (!try parseTensorType(a, argv[arg_idx], &tensor_type_opts)) usage(exe);
        } else if (std.mem.eql(u8, arg, "--tensor-type-file")) {
            if (last) usage(exe);
            arg_idx += 1;
            if (!try parseTensorTypeFile(a, argv[arg_idx], &tensor_type_opts)) usage(exe);
        } else if (std.mem.eql(u8, arg, "--prune-layers")) {
            if (last) usage(exe);
            arg_idx += 1;
            if (!try parseLayerPrune(a, argv[arg_idx], &prune_layers)) usage(exe);
        } else if (std.mem.eql(u8, arg, "--override-kv")) {
            if (last) usage(exe);
            arg_idx += 1;
            if (!try common.strings.parseKvOverride(argv[arg_idx], &kv_overrides, a)) usage(exe);
        } else if (std.mem.eql(u8, arg, "--dry-run")) {
            params.dry_run = true;
        } else if (std.mem.eql(u8, arg, "--allow-requantize")) {
            params.allow_requantize = true;
        } else if (std.mem.eql(u8, arg, "--pure")) {
            params.pure = true;
        } else if (std.mem.eql(u8, arg, "--imatrix")) {
            if (last) usage(exe);
            arg_idx += 1;
            imatrix_file = argv[arg_idx];
        } else if (std.mem.eql(u8, arg, "--include-weights")) {
            if (last) usage(exe);
            arg_idx += 1;
            try included_weights.append(a, argv[arg_idx]);
        } else if (std.mem.eql(u8, arg, "--exclude-weights")) {
            if (last) usage(exe);
            arg_idx += 1;
            try excluded_weights.append(a, argv[arg_idx]);
        } else if (std.mem.eql(u8, arg, "--keep-split")) {
            params.keep_split = true;
        } else if (std.mem.eql(u8, arg, "--max-buffer-size")) {
            if (last) usage(exe);
            arg_idx += 1;
            const mib = atoi(argv[arg_idx].ptr);
            if (mib <= 0) {
                io.eprintf("%s: invalid --max-buffer-size '%s'\n", .{ func, argv[arg_idx].ptr });
                return 1;
            }
            params.max_buf_size = @as(usize, @intCast(mib)) * 1024 * 1024;
        } else {
            usage(exe);
        }
    }

    if (argv.len - arg_idx < 2) {
        io.printf("%s: bad arguments\n", .{exe});
        usage(exe);
    }
    if (included_weights.items.len != 0 and excluded_weights.items.len != 0) usage(exe);

    var imatrix_datasets: std.ArrayList([]const u8) = .empty;
    var imatrix_data: ImatrixData = .empty;
    const m_last_call = try prepareImatrix(a, imatrix_file, &imatrix_datasets, included_weights.items, excluded_weights.items, &imatrix_data);

    var i_data: std.ArrayList(c.llama_model_imatrix_data) = .empty;
    var t_override: std.ArrayList(c.llama_model_tensor_override) = .empty;
    if (imatrix_data.count() != 0) {
        for (imatrix_data.keys(), imatrix_data.values()) |k, v| {
            try i_data.append(a, .{ .name = (try a.dupeSentinel(u8, k, 0)).ptr, .data = v.ptr, .size = v.len });
        }
        try i_data.append(a, .{ .name = null, .data = null, .size = 0 }); // array terminator
        params.imatrix = i_data.items.ptr;
        {
            var kvo = std.mem.zeroes(c.llama_model_kv_override);
            @memcpy(kvo.key[0..kv_imatrix_file.len], kv_imatrix_file);
            kvo.tag = c.LLAMA_KV_OVERRIDE_TYPE_STR;
            copyStr128(&kvo.unnamed_0.val_str, imatrix_file);
            try kv_overrides.append(a, kvo);
        }
        if (imatrix_datasets.items.len != 0) {
            var kvo = std.mem.zeroes(c.llama_model_kv_override);
            // TODO: list multiple datasets when there are more than one
            @memcpy(kvo.key[0..kv_imatrix_dataset.len], kv_imatrix_dataset);
            kvo.tag = c.LLAMA_KV_OVERRIDE_TYPE_STR;
            const d0 = imatrix_datasets.items[0];
            copyStr128(&kvo.unnamed_0.val_str, d0[0 .. std.mem.indexOfScalar(u8, d0, 0) orelse d0.len]);
            try kv_overrides.append(a, kvo);
        }
        {
            var kvo = std.mem.zeroes(c.llama_model_kv_override);
            @memcpy(kvo.key[0..kv_imatrix_n_entries.len], kv_imatrix_n_entries);
            kvo.tag = c.LLAMA_KV_OVERRIDE_TYPE_INT;
            kvo.unnamed_0.val_i64 = @intCast(imatrix_data.count());
            try kv_overrides.append(a, kvo);
        }
        if (m_last_call > 0) {
            var kvo = std.mem.zeroes(c.llama_model_kv_override);
            @memcpy(kvo.key[0..kv_imatrix_n_chunks.len], kv_imatrix_n_chunks);
            kvo.tag = c.LLAMA_KV_OVERRIDE_TYPE_INT;
            kvo.unnamed_0.val_i64 = m_last_call;
            try kv_overrides.append(a, kvo);
        }
    }
    if (kv_overrides.items.len != 0) {
        try kv_overrides.append(a, std.mem.zeroes(c.llama_model_kv_override));
        params.kv_overrides = kv_overrides.items.ptr;
    }
    if (tensor_type_opts.items.len != 0) {
        for (tensor_type_opts.items) |tt| try t_override.append(a, .{ .pattern = tt.name.ptr, .type = tt.type });
        try t_override.append(a, .{ .pattern = null, .type = c.GGML_TYPE_COUNT }); // array terminator
        params.tt_overrides = t_override.items.ptr;
    }
    if (prune_layers.items.len != 0) {
        try prune_layers.append(a, -1); // array terminator
        params.prune_layers = prune_layers.items.ptr;
    }

    c.llama_backend_init();

    // parse command line arguments
    const fname_inp = argv[arg_idx];
    arg_idx += 1;
    var fname_out: []const u8 = "";

    var ftype_str: []const u8 = "";
    const suffix = ".gguf";
    if (try tryParseFtype(a, argv[arg_idx], &params.ftype, &ftype_str)) {
        // argv[arg_idx] is the ftype directly: <input> <ftype>
        if (!params.dry_run) {
            var fpath: []const u8 = "";
            if (std.mem.lastIndexOfAny(u8, fname_inp, "/\\")) |pos| fpath = fname_inp[0 .. pos + 1];

            // export as [inp path]/ggml-model-[ftype]. Only add extension if there is no splitting
            fname_out = try std.mem.concat(a, u8, &.{ fpath, "ggml-model-", ftype_str, if (params.keep_split) "" else suffix });
        }
        arg_idx += 1;
        if (std.mem.eql(u8, ftype_str, "COPY")) params.only_copy = true;
    } else {
        // argv[arg_idx] is not a valid ftype, so treat it as output path: <input> <output> <ftype>
        fname_out = argv[arg_idx];
        if (params.keep_split and std.mem.indexOf(u8, fname_out, suffix) != null) {
            fname_out = fname_out[0 .. fname_out.len - suffix.len];
        }
        arg_idx += 1;

        if (argv.len <= arg_idx) {
            io.eprintf("%s: missing ftype\n", .{func});
            return 1;
        }
        if (!try tryParseFtype(a, argv[arg_idx], &params.ftype, &ftype_str)) {
            io.eprintf("%s: invalid ftype '%s'\n", .{ func, argv[arg_idx].ptr });
            return 1;
        }
        if (std.mem.eql(u8, ftype_str, "COPY")) params.only_copy = true;
        arg_idx += 1;
    }

    // parse nthreads
    if (argv.len > arg_idx) {
        switch (common.cxx.stoi(argv[arg_idx])) {
            .ok => |n| params.nthread = n,
            else => |e| {
                io.eprintf("%s: invalid nthread '%s' (%s)\n", .{ func, argv[arg_idx].ptr, e.what() });
                return 1;
            },
        }
    }

    const out_z = try a.dupeSentinel(u8, fname_out, 0);
    const ftype_z = try a.dupeSentinel(u8, ftype_str, 0);
    if (!params.dry_run) {
        if (equivalent(fname_inp, out_z)) {
            io.eprintf("%s: error: input and output files are the same: '%s'\n", .{ func, fname_inp.ptr });
            return 1;
        }
    }

    common.build_info.print(c.llama_version(), common.io.cStderr());

    if (params.dry_run) {
        io.eprintf("%s: calculating quantization size for '%s' as %s", .{ func, fname_inp.ptr, ftype_z.ptr });
    } else {
        io.eprintf("%s: quantizing '%s' to '%s' as %s", .{ func, fname_inp.ptr, out_z.ptr, ftype_z.ptr });
    }

    if (params.nthread > 0) io.eprintf(" using %d threads", .{params.nthread});
    io.eprintf("\n", .{});

    const t_main_start_us = c.llama_time_us();

    var t_quantize_us: i64 = 0;

    // load the model
    {
        const t_start_us = c.llama_time_us();

        if (c.llama_model_quantize(fname_inp.ptr, out_z.ptr, &params) != 0) {
            io.eprintf("%s: failed to quantize model from '%s'\n", .{ func, fname_inp.ptr });
            return 1;
        }

        t_quantize_us = c.llama_time_us() - t_start_us;
    }

    // report timing
    {
        const t_main_end_us = c.llama_time_us();

        io.printf("\n", .{});
        io.printf("%s: quantize time = %8.2f ms\n", .{ func, @as(f64, @floatFromInt(t_quantize_us)) / 1000.0 });
        io.printf("%s:    total time = %8.2f ms\n", .{ func, @as(f64, @floatFromInt(t_main_end_us - t_main_start_us)) / 1000.0 });
    }

    c.llama_backend_free();

    return 0;
}

/// Ports `main` (tools/quantize/main.cpp:3 @b29c606e2).
pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return quantize(arena, argv);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
