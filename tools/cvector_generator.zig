//! `llama-cvector-generator`: derive a control vector from pairs of opposed
//! prompts.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/cvector-generator/cvector-generator.cpp`,
//! `pca.hpp` and `mean.hpp` at v0.4.0 (`5266f24da`). Each declaration names
//! the C++ it replaces and the line it began at.
//!
//! # The PCA starts from the clock
//!
//! `pca_model` seeds `std::default_random_engine` -- libc++'s `minstd_rand`
//! -- with `std::time(0)`, so upstream's PCA output moves from one second to
//! the next. The engine and `std::uniform_real_distribution<float>` are
//! reproduced exactly (`MinstdRand`), from the same seed; the `mean` method
//! has no randomness, and is what `scripts/tools-diff` compares on bytes.

const std = @import("std");
const tc = @import("tools_common");
const c = tc.c;
const io = tc.io;
const arg = tc.arg;
const backends = @import("backends");

comptime {
    // libllama, compiled in (see `build/tools.zig`).
    _ = @import("llama");
}

/// `GGML_ASSERT`, which upstream keeps in every build mode.
fn ggmlAssert(ok: bool, comptime file: [*:0]const u8, comptime line: c_int, comptime expr: [*:0]const u8) void {
    if (!ok) _ = c.ggml_abort(file, line, "GGML_ASSERT(%s) failed", expr);
}

extern fn malloc(n: usize) ?*anyopaque;
extern fn free(p: ?*anyopaque) void;
extern fn time(t: ?*i64) i64;

/// Ports `tokens_to_str` (cvector-generator.cpp:36 @5266f24da).
fn tokensToStr(a: std.mem.Allocator, vocab: ?*const c.llama_vocab, tokens: []const c.llama_token) ![:0]u8 {
    var ret: std.ArrayList(u8) = .empty;
    for (tokens) |t| try ret.appendSlice(a, try tc.common.tokenToPiece(a, vocab, t, true));
    return a.dupeSentinel(u8, ret.items, 0);
}

/// Ports `print_usage` (cvector-generator.cpp:45 @5266f24da).
fn printUsage(argv: []const [:0]const u8) void {
    const e = argv[0].ptr;
    io.printf("\nexample usage:\n", .{});
    io.printf("\n    CPU only:   %s -m ./llama-3.Q4_K_M.gguf\n", .{e});
    io.printf("\n    with GPU:   %s -m ./llama-3.Q4_K_M.gguf -ngl 99\n", .{e});
    io.printf("\n    advanced:   %s -m ./llama-3.Q4_K_M.gguf -ngl 99 --pca-iter 2000 --pca-batch 100\n", .{e});
    io.printf("\n    using mean: %s -m ./llama-3.Q4_K_M.gguf --method mean\n", .{e});
    io.printf("\n", .{});
}

/// `DEBUG_POS` (pca.hpp:19 @5266f24da).
const debug_pos = 5;

/// Ports `print_debug_tensor` (pca.hpp:21 @5266f24da).
fn printDebugTensor(t: *c.ggml_tensor) void {
    const func = "print_debug_tensor";
    io.printf("%s: %s (%s): [%d, %d]\n", .{ func, @as([*:0]const u8, @ptrCast(&t.name)), c.ggml_type_name(t.type), @as(c_int, @intCast(t.ne[0])), @as(c_int, @intCast(t.ne[1])) });
    io.printf("%s: %s[0] = [", .{ func, @as([*:0]const u8, @ptrCast(&t.name)) });
    var i: c_int = 0;
    while (i <= debug_pos) : (i += 1) io.printf(" %f,", .{@as(f64, c.ggml_get_f32_nd(t, i, 0, 0, 0))});
    io.printf(" ... ]\n", .{});
}

/// Mirrors `callback_data` (cvector-generator.cpp:58 @5266f24da).
const CallbackData = struct {
    a: std.mem.Allocator,
    ctx_ggml: ?*c.ggml_context = null, // holds v_pos, v_neg, v_diff_filtered

    n_layers: c_int = 0,
    n_tokens: c_int = 0,
    is_eval_pos: bool = true,

    // each element of the vector correspond to one layer
    v_pos: std.ArrayList(*c.ggml_tensor) = .empty, // vector of matrices of size [n_embd, n_tokens]
    v_neg: std.ArrayList(*c.ggml_tensor) = .empty, // vector of matrices of size [n_embd, n_tokens]
    v_diff_filtered: std.ArrayList(*c.ggml_tensor) = .empty, // vector of matrices of size [n_embd, n_nonzero_rows]

    /// Ports `save_tensor_for_layer` (cvector-generator.cpp:71 @5266f24da), a member of `callback_data`.
    fn saveTensorForLayer(self: *CallbackData, t: *c.ggml_tensor) !void {
        ggmlAssert(t.type == c.GGML_TYPE_F32, "cvector-generator.cpp", 72, "t->type == GGML_TYPE_F32");

        if (self.ctx_ggml == null) {
            // alloc a new ctx_ggml if needed
            const params_ggml: c.ggml_init_params = .{ .mem_size = c.ggml_tensor_overhead() * @as(usize, @intCast(self.n_layers)) * 3, .mem_buffer = null, .no_alloc = true };
            self.ctx_ggml = c.ggml_init(params_ggml);
        }

        // copy tensor data
        const n_bytes = c.ggml_nbytes(t);
        const t_layer = c.ggml_new_tensor_2d(self.ctx_ggml, t.type, t.ne[0], t.ne[1]);
        t_layer.*.data = malloc(n_bytes);
        c.ggml_backend_tensor_get(t, t_layer.*.data, 0, n_bytes);
        _ = c.ggml_set_name(t_layer, c.ggml_get_name(t));

        if (self.is_eval_pos) try self.v_pos.append(self.a, t_layer) else try self.v_neg.append(self.a, t_layer);
    }

    /// Ports `calc_diff` (cvector-generator.cpp:102 @5266f24da), a member of `callback_data`.
    fn calcDiff(self: *CallbackData) ![]*c.ggml_tensor {
        for (self.v_pos.items, 0..) |vp, il| {
            const a: [*]f32 = @ptrCast(@alignCast(vp.data));
            const b: [*]const f32 = @ptrCast(@alignCast(self.v_neg.items[il].data));
            const n_elem: usize = @intCast(c.ggml_nelements(vp));
            for (0..n_elem) |j| a[j] -= b[j];
            try self.v_diff_filtered.append(self.a, self.filterNonzeroRows(vp));
        }
        return self.v_diff_filtered.items; // for convenient, we return the result std::vector
    }

    /// Ports `filter_nonzero_rows` (cvector-generator.cpp:118 @5266f24da), a member of `callback_data`.
    ///
    /// A row is all zeros unless an element is **greater than** `eps`: a row
    /// of negative numbers counts as zero, as the C counts it.
    fn filterNonzeroRows(self: *CallbackData, a: *c.ggml_tensor) *c.ggml_tensor {
        const eps: f32 = 1e-6;
        var rows_to_copy: std.ArrayList(c_int) = .empty; // the idx of non-zero cols
        var i_row: c_int = 0;
        while (i_row < a.ne[1]) : (i_row += 1) {
            const n_cols: c_int = @intCast(a.ne[0]); // hint: should be equal to n_embd
            var all_zero = true;
            var col: c_int = 0;
            while (col < n_cols) : (col += 1) {
                if (c.ggml_get_f32_nd(a, col, i_row, 0, 0) > eps) {
                    all_zero = false;
                    break;
                }
            }
            if (!all_zero) rows_to_copy.append(self.a, i_row) catch unreachable;
        }

        // get "n_nonzero_rows" for the output "diff_filtered"
        const n_nonzero_rows: c_int = @intCast(rows_to_copy.items.len);
        const n_embd: c_int = @intCast(a.ne[0]);
        ggmlAssert(n_nonzero_rows > 0, "cvector-generator.cpp", 141, "n_nonzero_rows > 0");

        // diff_filtered: [n_embd, n_nonzero_rows]
        const diff_filtered = c.ggml_new_tensor_2d(self.ctx_ggml, c.GGML_TYPE_F32, n_embd, n_nonzero_rows);
        _ = c.ggml_format_name(diff_filtered, "diff_filtered_%s", @as([*:0]const u8, @ptrCast(&a.name)));
        diff_filtered.*.data = malloc(c.ggml_nbytes(diff_filtered));

        // copy non-zero rows
        var dest_row: c_int = 0;
        while (dest_row < n_nonzero_rows) : (dest_row += 1) {
            const src_row = rows_to_copy.items[@intCast(dest_row)];
            var i: c_int = 0;
            while (i < n_embd) : (i += 1) {
                const src_elem = c.ggml_get_f32_nd(a, i, src_row, 0, 0);
                c.ggml_set_f32_nd(diff_filtered, i, dest_row, 0, 0, src_elem);
            }
        }
        return diff_filtered;
    }

    /// Ports `reset` (cvector-generator.cpp:164 @5266f24da), a member of `callback_data`.
    fn reset(self: *CallbackData) void {
        for (self.v_pos.items) |p| free(p.data);
        for (self.v_neg.items) |p| free(p.data);
        for (self.v_diff_filtered.items) |p| free(p.data);
        self.v_pos.clearRetainingCapacity();
        self.v_neg.clearRetainingCapacity();
        self.v_diff_filtered.clearRetainingCapacity();
        if (self.ctx_ggml) |ctx| c.ggml_free(ctx);
        self.ctx_ggml = null;
    }
};

/// Mirrors `train_context` (cvector-generator.cpp:182 @5266f24da).
const TrainContext = struct {
    a: std.mem.Allocator,
    ctx_ggml: ?*c.ggml_context,
    n_embd: c_int,
    n_layers: c_int,

    /// pair of prompts to be used for generating final vector
    positive_entries: std.ArrayList([]const u8) = .empty,
    negative_entries: std.ArrayList([]const u8) = .empty,

    v_diff: std.ArrayList(*c.ggml_tensor) = .empty, // vector of matrices of size [m, n_embd]
    v_final: std.ArrayList(*c.ggml_tensor) = .empty, // vector of vectors of size [n_embd] to be written to file

    v_diff_tmp: std.ArrayList(std.ArrayList(u8)) = .empty,

    fn init(a: std.mem.Allocator, n_embd: c_int, n_layers: c_int) !TrainContext {
        const params_ggml: c.ggml_init_params = .{ .mem_size = c.ggml_tensor_overhead() * @as(usize, @intCast(n_layers - 1)) * 2, .mem_buffer = null, .no_alloc = true };
        var self: TrainContext = .{ .a = a, .ctx_ggml = c.ggml_init(params_ggml), .n_embd = n_embd, .n_layers = n_layers };
        var il: c_int = 0;
        while (il < n_layers - 1) : (il += 1) {
            try self.v_diff_tmp.append(a, .empty);
            const t = c.ggml_new_tensor_1d(self.ctx_ggml, c.GGML_TYPE_F32, n_embd);
            t.*.data = malloc(c.ggml_nbytes(t));
            try self.v_final.append(a, t);
        }
        return self;
    }

    /// Ports `concat_diff_tmp` (cvector-generator.cpp:220 @5266f24da), a member of `train_context`.
    fn concatDiffTmp(self: *TrainContext, diff_filtered: []const *c.ggml_tensor) !void {
        ggmlAssert(@as(c_int, @intCast(diff_filtered.len)) == self.n_layers - 1, "cvector-generator.cpp", 221, "(int) diff_filtered.size() == n_layers - 1");
        for (0..@intCast(self.n_layers - 1)) |il| {
            const t = diff_filtered[il];
            const bytes: [*]const u8 = @ptrCast(t.data);
            try self.v_diff_tmp.items[il].appendSlice(self.a, bytes[0..c.ggml_nbytes(t)]);
        }
    }

    /// Ports `build_v_diff` (cvector-generator.cpp:233 @5266f24da), a member of `train_context`.
    fn buildVDiff(self: *TrainContext, transpose: bool) !void {
        io.printf("build_v_diff\n", .{});
        var il: c_int = 0;
        while (il < self.n_layers - 1) : (il += 1) {
            const diff_tmp = &self.v_diff_tmp.items[@intCast(il)];
            const n_elem: c_int = @intCast(diff_tmp.items.len / @sizeOf(f32));
            ggmlAssert(@rem(n_elem, self.n_embd) == 0, "cvector-generator.cpp", 238, "n_elem % n_embd == 0");
            const n_rows = @divTrunc(n_elem, self.n_embd);
            const diff = if (transpose)
                c.ggml_new_tensor_2d(self.ctx_ggml, c.GGML_TYPE_F32, n_rows, self.n_embd)
            else
                c.ggml_new_tensor_2d(self.ctx_ggml, c.GGML_TYPE_F32, self.n_embd, n_rows);
            const name = try std.fmt.allocPrintSentinel(self.a, "diff_{d}", .{il}, 0);
            _ = c.ggml_set_name(diff, name.ptr);
            diff.*.data = malloc(c.ggml_nbytes(diff));
            if (transpose) {
                // copy data & transpose
                const arr: [*]const f32 = @ptrCast(@alignCast(diff_tmp.items.ptr));
                var ir: c_int = 0;
                while (ir < n_rows) : (ir += 1) {
                    var ic: c_int = 0;
                    while (ic < self.n_embd) : (ic += 1) {
                        const f = arr[@intCast(ir * self.n_embd + ic)];
                        c.ggml_set_f32_nd(diff, ir, ic, 0, 0, f);
                    }
                }
            } else {
                // only copy
                const dst: [*]u8 = @ptrCast(diff.*.data);
                @memcpy(dst[0..c.ggml_nbytes(diff)], diff_tmp.items[0..c.ggml_nbytes(diff)]);
            }
            try self.v_diff.append(self.a, diff);
            printDebugTensor(diff);
            // free memory of diff_tmp
            diff_tmp.clearAndFree(self.a);
        }
    }
};

/// Mirrors `tokenized_prompt` (cvector-generator.cpp:273 @5266f24da).
const TokenizedPrompt = struct {
    tokens_pos: []c.llama_token,
    tokens_neg: []c.llama_token,
    max_seq_len: usize,

    fn init(a: std.mem.Allocator, ctx: ?*c.llama_context, pos: []const u8, neg: []const u8) !TokenizedPrompt {
        const model = c.llama_get_model(ctx);
        const vocab = c.llama_model_get_vocab(model);
        const add_bos = c.llama_vocab_get_add_bos(vocab);
        var self: TokenizedPrompt = .{
            .tokens_pos = try tc.common.tokenize(a, vocab, pos, add_bos, true),
            .tokens_neg = try tc.common.tokenize(a, vocab, neg, add_bos, true),
            .max_seq_len = 0,
        };
        self.max_seq_len = @max(self.tokens_pos.len, self.tokens_neg.len);
        self.tokens_pos = try paddingSeq(a, vocab, self.tokens_pos, self.max_seq_len);
        self.tokens_neg = try paddingSeq(a, vocab, self.tokens_neg, self.max_seq_len);
        return self;
    }

    /// Ports `padding_seq` (cvector-generator.cpp:289 @5266f24da), a member of `tokenized_prompt`.
    fn paddingSeq(a: std.mem.Allocator, vocab: ?*const c.llama_vocab, tokens: []c.llama_token, len: usize) ![]c.llama_token {
        // TODO: customize padding token
        const pad_tokens = try tc.common.tokenize(a, vocab, " ", false, false);
        const pad_tok = pad_tokens[pad_tokens.len - 1];
        var out: std.ArrayList(c.llama_token) = .empty;
        try out.appendSlice(a, tokens);
        while (out.items.len < len) try out.append(a, pad_tok);
        return out.items;
    }
};

/// Ports `ctrlvec_load_prompt_file` (cvector-generator.cpp:308 @5266f24da).
fn loadPromptFile(a: std.mem.Allocator, path: []const u8, skip_empty_lines: bool) !std.ArrayList([]const u8) {
    var output: std.ArrayList([]const u8) = .empty;
    const z = try a.dupeSentinel(u8, path, 0);
    const f = std.c.fopen(z.ptr, "r") orelse {
        io.eprintf("error: unable to open file: %s\n", .{z.ptr});
        tc.cExit(1);
    };
    defer _ = std.c.fclose(f);
    var content: std.ArrayList(u8) = .empty;
    var buf: [65536]u8 = undefined;
    while (true) {
        const n = std.c.fread(&buf, 1, buf.len, f);
        if (n == 0) break;
        try content.appendSlice(a, buf[0..n]);
    }
    var lines = std.mem.splitScalar(u8, content.items, '\n');
    var consumed: usize = 0;
    while (lines.next()) |line| {
        consumed += line.len + 1;
        // `std::getline` yields no last empty line after a final '\n'
        if (line.len == 0 and consumed > content.items.len) break;
        const is_skip = skip_empty_lines and line.len == 0;
        if (!is_skip) try output.append(a, try arg.processEscapes(a, line));
    }
    return output;
}

/// Ports `cb_eval` (cvector-generator.cpp:329 @5266f24da).
fn cbEval(t: [*c]c.ggml_tensor, ask: bool, user_data: ?*anyopaque) callconv(.c) bool {
    const cb_data: *CallbackData = @ptrCast(@alignCast(user_data.?));
    const is_l_out = std.mem.startsWith(u8, std.mem.sliceTo(&t.*.name, 0), "l_out");

    if (ask) return is_l_out;

    if (!is_l_out or t.*.ne[1] != cb_data.n_tokens) return true;

    // save the tensor to current context
    cb_data.saveTensorForLayer(t) catch return false;
    return true;
}

/// Ports `get_hidden_layers` (cvector-generator.cpp:347 @5266f24da).
fn getHiddenLayers(ctx: ?*c.llama_context, tokens: []c.llama_token) bool {
    c.llama_memory_clear(c.llama_get_memory(ctx), true);
    if (c.llama_decode(ctx, c.llama_batch_get_one(tokens.ptr, @intCast(tokens.len))) != 0) {
        io.eprintf("%s : failed to eval\n", .{"get_hidden_layers"});
        return false;
    }
    return true;
}

/// Ports `export_gguf` (cvector-generator.cpp:356 @5266f24da).
fn exportGguf(a: std.mem.Allocator, v_ctrl: []const *c.ggml_tensor, fname: []const u8, model_hint: [*:0]const u8) !void {
    const func = "export_gguf";
    const ctx = c.gguf_init_empty();

    const arch = "controlvector";
    c.gguf_set_val_str(ctx, "general.architecture", arch);
    c.gguf_set_val_str(ctx, arch ++ ".model_hint", model_hint);
    c.gguf_set_val_i32(ctx, arch ++ ".layer_count", @intCast(v_ctrl.len));

    for (v_ctrl) |t| {
        c.gguf_add_tensor(ctx, t);
        printDebugTensor(t);
        io.printf("Added tensor: %s\n", .{@as([*:0]const u8, @ptrCast(&t.name))});
    }

    io.printf("%s: writing file...\n", .{func});
    const z = try a.dupeSentinel(u8, fname, 0);
    _ = c.gguf_write_to_file(ctx, z.ptr, false);
    io.printf("%s: wrote file '%s'\n", .{ func, z.ptr });
    c.gguf_free(ctx);
}

/// Ports `prepare_entries` (cvector-generator.cpp:380 @5266f24da).
fn prepareEntries(a: std.mem.Allocator, params: *tc.params.Params, ctx_train: *TrainContext) !c_int {
    // load prompts
    const positive_prompts = try loadPromptFile(a, params.cvector_positive_file, true);
    const negative_prompts = try loadPromptFile(a, params.cvector_negative_file, true);
    if (positive_prompts.items.len != negative_prompts.items.len) {
        io.eprintf("number of positive and negative prompts must be equal\n", .{});
        return 1;
    }
    if (positive_prompts.items.len == 0) {
        io.eprintf("must provide at least one prompt pair\n", .{});
        return 1;
    }
    ctx_train.positive_entries = positive_prompts;
    ctx_train.negative_entries = negative_prompts;
    return 0;
}

// -----------------------------------------------------------------------------
// PCA (pca.hpp)

/// `std::default_random_engine`, libc++'s `minstd_rand`
/// (`linear_congruential_engine<uint_fast32_t, 48271, 0, 2147483647>`).
const MinstdRand = struct {
    x: u64,

    fn init(s: u32) MinstdRand {
        const m: u64 = 2147483647;
        const v = @as(u64, s) % m;
        return .{ .x = if (v == 0) 1 else v };
    }

    fn next(self: *MinstdRand) u64 {
        self.x = (self.x * 48271) % 2147483647;
        return self.x;
    }

    /// `std::uniform_real_distribution<float>(0, 1)(g)`: libc++'s
    /// `generate_canonical<float, 24>` over a range of 2^31 - 2, one draw,
    /// divided by `float(2147483645) + 1`, which is 2^31.
    fn uniform01(self: *MinstdRand) f32 {
        const sp: f32 = @floatFromInt(self.next() - 1);
        const base: f32 = @as(f32, @floatFromInt(@as(u64, 2147483645))) + 1.0;
        return 1.0 * (sp / base) + 0.0;
    }
};

/// Mirrors `pca_params` (pca.hpp:34 @5266f24da), a member of `PCA`.
const PcaParams = struct {
    n_threads: c_int = 1,
    n_batch: c_int = 20, // number of iterations do to in one batch
    n_iterations: c_int = 1000,
    tolerance: f32 = 1e-7,
    // for debugging
    i_layer: c_int = 0,
    n_layers: c_int = 0,
};

/// Mirrors `pca_result` (pca.hpp:46 @5266f24da), a member of `PCA`.
const PcaResult = struct {
    calculated_square: ?*c.ggml_tensor = null,
    eigenvectors: std.ArrayList(?*c.ggml_tensor) = .empty,
    distances: std.ArrayList(f32) = .empty,
};

extern fn ggml_backend_cuda_init(device: c_int) c.ggml_backend_t;

/// Mirrors `pca_model` (pca.hpp:52 @5266f24da), a member of `PCA`: CUDA where ggml has
/// it, the CPU otherwise -- upstream leaves Metal commented out.
const PcaModel = struct {
    backend: c.ggml_backend_t = null,
    buffer: c.ggml_backend_buffer_t = null,
    ctx: ?*c.ggml_context = null,
    dev_input: *c.ggml_tensor = undefined,
    dev_square: *c.ggml_tensor = undefined,
    dev_eigenvector: *c.ggml_tensor = undefined,

    fn init(a: std.mem.Allocator, t_input: *c.ggml_tensor) !PcaModel {
        var self: PcaModel = .{};
        if (comptime backends.cuda) {
            io.eprintf("%s: using CUDA backend\n", .{"pca_model"});
            self.backend = ggml_backend_cuda_init(0); // init device 0
            if (self.backend == null) io.eprintf("%s: ggml_backend_cuda_init() failed\n", .{"pca_model"});
        }
        // Metal stays off, as upstream's TODO leaves it

        // if there aren't GPU Backends fallback to CPU backend
        if (self.backend == null) self.backend = c.ggml_backend_cpu_init();

        const num_tensors = 4;
        const params: c.ggml_init_params = .{ .mem_size = c.ggml_tensor_overhead() * num_tensors, .mem_buffer = null, .no_alloc = true };
        self.ctx = c.ggml_init(params);

        const n_samples = t_input.ne[0];
        const n_embd = t_input.ne[1];

        self.dev_input = c.ggml_new_tensor_2d(self.ctx, c.GGML_TYPE_F32, n_samples, n_embd);
        self.dev_square = c.ggml_new_tensor_2d(self.ctx, c.GGML_TYPE_F32, n_embd, n_embd);
        self.dev_eigenvector = c.ggml_new_tensor_1d(self.ctx, c.GGML_TYPE_F32, n_embd);

        _ = c.ggml_set_name(self.dev_input, "dev_input");
        _ = c.ggml_set_name(self.dev_square, "dev_square");
        _ = c.ggml_set_name(self.dev_eigenvector, "dev_eigenvector");
        self.buffer = c.ggml_backend_alloc_ctx_tensors(self.ctx, self.backend);
        c.ggml_backend_tensor_set(self.dev_input, t_input.data, 0, c.ggml_nbytes(t_input));

        // initialize eigenvector to random normalized vector
        {
            const n: usize = @intCast(c.ggml_nelements(self.dev_eigenvector));
            const random_vec = try a.alloc(f32, n);
            var generator = MinstdRand.init(@truncate(@as(u64, @bitCast(time(null)))));
            var sum_sqr: f32 = 0.0; // for normalizing random_vec
            for (random_vec) |*v| {
                const f = generator.uniform01();
                sum_sqr = @mulAdd(f32, f, f, sum_sqr);
                v.* = f;
            }
            // normalize it
            const random_vec_norm = @sqrt(sum_sqr);
            for (random_vec) |*v| v.* /= random_vec_norm;
            c.ggml_backend_tensor_set(self.dev_eigenvector, random_vec.ptr, 0, c.ggml_nbytes(self.dev_eigenvector));
        }
        return self;
    }

    fn deinit(self: *PcaModel) void {
        c.ggml_free(self.ctx);
        c.ggml_backend_buffer_free(self.buffer);
        c.ggml_backend_free(self.backend);
    }
};

/// The `static std::vector<uint8_t> buf` the graph is built in.
var piter_buf: ?[]u8 = null;

/// Ports `build_graph_piter` (pca.hpp:134 @5266f24da), a member of `PCA`.
fn buildGraphPiter(params: *const PcaParams, model: *const PcaModel, calc_square: bool) !*c.ggml_cgraph {
    ggmlAssert(params.n_batch > 0, "pca.hpp", 138, "params.n_batch > 0");
    const buf_size = c.ggml_tensor_overhead() * c.GGML_DEFAULT_GRAPH_SIZE + c.ggml_graph_overhead();
    if (piter_buf == null) piter_buf = try std.heap.c_allocator.alloc(u8, buf_size);

    const params0: c.ggml_init_params = .{ .mem_size = buf_size, .mem_buffer = piter_buf.?.ptr, .no_alloc = true };
    // create a temporally context to build the graph
    const ctx0 = c.ggml_init(params0);
    const gf = c.ggml_new_graph(ctx0);

    // turn v_diff_original into square matrix if needed
    var tmp_square: ?*c.ggml_tensor = null;
    if (calc_square) {
        tmp_square = c.ggml_mul_mat(ctx0, model.dev_input, model.dev_input);
        _ = c.ggml_set_name(tmp_square, "tmp_square");
    }

    var old_eigen: ?*c.ggml_tensor = model.dev_eigenvector;
    const input_square = if (calc_square) tmp_square else model.dev_square;

    var i: c_int = 0;
    while (i < params.n_batch) : (i += 1) {
        // b_tensor = square * eigenvector^T
        var b_tensor = c.ggml_mul_mat(ctx0, input_square, old_eigen);
        _ = c.ggml_set_name(b_tensor, "b_tensor");

        // normalize
        b_tensor = c.ggml_div_inplace(ctx0, b_tensor, c.ggml_sqrt_inplace(ctx0, c.ggml_sum_rows(ctx0, c.ggml_sqr(ctx0, b_tensor))));
        _ = c.ggml_format_name(b_tensor, "b_tensor_norm_%d", i);

        // calculate distance(new eigenvector - old eigenvector)
        const new_sub_old = c.ggml_add(ctx0, old_eigen, c.ggml_scale(ctx0, b_tensor, -1));
        const distance = c.ggml_sqrt_inplace(ctx0, c.ggml_sum_rows(ctx0, c.ggml_sqr_inplace(ctx0, new_sub_old)));
        _ = c.ggml_format_name(distance, "distance_%d", i);

        old_eigen = b_tensor;

        // build operations nodes
        c.ggml_build_forward_expand(gf, distance);
    }

    // delete the temporally context used to build the graph
    c.ggml_free(ctx0);
    return gf.?;
}

/// The `extract_i` lambda of `compute_piter` (pca.hpp:194 @5266f24da):
/// `sscanf(str, prefix + "%d", &i)` when `str` starts with `prefix`.
fn extractI(prefix: []const u8, str: []const u8) c_int {
    if (!std.mem.startsWith(u8, str, prefix)) return -1;
    const rest = str[prefix.len..];
    var i: usize = 0;
    while (i < rest.len and std.ascii.isWhitespace(rest[i])) i += 1;
    const st = i;
    if (i < rest.len and (rest[i] == '+' or rest[i] == '-')) i += 1;
    while (i < rest.len and std.ascii.isDigit(rest[i])) i += 1;
    return std.fmt.parseInt(c_int, rest[st..i], 10) catch -1;
}

/// Ports `compute_piter` (pca.hpp:194 @5266f24da), a member of `PCA`.
fn computePiter(a: std.mem.Allocator, params: *const PcaParams, model: *const PcaModel, gf: *c.ggml_cgraph, allocr: c.ggml_gallocr_t, result: *PcaResult) !c.ggml_status {
    // allocate tensors
    _ = c.ggml_gallocr_alloc_graph(allocr, gf);

    if (c.ggml_backend_is_cpu(model.backend)) c.ggml_backend_cpu_set_n_threads(model.backend, params.n_threads);

    const res = c.ggml_backend_graph_compute(model.backend, gf);
    if (res == c.GGML_STATUS_SUCCESS) {
        result.calculated_square = null;
        result.eigenvectors.clearRetainingCapacity();
        result.distances.clearRetainingCapacity();
        try result.eigenvectors.appendNTimes(a, null, @intCast(params.n_batch));
        try result.distances.appendNTimes(a, 0, @intCast(params.n_batch));
        // get output nodes
        var i: c_int = 0;
        while (i < c.ggml_graph_n_nodes(gf)) : (i += 1) {
            const node = c.ggml_graph_node(gf, i);
            const name = std.mem.sliceTo(&node.*.name, 0);
            // find b_tensor (without copying data from device)
            var iter = extractI("b_tensor_norm_", name);
            if (iter > -1) result.eigenvectors.items[@intCast(iter)] = node;
            // find distances, then copy data from device
            iter = extractI("distance_", name);
            if (iter > -1) {
                var d: f32 = 0;
                c.ggml_backend_tensor_get(node, &d, 0, @sizeOf(f32));
                result.distances.items[@intCast(iter)] = d;
            }
            // find tmp_square if it exists (without copying data from device)
            if (std.mem.eql(u8, name, "tmp_square")) result.calculated_square = node;
        }
    }
    return res;
}

/// Ports `power_iteration` (pca.hpp:245 @5266f24da), a member of `PCA`.
fn powerIteration(a: std.mem.Allocator, params: *const PcaParams, input: *c.ggml_tensor, output: *c.ggml_tensor) !void {
    var model = try PcaModel.init(a, input);
    defer model.deinit();

    const allocr = c.ggml_gallocr_new(c.ggml_backend_get_default_buffer_type(model.backend));
    var result: PcaResult = .{};
    var last_eigenvector: ?*c.ggml_tensor = null;

    const n_iters = @divTrunc(params.n_iterations, params.n_batch); // more batch, fewer iterations
    var iter: c_int = 0;
    while (iter < n_iters) : (iter += 1) {
        const calc_square = iter == 0; // only need to calculate square for first iteration
        const gf = try buildGraphPiter(params, &model, calc_square);
        _ = try computePiter(a, params, &model, gf, allocr, &result);

        for (result.distances.items, 0..) |d, k| {
            last_eigenvector = result.eigenvectors.items[k];
            if (d < params.tolerance) break; // done
        }

        if (calc_square) {
            // copy and store the square matrix if needed
            ggmlAssert(result.calculated_square != null, "pca.hpp", 272, "result.calculated_square != NULL");
            c.ggml_backend_tensor_copy(result.calculated_square, model.dev_square);
        }

        // copy last eigen vector and store as input for next iteration
        ggmlAssert(last_eigenvector != null, "pca.hpp", 278, "last_eigenvector != NULL");
        c.ggml_backend_tensor_copy(last_eigenvector, model.dev_eigenvector);

        io.printf("%s: layer %d/%d, iteration: %d / total: %d (batch = %d) ...\n", .{ "power_iteration", params.i_layer + 1, params.n_layers, iter + 1, n_iters, params.n_batch });
    }

    // get output tensor
    ggmlAssert(last_eigenvector != null, "pca.hpp", 287, "last_eigenvector");
    c.ggml_backend_tensor_get(last_eigenvector, output.data, 0, c.ggml_nbytes(last_eigenvector));
    c.ggml_gallocr_free(allocr);
}

/// Ports `run_pca` (pca.hpp:296 @5266f24da), a member of `PCA`.
fn runPca(a: std.mem.Allocator, params: *PcaParams, v_input: []const *c.ggml_tensor, v_output: []const *c.ggml_tensor) !void {
    io.printf("%s: Running PCA...\n", .{"run_pca"});
    for (v_input, 0..) |inp, il| {
        // prepare output vector
        const ctrl_out = v_output[il];
        _ = c.ggml_format_name(ctrl_out, "direction.%zu", il + 1);

        // run power_iteration
        params.i_layer = @intCast(il);
        params.n_layers = @intCast(v_input.len);
        try powerIteration(a, params, inp, ctrl_out);
        io.printf("%s: Done layer %d / %d\n", .{ "run_pca", @as(c_int, @intCast(il + 1)), @as(c_int, @intCast(v_input.len)) });
    }
}

/// Ports `run` (mean.hpp:11 @5266f24da), a member of `mean`.
fn runMean(v_input: []const *c.ggml_tensor, v_output: []const *c.ggml_tensor) void {
    io.printf("%s: Running mean...\n", .{"run"});
    for (v_input, 0..) |t_layer, il| {
        // prepare output vector
        const ctrl_out = v_output[il];
        _ = c.ggml_format_name(ctrl_out, "direction.%zu", il + 1);

        // calculate mean vector
        ggmlAssert(t_layer.ne[0] == ctrl_out.ne[0], "mean.hpp", 22, "t_layer->ne[0] == ctrl_out->ne[0]"); // == n_embd
        var ic: c_int = 0;
        while (ic < t_layer.ne[0]) : (ic += 1) {
            var f: f32 = 0.0;
            var ir: c_int = 0;
            while (ir < t_layer.ne[1]) : (ir += 1) f += c.ggml_get_f32_nd(t_layer, ic, ir, 0, 0);
            f /= @floatFromInt(t_layer.ne[1]);
            c.ggml_set_f32_1d(ctrl_out, ic, f);
        }

        // normalize output vector
        var norm: f32 = 0.0;
        var i: c_int = 0;
        while (i < c.ggml_nelements(ctrl_out)) : (i += 1) {
            const f = c.ggml_get_f32_1d(ctrl_out, i);
            norm = @mulAdd(f32, f, f, norm);
        }
        // `sqrt(norm)` in `<math.h>`: the double overload, narrowed back
        norm = @floatCast(@sqrt(@as(f64, norm)));
        i = 0;
        while (i < c.ggml_nelements(ctrl_out)) : (i += 1) {
            const f = c.ggml_get_f32_1d(ctrl_out, i);
            c.ggml_set_f32_1d(ctrl_out, i, f / norm);
        }

        io.printf("%s: Done layer %d / %d\n", .{ "run", @as(c_int, @intCast(il + 1)), @as(c_int, @intCast(v_input.len)) });
    }
}

// -----------------------------------------------------------------------------

/// Ports `main` (cvector-generator.cpp:397 @5266f24da).
fn run(init: std.process.Init, a: std.mem.Allocator, argv: []const [:0]const u8) !u8 {
    var params = try tc.params.Params.init(a);

    params.out_file = "control_vector.gguf";

    tc.common.init();

    if (!try arg.parse(argv, &params, .cvector_generator, printUsage)) return 1;

    if (@rem(params.n_pca_iterations, params.n_pca_batch) != 0) {
        io.eprintf("PCA iterations must by multiply of PCA batch size\n", .{});
        return 1;
    }

    var cb_data: CallbackData = .{ .a = a };

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = cbEval;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    tc.build_info.print(c.llama_version(), tc.io.cStderr());
    c.llama_backend_init();
    c.llama_numa_init(params.numa);

    // load the model to get hparams
    var llama_init = try tc.init.initFromParams(init.gpa, &params, false);
    defer llama_init.deinit();

    const model = llama_init.model;
    const ctx = llama_init.context;

    const n_layers = c.llama_model_n_layer(model);
    const n_embd = c.llama_model_n_embd(model);

    // get model hint param (a.k.a model arch name)
    var model_hint: [128]u8 = @splat(0);
    _ = c.llama_model_meta_val_str(model, "general.architecture", &model_hint, 128);

    // init train_context
    var ctx_train = try TrainContext.init(a, n_embd, n_layers);

    // load and prepare entries for training
    _ = try prepareEntries(a, &params, &ctx_train);

    // we have to pretokenize everything because otherwise we don't know how much overhead to allocate ctx_diffs_wrapped
    var tokenized_prompts: std.ArrayList(TokenizedPrompt) = .empty;
    var n_total_tokens: usize = 0;
    for (ctx_train.positive_entries.items, 0..) |pos, i| {
        const t = try TokenizedPrompt.init(a, ctx, pos, ctx_train.negative_entries.items[i]);
        n_total_tokens += 2 * t.max_seq_len;
        try tokenized_prompts.append(a, t);
    }

    io.printf("n_total_tokens: %zu\n", .{n_total_tokens});
    io.flushStdout(); // `std::endl`

    const vocab = c.llama_model_get_vocab(model);
    for (0..ctx_train.positive_entries.items.len) |i| {
        var success = false;
        const t = tokenized_prompts.items[i];
        cb_data.n_layers = n_layers;
        cb_data.n_tokens = @intCast(t.max_seq_len);

        io.printf("Evaluating prompt[%d/%d]: \"%s\" - \"%s\" (%d tokens)\n", .{
            @as(c_int, @intCast(i + 1)),                   @as(c_int, @intCast(ctx_train.positive_entries.items.len)),
            (try tokensToStr(a, vocab, t.tokens_pos)).ptr, (try tokensToStr(a, vocab, t.tokens_neg)).ptr,
            @as(c_int, @intCast(t.max_seq_len)),
        });

        cb_data.is_eval_pos = true;
        success = getHiddenLayers(ctx, t.tokens_pos);
        if (!success) break;

        cb_data.is_eval_pos = false;
        success = getHiddenLayers(ctx, t.tokens_neg);
        if (!success) break;

        // calculate diff and remove all zero rows
        const v_diff_filtered = try cb_data.calcDiff();

        // save & concat the filtered v_diff to ctx_train
        try ctx_train.concatDiffTmp(v_diff_filtered);

        // reset for next iteration
        cb_data.reset();
    }

    // done with the model, we can now free it to make gain some memory
    io.printf("Done evaluate prompts, unload model...\n", .{});

    const use_pca = params.cvector_dimre_method == .pca;

    // prepare ctx_train for PCA
    try ctx_train.buildVDiff(use_pca);

    if (use_pca) {
        // run PCA
        var pca_params: PcaParams = .{};
        pca_params.n_threads = params.cpuparams.n_threads;
        pca_params.n_batch = params.n_pca_batch;
        pca_params.n_iterations = params.n_pca_iterations;
        try runPca(a, &pca_params, ctx_train.v_diff.items, ctx_train.v_final.items);
    } else {
        // run mean
        runMean(ctx_train.v_diff.items, ctx_train.v_final.items);
    }

    // write output vectors to gguf
    try exportGguf(a, ctx_train.v_final.items, params.out_file, @ptrCast(&model_hint));

    c.llama_backend_free();

    return 0;
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return run(init, arena, argv);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
