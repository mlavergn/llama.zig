//! `llama-export-lora`: merge LoRA adapters into a model and write the result.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/export-lora/export-lora.cpp` at v1.0.0
//! (`b1b770d90`). Each declaration names the C++ it replaces and the line it
//! began at.
//!
//! # Order
//!
//! `file_input::tensors` is a `std::map`, so the base model's tensors are
//! merged, and written, in their names' bytewise order -- not the file's.
//! `Input.names` keeps that order.
//!
//! # When the output file appears
//!
//! The C's `fout` is a member constructed after the base model, before the
//! adapters: an adapter that fails to load leaves an empty output file
//! behind. So does this.

const std = @import("std");
const llama = @import("llama");
const stdlib = @import("stdlib");
const tools_common = @import("tools_common");

/// Ports `g_verbose` (export-lora.cpp:14 @b1b770d90).
var g_verbose = false;

/// A thrown `std::exception`: `main` prints `what()` and exits 1.
const Throw = error{Throw};

fn throw(comptime fmt: []const u8, args: anytype) Throw {
    var buf: [1024]u8 = undefined;
    return tools_common.arg.throwStr(std.fmt.bufPrint(&buf, fmt, args) catch "error");
}

/// Mirrors `tensor_transformation` (export-lora.cpp:16 @b1b770d90).
const TensorTransformation = struct {
    in: *llama.impl.llama_c.ggml_tensor,
    out: *llama.impl.llama_c.ggml_tensor,
    is_copy: bool,
};

/// Ports `get_kv_str` (export-lora.cpp:22 @b1b770d90).
fn getKvStr(ctx_gguf: ?*llama.impl.llama_c.gguf_context, key: [*:0]const u8) []const u8 {
    const id = llama.impl.llama_c.gguf_find_key(ctx_gguf, key);
    return if (id < 0) "" else std.mem.span(llama.impl.llama_c.gguf_get_val_str(ctx_gguf, id));
}

/// Ports `get_kv_f32` (export-lora.cpp:27 @b1b770d90).
fn getKvF32(ctx_gguf: ?*llama.impl.llama_c.gguf_context, key: [*:0]const u8) f32 {
    const id: c_int = @truncate(llama.impl.llama_c.gguf_find_key(ctx_gguf, key));
    return if (id < 0) 0.0 else llama.impl.llama_c.gguf_get_val_f32(ctx_gguf, id);
}

/// `fout`'s `failbit` exception, which a failed write raises as a failed
/// open does, to `main`, which prints it and exits 1.
fn writeFailed() Throw {
    return throw("ios_base::clear: unspecified iostream_category error", .{});
}

/// Ports `zeros` (export-lora.cpp:32 @b1b770d90).
fn zeros(f: *stdlib.file.Out, n: usize) !void {
    f.zeros(n) catch return writeFailed();
}

/// Ports `ggml_ne_string` (export-lora.cpp:39 @b1b770d90).
fn neString(a: std.mem.Allocator, t: *const llama.impl.llama_c.ggml_tensor) ![:0]u8 {
    var s: std.ArrayList(u8) = .empty;
    for (0..llama.impl.llama_c.GGML_MAX_DIMS) |i| {
        try s.print(a, "{d}", .{t.ne[i]});
        if (i + 1 < llama.impl.llama_c.GGML_MAX_DIMS) try s.appendSlice(a, ", ");
    }
    return a.dupeSentinel(u8, s.items, 0);
}

/// `GGML_PAD(x, GGUF_DEFAULT_ALIGNMENT)`.
fn pad(x: usize) usize {
    return stdlib.num.pad(usize, x, llama.impl.llama_c.GGUF_DEFAULT_ALIGNMENT);
}

/// Ports `load_gguf` (export-lora.cpp:50 @b1b770d90).
fn loadGguf(fname: [:0]const u8, ctx_ggml: *?*llama.impl.llama_c.ggml_context) !*llama.impl.llama_c.gguf_context {
    const params: llama.impl.llama_c.gguf_init_params = .{ .no_alloc = true, .ctx = ctx_ggml };
    return llama.impl.llama_c.gguf_init_from_file(fname.ptr, params) orelse throw("failed to load input GGUF from {s}", .{fname});
}

/// Ports `file_input` (export-lora.cpp:62 @b1b770d90).
const Input = struct {
    ctx_meta: ?*llama.impl.llama_c.ggml_context = null,
    ctx_gguf: ?*llama.impl.llama_c.gguf_context = null,
    io: std.Io,
    f_in: std.Io.File,
    /// `tensors`, by name; `names` walks it in `std::map` order.
    tensors: std.StringHashMapUnmanaged(*llama.impl.llama_c.ggml_tensor) = .empty,
    names: std.ArrayList([]const u8) = .empty,
    alpha: f32 = 0,
    scale: f32,

    fn init(a: std.mem.Allocator, std_io: std.Io, fname: [:0]const u8, scale: f32) !Input {
        const f_in = stdlib.file.openIn(std_io, fname) catch return throw("failed to open input gguf from {s}", .{fname});
        var self: Input = .{ .io = std_io, .f_in = f_in, .scale = scale };

        self.ctx_gguf = try loadGguf(fname, &self.ctx_meta);
        self.alpha = getKvF32(self.ctx_gguf, "adapter.lora.alpha");
        tools_common.io.printf("%s: loaded gguf from %s\n", .{ "file_input", fname.ptr });

        var cur = llama.impl.llama_c.ggml_get_first_tensor(self.ctx_meta);
        while (cur) |t| : (cur = llama.impl.llama_c.ggml_get_next_tensor(self.ctx_meta, t)) {
            const name = try a.dupe(u8, std.mem.sliceTo(&t.*.name, 0));
            const gop = try self.tensors.getOrPut(a, name);
            if (!gop.found_existing) try self.names.append(a, name);
            gop.value_ptr.* = t;
            if (g_verbose) tools_common.io.printf("%s: %s\n", .{ "file_input", @as([*:0]const u8, @ptrCast(&t.*.name)) });
        }
        std.mem.sort([]const u8, self.names.items, {}, struct {
            fn lt(_: void, x: []const u8, y: []const u8) bool {
                return std.mem.order(u8, x, y) == .lt;
            }
        }.lt);
        return self;
    }

    /// Ports `get_tensor` (export-lora.cpp:88 @b1b770d90), a member of `file_input`.
    fn getTensor(self: *const Input, name: []const u8) ?*llama.impl.llama_c.ggml_tensor {
        return self.tensors.get(name);
    }

    /// Ports `read_tensor_data` (export-lora.cpp:95 @b1b770d90), a member of `file_input`.
    fn readTensorData(self: *Input, a: std.mem.Allocator, name: []const u8, buf: *std.ArrayList(u8)) !void {
        const t = self.tensors.get(name) orelse return throw("cannot find tensor with name: {s}", .{name});
        const len = llama.impl.llama_c.ggml_nbytes(t);
        if (buf.items.len < len) try buf.resize(a, len);
        const zname = try a.dupeSentinel(u8, name, 0);
        const i_tensor_in = llama.impl.llama_c.gguf_find_tensor(self.ctx_gguf, zname.ptr); // idx of tensor in the input file
        const offset = llama.impl.llama_c.gguf_get_data_offset(self.ctx_gguf) + llama.impl.llama_c.gguf_get_tensor_offset(self.ctx_gguf, i_tensor_in);
        if (!stdlib.file.readAt(self.io, self.f_in, buf.items[0..len], offset)) {
            return throw("failed to read data for tensor {s}, input file may be truncated", .{name});
        }
    }

    fn deinit(self: *Input) void {
        llama.impl.llama_c.gguf_free(self.ctx_gguf);
        llama.impl.llama_c.ggml_free(self.ctx_meta);
    }
};

/// Ports `lora_merge_ctx` (export-lora.cpp:118 @b1b770d90).
const MergeCtx = struct {
    a: std.mem.Allocator,
    // input base model + adapters
    base_model: Input,
    adapters: std.ArrayList(*Input) = .empty,

    // for computing merged tensor
    n_threads: c_int,
    backend: llama.impl.llama_c.ggml_backend_t = null,
    allocr: llama.impl.llama_c.ggml_gallocr_t = null,
    read_buf: std.ArrayList(u8) = .empty,

    // output file
    ctx_out: ?*llama.impl.llama_c.gguf_context = null,
    ctx_out_ggml: ?*llama.impl.llama_c.ggml_context = null,
    io: std.Io,
    fout: stdlib.file.Out = undefined,

    fn init(a: std.mem.Allocator, std_io: std.Io, base_fname: [:0]const u8, lora_files: []const tools_common.params.LoraInfo, outfile: [:0]const u8, n_threads: c_int) !MergeCtx {
        var self: MergeCtx = .{ .a = a, .io = std_io, .base_model = try Input.init(a, std_io, base_fname, 0), .n_threads = n_threads };
        // `fout(outfile, std::ios::binary)` and `fout.exceptions(failbit)`:
        // a stream that did not open throws at once
        self.fout = stdlib.file.Out.create(std_io, outfile) catch return throw("ios_base::clear: unspecified iostream_category error", .{});

        if (llama.impl.llama_c.gguf_find_key(self.base_model.ctx_gguf, tools_common.kv_split_count) >= 0) {
            return throw("split model is not yet supported", .{});
        }

        for (lora_files) |lora_inp| {
            const adapter = try a.create(Input);
            adapter.* = try Input.init(a, std_io, lora_inp.path, lora_inp.scale);
            try self.checkMetadataLora(adapter);
            try self.adapters.append(a, adapter);
        }

        self.ctx_out = llama.impl.llama_c.gguf_init_empty();
        const params: llama.impl.llama_c.ggml_init_params = .{
            .mem_size = @as(usize, @intCast(llama.impl.llama_c.gguf_get_n_tensors(self.base_model.ctx_gguf))) * llama.impl.llama_c.ggml_tensor_overhead(),
            .mem_buffer = null,
            .no_alloc = true,
        };
        self.ctx_out_ggml = llama.impl.llama_c.ggml_init(params);
        self.backend = llama.impl.llama_c.ggml_backend_cpu_init();
        self.allocr = llama.impl.llama_c.ggml_gallocr_new(llama.impl.llama_c.ggml_backend_get_default_buffer_type(self.backend));
        return self;
    }

    /// Ports `check_metadata_lora` (export-lora.cpp:164 @b1b770d90), a member of `lora_merge_ctx`.
    fn checkMetadataLora(self: *MergeCtx, adapter: *Input) !void {
        const general_type = getKvStr(adapter.ctx_gguf, "general.type");
        if (!std.mem.eql(u8, general_type, "adapter")) return throw("expect general.type to be 'adapter', but got: {s}", .{general_type});

        const adapter_type = getKvStr(adapter.ctx_gguf, "adapter.type");
        if (!std.mem.eql(u8, adapter_type, "lora")) return throw("expect adapter.type to be 'lora', but got: {s}", .{adapter_type});

        const general_arch_base = getKvStr(self.base_model.ctx_gguf, "general.architecture");
        const general_arch_lora = getKvStr(adapter.ctx_gguf, "general.architecture");
        if (!std.mem.eql(u8, general_arch_base, general_arch_lora)) return throw("model arch and LoRA arch mismatch", .{});
    }

    /// Ports `get_out_tensor_type` (export-lora.cpp:182 @b1b770d90), a member of `lora_merge_ctx`.
    fn getOutTensorType(t: *const llama.impl.llama_c.ggml_tensor) llama.impl.llama_c.ggml_type {
        if (t.type == llama.impl.llama_c.GGML_TYPE_F32) {
            return llama.impl.llama_c.GGML_TYPE_F32;
        } else if (t.type == llama.impl.llama_c.GGML_TYPE_BF16) {
            return llama.impl.llama_c.GGML_TYPE_BF16;
        } else {
            return llama.impl.llama_c.GGML_TYPE_F16;
        }
    }

    /// Ports `run_merge` (export-lora.cpp:192 @b1b770d90), a member of `lora_merge_ctx`.
    fn runMerge(self: *MergeCtx) !void {
        const a = self.a;
        // prepare metadata
        llama.impl.llama_c.gguf_set_kv(self.ctx_out, self.base_model.ctx_gguf);
        // output is forced to f16 for now
        llama.impl.llama_c.gguf_set_val_u32(self.ctx_out, "general.file_type", llama.impl.llama_c.LLAMA_FTYPE_MOSTLY_F16);

        // check if all lora adapters have the same tensors
        const err_no_subset_adapter = "Input adapters do not have the same list of tensors. This is not yet supported. Please merge the adapter one-by-one instead of merging all at once.";
        if (self.adapters.items.len > 1) {
            for (self.adapters.items[1..]) |ad| {
                if (self.adapters.items[0].tensors.count() != ad.tensors.count()) return throw(err_no_subset_adapter, .{});
                for (ad.names.items) |n| {
                    if (self.adapters.items[0].getTensor(n) == null) return throw(err_no_subset_adapter, .{});
                }
            }
        }

        // mapping base tensor to out tensor (same shape with base, but different type)
        var trans: std.ArrayList(TensorTransformation) = .empty;
        for (self.base_model.names.items) |name| {
            var t_a = true;
            var t_b = true;
            for (self.adapters.items) |ad| {
                t_a = t_a and ad.getTensor(try std.mem.concat(a, u8, &.{ name, ".lora_a" })) != null;
                t_b = t_b and ad.getTensor(try std.mem.concat(a, u8, &.{ name, ".lora_b" })) != null;
            }
            const base_tensor = self.base_model.getTensor(name).?;
            if (!t_a and !t_b) {
                // only copy
                const cpy_tensor = llama.impl.llama_c.ggml_dup_tensor(self.ctx_out_ggml, base_tensor);
                _ = llama.impl.llama_c.ggml_set_name(cpy_tensor, &base_tensor.name);
                try trans.append(a, .{ .in = cpy_tensor, .out = cpy_tensor, .is_copy = true });
                llama.impl.llama_c.gguf_add_tensor(self.ctx_out, cpy_tensor);
            } else if (t_a and t_b) {
                // need merging
                const out_tensor = llama.impl.llama_c.ggml_new_tensor(self.ctx_out_ggml, getOutTensorType(base_tensor), llama.impl.llama_c.GGML_MAX_DIMS, &base_tensor.ne);
                _ = llama.impl.llama_c.ggml_set_name(out_tensor, &base_tensor.name);
                try trans.append(a, .{ .in = base_tensor, .out = out_tensor, .is_copy = false });
                llama.impl.llama_c.gguf_add_tensor(self.ctx_out, out_tensor);
            } else {
                return throw("tensor {s} missing either lora_a or lora_b", .{name});
            }
        }

        // placeholder for the meta data
        try zeros(&self.fout, llama.impl.llama_c.gguf_get_meta_size(self.ctx_out));

        // process base model tensors
        var n_merged: usize = 0;
        for (trans.items) |it| {
            if (!it.is_copy) {
                try self.mergeTensor(it.in, it.out);
                n_merged += 1;
            } else {
                try self.copyTensor(it.in);
            }
        }

        // write output metadata
        {
            const data = try a.alloc(u8, llama.impl.llama_c.gguf_get_meta_size(self.ctx_out));
            llama.impl.llama_c.gguf_get_meta_data(self.ctx_out, data.ptr);
            self.fout.seek(0) catch return writeFailed();
            self.fout.write(data) catch return writeFailed();
        }

        tools_common.io.printf("%s : merged %zu tensors with lora adapters\n", .{ "run_merge", n_merged });
        tools_common.io.printf("%s : wrote %zu tensors to output file\n", .{ "run_merge", trans.items.len });
    }

    /// Ports `copy_tensor` (export-lora.cpp:279 @b1b770d90), a member of `lora_merge_ctx`.
    fn copyTensor(self: *MergeCtx, base: *llama.impl.llama_c.ggml_tensor) !void {
        tools_common.io.printf("%s :  %s [%s]\n", .{ "copy_tensor", @as([*:0]const u8, @ptrCast(&base.name)), (try neString(self.a, base)).ptr });
        const len = llama.impl.llama_c.ggml_nbytes(base);
        try self.base_model.readTensorData(self.a, std.mem.sliceTo(&base.name, 0), &self.read_buf);
        self.fout.write(self.read_buf.items[0..len]) catch return writeFailed();
        try zeros(&self.fout, pad(len) - len);
    }

    /// Ports `merge_tensor` (export-lora.cpp:287 @b1b770d90), a member of `lora_merge_ctx`.
    fn mergeTensor(self: *MergeCtx, base: *llama.impl.llama_c.ggml_tensor, out: *llama.impl.llama_c.ggml_tensor) !void {
        const a = self.a;
        const func = "merge_tensor";
        const name_base = std.mem.sliceTo(&base.name, 0);
        const name_lora_a = try std.mem.concat(a, u8, &.{ name_base, ".lora_a" });
        const name_lora_b = try std.mem.concat(a, u8, &.{ name_base, ".lora_b" });

        tools_common.io.printf("%s : %s [%s]\n", .{ func, @as([*:0]const u8, @ptrCast(&base.name)), (try neString(a, base)).ptr });

        // context for input tensor
        const n_ad = self.adapters.items.len;
        const inp_a = try a.alloc(*llama.impl.llama_c.ggml_tensor, n_ad);
        const inp_b = try a.alloc(*llama.impl.llama_c.ggml_tensor, n_ad);
        const params: llama.impl.llama_c.ggml_init_params = .{ .mem_size = llama.impl.llama_c.ggml_tensor_overhead() * (2 + n_ad * 2), .mem_buffer = null, .no_alloc = true };
        const ctx = llama.impl.llama_c.ggml_init(params);

        // alloc tensors
        const inp_base = llama.impl.llama_c.ggml_new_tensor(ctx, llama.impl.llama_c.GGML_TYPE_F32, llama.impl.llama_c.GGML_MAX_DIMS, &base.ne);
        for (self.adapters.items, 0..) |ad, i| {
            const t_a = ad.getTensor(name_lora_a).?;
            const t_b = ad.getTensor(name_lora_b).?;
            // TODO: add support for quantized lora
            if (llama.impl.llama_c.ggml_is_quantized(t_a.type) or llama.impl.llama_c.ggml_is_quantized(t_b.type)) {
                return throw("quantized LoRA adapters is not supported, please retry with f16 or f32", .{});
            }
            inp_a[i] = llama.impl.llama_c.ggml_dup_tensor(ctx, t_a);
            inp_b[i] = llama.impl.llama_c.ggml_dup_tensor(ctx, t_b);
        }
        const buffer = llama.impl.llama_c.ggml_backend_alloc_ctx_tensors(ctx, self.backend);

        // load base tensor to backend buffer
        try self.base_model.readTensorData(a, name_base, &self.read_buf);
        if (base.type != llama.impl.llama_c.GGML_TYPE_F32) {
            // optionally dequantize it
            tools_common.io.printf("%s :   + dequantize base tensor from %s to F32\n", .{ func, llama.impl.llama_c.ggml_type_name(base.type) });
            const nels: usize = @intCast(llama.impl.llama_c.ggml_nelements(inp_base));
            const qtype = llama.impl.llama_c.ggml_get_type_traits(base.type);
            const dequant_buf = try a.alloc(f32, nels);
            qtype.*.to_float.?(self.read_buf.items.ptr, dequant_buf.ptr, @intCast(nels));
            llama.impl.llama_c.ggml_backend_tensor_set(inp_base, dequant_buf.ptr, 0, nels * @sizeOf(f32));
        } else {
            llama.impl.llama_c.ggml_backend_tensor_set(inp_base, self.read_buf.items.ptr, 0, llama.impl.llama_c.ggml_nbytes(inp_base));
        }

        // load lora tensors to backend buffer
        for (self.adapters.items, 0..) |ad, i| {
            try ad.readTensorData(a, name_lora_a, &self.read_buf);
            llama.impl.llama_c.ggml_backend_tensor_set(inp_a[i], self.read_buf.items.ptr, 0, llama.impl.llama_c.ggml_nbytes(inp_a[i]));
            try ad.readTensorData(a, name_lora_b, &self.read_buf);
            llama.impl.llama_c.ggml_backend_tensor_set(inp_b[i], self.read_buf.items.ptr, 0, llama.impl.llama_c.ggml_nbytes(inp_b[i]));
        }

        // build graph
        var gf: ?*llama.impl.llama_c.ggml_cgraph = null;
        {
            const buf_size = llama.impl.llama_c.ggml_tensor_overhead() * llama.impl.llama_c.GGML_DEFAULT_GRAPH_SIZE + llama.impl.llama_c.ggml_graph_overhead();
            if (graph_buf == null) graph_buf = try std.heap.c_allocator.alloc(u8, buf_size);
            const params0: llama.impl.llama_c.ggml_init_params = .{ .mem_size = buf_size, .mem_buffer = graph_buf.?.ptr, .no_alloc = true };
            const ctx0 = llama.impl.llama_c.ggml_init(params0);
            gf = llama.impl.llama_c.ggml_new_graph(ctx0);
            var cur = inp_base;
            for (self.adapters.items, 0..) |ad, i| {
                var delta: ?*llama.impl.llama_c.ggml_tensor = null;
                const is_tok_embd = std.mem.startsWith(u8, name_base, "token_embd");
                if (is_tok_embd) {
                    tools_common.io.printf("%s :     detected token embeddings tensor\n", .{func});
                    delta = llama.impl.llama_c.ggml_mul_mat(ctx0, llama.impl.llama_c.ggml_cast(ctx0, inp_b[i], llama.impl.llama_c.GGML_TYPE_F32), llama.impl.llama_c.ggml_cast(ctx0, inp_a[i], llama.impl.llama_c.GGML_TYPE_F32));
                } else {
                    delta = llama.impl.llama_c.ggml_mul_mat(ctx0, llama.impl.llama_c.ggml_cont(ctx0, llama.impl.llama_c.ggml_transpose(ctx0, llama.impl.llama_c.ggml_cast(ctx0, inp_a[i], llama.impl.llama_c.GGML_TYPE_F32))), llama.impl.llama_c.ggml_cast(ctx0, inp_b[i], llama.impl.llama_c.GGML_TYPE_F32));
                }
                // scale
                const alpha = ad.alpha;
                const rank: f32 = @floatFromInt(inp_b[i].ne[0]);
                const scale = if (alpha != 0) ad.scale * alpha / rank else ad.scale;
                delta = llama.impl.llama_c.ggml_scale(ctx0, delta, scale);
                cur = llama.impl.llama_c.ggml_add(ctx0, delta, cur);
                tools_common.io.printf("%s :   + merging from adapter[%zu] type=%s\n", .{ func, i, llama.impl.llama_c.ggml_type_name(inp_a[i].type) });
                tools_common.io.printf("%s :     input_scale=%f calculated_scale=%f rank=%d\n", .{ func, @as(f64, ad.scale), @as(f64, scale), @as(c_int, @intCast(inp_b[i].ne[0])) });
            }
            cur = llama.impl.llama_c.ggml_cast(ctx0, cur, out.type);
            tools_common.io.printf("%s :   + output type is %s\n", .{ func, llama.impl.llama_c.ggml_type_name(out.type) });
            llama.impl.llama_c.ggml_build_forward_expand(gf, cur);
            llama.impl.llama_c.ggml_free(ctx0);
        }

        // compute
        _ = llama.impl.llama_c.ggml_gallocr_alloc_graph(self.allocr, gf);
        llama.impl.llama_c.ggml_backend_cpu_set_n_threads(self.backend, self.n_threads);
        _ = llama.impl.llama_c.ggml_backend_graph_compute(self.backend, gf);

        // write data to output file
        {
            const result = llama.impl.llama_c.ggml_graph_node(gf, -1);
            const len = llama.impl.llama_c.ggml_nbytes(result);
            if (self.read_buf.items.len < len) try self.read_buf.resize(a, len);
            llama.impl.llama_c.ggml_backend_tensor_get(result, self.read_buf.items.ptr, 0, len);
            self.fout.write(self.read_buf.items[0..len]) catch return writeFailed();
            try zeros(&self.fout, pad(len) - len);
        }

        llama.impl.llama_c.ggml_free(ctx);
        llama.impl.llama_c.ggml_backend_buffer_free(buffer);
    }

    fn deinit(self: *MergeCtx) void {
        llama.impl.llama_c.ggml_gallocr_free(self.allocr);
        llama.impl.llama_c.ggml_backend_free(self.backend);
        llama.impl.llama_c.gguf_free(self.ctx_out);
        llama.impl.llama_c.ggml_free(self.ctx_out_ggml);
        self.fout.close(self.io);
    }
};

/// The `static std::vector<uint8_t> buf` the graph is built in.
var graph_buf: ?[]u8 = null;

/// Ports `print_usage` (export-lora.cpp:412 @b1b770d90).
fn printUsage(argv: []const [:0]const u8) void {
    tools_common.io.printf("\nexample usage:\n", .{});
    tools_common.io.printf("\n  %s -m base-model.gguf --lora lora-file.gguf -o merged-model.gguf\n", .{argv[0].ptr});
    tools_common.io.printf("\n", .{});
}

/// Ports `main` (export-lora.cpp:418 @b1b770d90).
fn run(a: std.mem.Allocator, std_io: std.Io, argv: []const [:0]const u8) !u8 {
    var params = try tools_common.params.Params.init(a);

    params.out_file = "ggml-lora-merged-f16.gguf";

    tools_common.common.init();

    if (!try tools_common.arg.parse(argv, &params, .export_lora, printUsage)) return 1;

    g_verbose = params.verbosity > 1;
    const out_file = try a.dupeSentinel(u8, params.out_file, 0);
    blk: {
        var ctx = MergeCtx.init(a, std_io, try a.dupeSentinel(u8, params.model.path, 0), params.lora_adapters.items, out_file, params.cpuparams.n_threads) catch |e| {
            if (e == error.Throw) break :blk;
            return e;
        };
        defer ctx.deinit();
        ctx.runMerge() catch |e| {
            if (e == error.Throw) break :blk;
            return e;
        };
        tools_common.io.printf("done, output file is %s\n", .{out_file.ptr});
        return 0;
    }
    tools_common.io.eprintf("%.*s\n", .{ @as(c_int, @intCast(tools_common.arg.what().len)), tools_common.arg.what().ptr });
    return 1;
}

pub fn main(init: std.process.Init) !u8 {
    stdlib.stdio.bufferAsC();
    stdlib.env.set(init.environ_map);
    defer stdlib.stdio.flushAll();
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return run(arena, init.io, argv);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
