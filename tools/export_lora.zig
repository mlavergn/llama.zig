//! `llama-export-lora`: merge LoRA adapters into a model and write the result.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/export-lora/export-lora.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration names the C++ it replaces and the line it
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
const tc = @import("tools_common");
const c = tc.c;
const io = tc.io;
const arg = tc.arg;

comptime {
    // libllama, compiled in (see `build/tools.zig`).
    _ = @import("llama");
}

/// Ports `g_verbose` (export-lora.cpp:14 @c1d0e7a00).
var g_verbose = false;

/// A thrown `std::exception`: `main` prints `what()` and exits 1.
const Throw = error{Throw};

fn throw(comptime fmt: []const u8, args: anytype) Throw {
    var buf: [1024]u8 = undefined;
    return arg.throwStr(std.fmt.bufPrint(&buf, fmt, args) catch "error");
}

/// Mirrors `tensor_transformation` (export-lora.cpp:16 @c1d0e7a00).
const TensorTransformation = struct {
    in: *c.ggml_tensor,
    out: *c.ggml_tensor,
    is_copy: bool,
};

/// Ports `get_kv_str` (export-lora.cpp:22 @c1d0e7a00).
fn getKvStr(ctx_gguf: ?*c.gguf_context, key: [*:0]const u8) []const u8 {
    const id = c.gguf_find_key(ctx_gguf, key);
    return if (id < 0) "" else std.mem.span(c.gguf_get_val_str(ctx_gguf, id));
}

/// Ports `get_kv_f32` (export-lora.cpp:27 @c1d0e7a00).
fn getKvF32(ctx_gguf: ?*c.gguf_context, key: [*:0]const u8) f32 {
    const id: c_int = @truncate(c.gguf_find_key(ctx_gguf, key));
    return if (id < 0) 0.0 else c.gguf_get_val_f32(ctx_gguf, id);
}

extern fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern fn fclose(f: *anyopaque) c_int;
extern fn fread(ptr: [*]u8, size: usize, n: usize, f: *anyopaque) usize;
extern fn fwrite(ptr: [*]const u8, size: usize, n: usize, f: *anyopaque) usize;
extern fn fseeko(f: *anyopaque, off: i64, whence: c_int) c_int;
extern fn _fseeki64(f: *anyopaque, off: i64, whence: c_int) c_int;

fn seek(f: *anyopaque, off: u64) void {
    _ = if (comptime @import("builtin").os.tag == .windows) _fseeki64(f, @intCast(off), 0) else fseeko(f, @intCast(off), 0);
}

fn write(f: *anyopaque, bytes: []const u8) void {
    if (bytes.len != 0) _ = fwrite(bytes.ptr, 1, bytes.len, f);
}

/// Ports `zeros` (export-lora.cpp:32 @c1d0e7a00).
fn zeros(f: *anyopaque, n: usize) void {
    const zero = [1]u8{0};
    for (0..n) |_| write(f, &zero);
}

/// Ports `ggml_ne_string` (export-lora.cpp:39 @c1d0e7a00).
fn neString(a: std.mem.Allocator, t: *const c.ggml_tensor) ![:0]u8 {
    var s: std.ArrayList(u8) = .empty;
    for (0..c.GGML_MAX_DIMS) |i| {
        try s.print(a, "{d}", .{t.ne[i]});
        if (i + 1 < c.GGML_MAX_DIMS) try s.appendSlice(a, ", ");
    }
    return a.dupeSentinel(u8, s.items, 0);
}

/// `GGML_PAD(x, GGUF_DEFAULT_ALIGNMENT)`.
fn pad(x: usize) usize {
    const al: usize = c.GGUF_DEFAULT_ALIGNMENT;
    return (x + al - 1) & ~(al - 1);
}

/// Ports `load_gguf` (export-lora.cpp:50 @c1d0e7a00).
fn loadGguf(fname: [:0]const u8, ctx_ggml: *?*c.ggml_context) !*c.gguf_context {
    const params: c.gguf_init_params = .{ .no_alloc = true, .ctx = ctx_ggml };
    return c.gguf_init_from_file(fname.ptr, params) orelse throw("failed to load input GGUF from {s}", .{fname});
}

/// Ports `file_input` (export-lora.cpp:62 @c1d0e7a00).
const Input = struct {
    ctx_meta: ?*c.ggml_context = null,
    ctx_gguf: ?*c.gguf_context = null,
    f_in: *anyopaque,
    /// `tensors`, by name; `names` walks it in `std::map` order.
    tensors: std.StringHashMapUnmanaged(*c.ggml_tensor) = .empty,
    names: std.ArrayList([]const u8) = .empty,
    alpha: f32 = 0,
    scale: f32,

    fn init(a: std.mem.Allocator, fname: [:0]const u8, scale: f32) !Input {
        const f_in = fopen(fname.ptr, "rb") orelse return throw("failed to open input gguf from {s}", .{fname});
        var self: Input = .{ .f_in = f_in, .scale = scale };

        self.ctx_gguf = try loadGguf(fname, &self.ctx_meta);
        self.alpha = getKvF32(self.ctx_gguf, "adapter.lora.alpha");
        io.printf("%s: loaded gguf from %s\n", .{ "file_input", fname.ptr });

        var cur = c.ggml_get_first_tensor(self.ctx_meta);
        while (cur) |t| : (cur = c.ggml_get_next_tensor(self.ctx_meta, t)) {
            const name = try a.dupe(u8, std.mem.sliceTo(&t.*.name, 0));
            const gop = try self.tensors.getOrPut(a, name);
            if (!gop.found_existing) try self.names.append(a, name);
            gop.value_ptr.* = t;
            if (g_verbose) io.printf("%s: %s\n", .{ "file_input", @as([*:0]const u8, @ptrCast(&t.*.name)) });
        }
        std.mem.sort([]const u8, self.names.items, {}, struct {
            fn lt(_: void, x: []const u8, y: []const u8) bool {
                return std.mem.order(u8, x, y) == .lt;
            }
        }.lt);
        return self;
    }

    /// Ports `get_tensor` (export-lora.cpp:88 @c1d0e7a00), a member of `file_input`.
    fn getTensor(self: *const Input, name: []const u8) ?*c.ggml_tensor {
        return self.tensors.get(name);
    }

    /// Ports `read_tensor_data` (export-lora.cpp:95 @c1d0e7a00), a member of `file_input`.
    fn readTensorData(self: *Input, a: std.mem.Allocator, name: []const u8, buf: *std.ArrayList(u8)) !void {
        const t = self.tensors.get(name) orelse return throw("cannot find tensor with name: {s}", .{name});
        const len = c.ggml_nbytes(t);
        if (buf.items.len < len) try buf.resize(a, len);
        const zname = try a.dupeSentinel(u8, name, 0);
        const i_tensor_in = c.gguf_find_tensor(self.ctx_gguf, zname.ptr); // idx of tensor in the input file
        const offset = c.gguf_get_data_offset(self.ctx_gguf) + c.gguf_get_tensor_offset(self.ctx_gguf, i_tensor_in);
        seek(self.f_in, offset);
        _ = fread(buf.items.ptr, 1, len, self.f_in);
    }

    fn deinit(self: *Input) void {
        c.gguf_free(self.ctx_gguf);
        c.ggml_free(self.ctx_meta);
    }
};

/// Ports `lora_merge_ctx` (export-lora.cpp:115 @c1d0e7a00).
const MergeCtx = struct {
    a: std.mem.Allocator,
    // input base model + adapters
    base_model: Input,
    adapters: std.ArrayList(*Input) = .empty,

    // for computing merged tensor
    n_threads: c_int,
    backend: c.ggml_backend_t = null,
    allocr: c.ggml_gallocr_t = null,
    read_buf: std.ArrayList(u8) = .empty,

    // output file
    ctx_out: ?*c.gguf_context = null,
    ctx_out_ggml: ?*c.ggml_context = null,
    fout: *anyopaque = undefined,

    fn init(a: std.mem.Allocator, base_fname: [:0]const u8, lora_files: []const tc.params.LoraInfo, outfile: [:0]const u8, n_threads: c_int) !MergeCtx {
        var self: MergeCtx = .{ .a = a, .base_model = try Input.init(a, base_fname, 0), .n_threads = n_threads };
        // `fout(outfile, std::ios::binary)` and `fout.exceptions(failbit)`:
        // a stream that did not open throws at once
        self.fout = fopen(outfile.ptr, "wb") orelse return throw("ios_base::clear: unspecified iostream_category error", .{});

        if (c.gguf_find_key(self.base_model.ctx_gguf, tc.kv_split_count) >= 0) {
            return throw("split model is not yet supported", .{});
        }

        for (lora_files) |lora_inp| {
            const adapter = try a.create(Input);
            adapter.* = try Input.init(a, lora_inp.path, lora_inp.scale);
            try self.checkMetadataLora(adapter);
            try self.adapters.append(a, adapter);
        }

        self.ctx_out = c.gguf_init_empty();
        const params: c.ggml_init_params = .{
            .mem_size = @as(usize, @intCast(c.gguf_get_n_tensors(self.base_model.ctx_gguf))) * c.ggml_tensor_overhead(),
            .mem_buffer = null,
            .no_alloc = true,
        };
        self.ctx_out_ggml = c.ggml_init(params);
        self.backend = c.ggml_backend_cpu_init();
        self.allocr = c.ggml_gallocr_new(c.ggml_backend_get_default_buffer_type(self.backend));
        return self;
    }

    /// Ports `check_metadata_lora` (export-lora.cpp:161 @c1d0e7a00), a member of `lora_merge_ctx`.
    fn checkMetadataLora(self: *MergeCtx, adapter: *Input) !void {
        const general_type = getKvStr(adapter.ctx_gguf, "general.type");
        if (!std.mem.eql(u8, general_type, "adapter")) return throw("expect general.type to be 'adapter', but got: {s}", .{general_type});

        const adapter_type = getKvStr(adapter.ctx_gguf, "adapter.type");
        if (!std.mem.eql(u8, adapter_type, "lora")) return throw("expect adapter.type to be 'lora', but got: {s}", .{adapter_type});

        const general_arch_base = getKvStr(self.base_model.ctx_gguf, "general.architecture");
        const general_arch_lora = getKvStr(adapter.ctx_gguf, "general.architecture");
        if (!std.mem.eql(u8, general_arch_base, general_arch_lora)) return throw("model arch and LoRA arch mismatch", .{});
    }

    /// Ports `get_out_tensor_type` (export-lora.cpp:179 @c1d0e7a00), a member of `lora_merge_ctx`.
    fn getOutTensorType(t: *const c.ggml_tensor) c.ggml_type {
        return if (t.type == c.GGML_TYPE_F32) c.GGML_TYPE_F32 else c.GGML_TYPE_F16;
    }

    /// Ports `run_merge` (export-lora.cpp:187 @c1d0e7a00), a member of `lora_merge_ctx`.
    fn runMerge(self: *MergeCtx) !void {
        const a = self.a;
        // prepare metadata
        c.gguf_set_kv(self.ctx_out, self.base_model.ctx_gguf);
        // output is forced to f16 for now
        c.gguf_set_val_u32(self.ctx_out, "general.file_type", c.LLAMA_FTYPE_MOSTLY_F16);

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
                const cpy_tensor = c.ggml_dup_tensor(self.ctx_out_ggml, base_tensor);
                _ = c.ggml_set_name(cpy_tensor, &base_tensor.name);
                try trans.append(a, .{ .in = cpy_tensor, .out = cpy_tensor, .is_copy = true });
                c.gguf_add_tensor(self.ctx_out, cpy_tensor);
            } else if (t_a and t_b) {
                // need merging
                const out_tensor = c.ggml_new_tensor(self.ctx_out_ggml, getOutTensorType(base_tensor), c.GGML_MAX_DIMS, &base_tensor.ne);
                _ = c.ggml_set_name(out_tensor, &base_tensor.name);
                try trans.append(a, .{ .in = base_tensor, .out = out_tensor, .is_copy = false });
                c.gguf_add_tensor(self.ctx_out, out_tensor);
            } else {
                return throw("tensor {s} missing either lora_a or lora_b", .{name});
            }
        }

        // placeholder for the meta data
        zeros(self.fout, c.gguf_get_meta_size(self.ctx_out));

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
            const data = try a.alloc(u8, c.gguf_get_meta_size(self.ctx_out));
            c.gguf_get_meta_data(self.ctx_out, data.ptr);
            seek(self.fout, 0);
            write(self.fout, data);
        }

        io.printf("%s : merged %zu tensors with lora adapters\n", .{ "run_merge", n_merged });
        io.printf("%s : wrote %zu tensors to output file\n", .{ "run_merge", trans.items.len });
    }

    /// Ports `copy_tensor` (export-lora.cpp:274 @c1d0e7a00), a member of `lora_merge_ctx`.
    fn copyTensor(self: *MergeCtx, base: *c.ggml_tensor) !void {
        io.printf("%s :  %s [%s]\n", .{ "copy_tensor", @as([*:0]const u8, @ptrCast(&base.name)), (try neString(self.a, base)).ptr });
        const len = c.ggml_nbytes(base);
        try self.base_model.readTensorData(self.a, std.mem.sliceTo(&base.name, 0), &self.read_buf);
        write(self.fout, self.read_buf.items[0..len]);
        zeros(self.fout, pad(len) - len);
    }

    /// Ports `merge_tensor` (export-lora.cpp:282 @c1d0e7a00), a member of `lora_merge_ctx`.
    fn mergeTensor(self: *MergeCtx, base: *c.ggml_tensor, out: *c.ggml_tensor) !void {
        const a = self.a;
        const func = "merge_tensor";
        const name_base = std.mem.sliceTo(&base.name, 0);
        const name_lora_a = try std.mem.concat(a, u8, &.{ name_base, ".lora_a" });
        const name_lora_b = try std.mem.concat(a, u8, &.{ name_base, ".lora_b" });

        io.printf("%s : %s [%s]\n", .{ func, @as([*:0]const u8, @ptrCast(&base.name)), (try neString(a, base)).ptr });

        // context for input tensor
        const n_ad = self.adapters.items.len;
        const inp_a = try a.alloc(*c.ggml_tensor, n_ad);
        const inp_b = try a.alloc(*c.ggml_tensor, n_ad);
        const params: c.ggml_init_params = .{ .mem_size = c.ggml_tensor_overhead() * (2 + n_ad * 2), .mem_buffer = null, .no_alloc = true };
        const ctx = c.ggml_init(params);

        // alloc tensors
        const inp_base = c.ggml_new_tensor(ctx, c.GGML_TYPE_F32, c.GGML_MAX_DIMS, &base.ne);
        for (self.adapters.items, 0..) |ad, i| {
            const t_a = ad.getTensor(name_lora_a).?;
            const t_b = ad.getTensor(name_lora_b).?;
            // TODO: add support for quantized lora
            if (c.ggml_is_quantized(t_a.type) or c.ggml_is_quantized(t_b.type)) {
                return throw("quantized LoRA adapters is not supported, please retry with f16 or f32", .{});
            }
            inp_a[i] = c.ggml_dup_tensor(ctx, t_a);
            inp_b[i] = c.ggml_dup_tensor(ctx, t_b);
        }
        const buffer = c.ggml_backend_alloc_ctx_tensors(ctx, self.backend);

        // load base tensor to backend buffer
        try self.base_model.readTensorData(a, name_base, &self.read_buf);
        if (base.type != c.GGML_TYPE_F32) {
            // optionally dequantize it
            io.printf("%s :   + dequantize base tensor from %s to F32\n", .{ func, c.ggml_type_name(base.type) });
            const nels: usize = @intCast(c.ggml_nelements(inp_base));
            const qtype = c.ggml_get_type_traits(base.type);
            const dequant_buf = try a.alloc(f32, nels);
            qtype.*.to_float.?(self.read_buf.items.ptr, dequant_buf.ptr, @intCast(nels));
            c.ggml_backend_tensor_set(inp_base, dequant_buf.ptr, 0, nels * @sizeOf(f32));
        } else {
            c.ggml_backend_tensor_set(inp_base, self.read_buf.items.ptr, 0, c.ggml_nbytes(inp_base));
        }

        // load lora tensors to backend buffer
        for (self.adapters.items, 0..) |ad, i| {
            try ad.readTensorData(a, name_lora_a, &self.read_buf);
            c.ggml_backend_tensor_set(inp_a[i], self.read_buf.items.ptr, 0, c.ggml_nbytes(inp_a[i]));
            try ad.readTensorData(a, name_lora_b, &self.read_buf);
            c.ggml_backend_tensor_set(inp_b[i], self.read_buf.items.ptr, 0, c.ggml_nbytes(inp_b[i]));
        }

        // build graph
        var gf: ?*c.ggml_cgraph = null;
        {
            const buf_size = c.ggml_tensor_overhead() * c.GGML_DEFAULT_GRAPH_SIZE + c.ggml_graph_overhead();
            if (graph_buf == null) graph_buf = try std.heap.c_allocator.alloc(u8, buf_size);
            const params0: c.ggml_init_params = .{ .mem_size = buf_size, .mem_buffer = graph_buf.?.ptr, .no_alloc = true };
            const ctx0 = c.ggml_init(params0);
            gf = c.ggml_new_graph(ctx0);
            var cur = inp_base;
            for (self.adapters.items, 0..) |ad, i| {
                var delta: ?*c.ggml_tensor = null;
                const is_tok_embd = std.mem.startsWith(u8, name_base, "token_embd");
                if (is_tok_embd) {
                    io.printf("%s :     detected token embeddings tensor\n", .{func});
                    delta = c.ggml_mul_mat(ctx0, c.ggml_cast(ctx0, inp_b[i], c.GGML_TYPE_F32), c.ggml_cast(ctx0, inp_a[i], c.GGML_TYPE_F32));
                } else {
                    delta = c.ggml_mul_mat(ctx0, c.ggml_cont(ctx0, c.ggml_transpose(ctx0, c.ggml_cast(ctx0, inp_a[i], c.GGML_TYPE_F32))), c.ggml_cast(ctx0, inp_b[i], c.GGML_TYPE_F32));
                }
                // scale
                const alpha = ad.alpha;
                const rank: f32 = @floatFromInt(inp_b[i].ne[0]);
                const scale = if (alpha != 0) ad.scale * alpha / rank else ad.scale;
                delta = c.ggml_scale(ctx0, delta, scale);
                cur = c.ggml_add(ctx0, delta, cur);
                io.printf("%s :   + merging from adapter[%zu] type=%s\n", .{ func, i, c.ggml_type_name(inp_a[i].type) });
                io.printf("%s :     input_scale=%f calculated_scale=%f rank=%d\n", .{ func, @as(f64, ad.scale), @as(f64, scale), @as(c_int, @intCast(inp_b[i].ne[0])) });
            }
            cur = c.ggml_cast(ctx0, cur, out.type);
            io.printf("%s :   + output type is %s\n", .{ func, c.ggml_type_name(out.type) });
            c.ggml_build_forward_expand(gf, cur);
            c.ggml_free(ctx0);
        }

        // compute
        _ = c.ggml_gallocr_alloc_graph(self.allocr, gf);
        c.ggml_backend_cpu_set_n_threads(self.backend, self.n_threads);
        _ = c.ggml_backend_graph_compute(self.backend, gf);

        // write data to output file
        {
            const result = c.ggml_graph_node(gf, -1);
            const len = c.ggml_nbytes(result);
            if (self.read_buf.items.len < len) try self.read_buf.resize(a, len);
            c.ggml_backend_tensor_get(result, self.read_buf.items.ptr, 0, len);
            write(self.fout, self.read_buf.items[0..len]);
            zeros(self.fout, pad(len) - len);
        }

        c.ggml_free(ctx);
        c.ggml_backend_buffer_free(buffer);
    }

    fn deinit(self: *MergeCtx) void {
        c.ggml_gallocr_free(self.allocr);
        c.ggml_backend_free(self.backend);
        c.gguf_free(self.ctx_out);
        c.ggml_free(self.ctx_out_ggml);
        _ = fclose(self.fout);
    }
};

/// The `static std::vector<uint8_t> buf` the graph is built in.
var graph_buf: ?[]u8 = null;

/// Ports `print_usage` (export-lora.cpp:407 @c1d0e7a00).
fn printUsage(argv: []const [:0]const u8) void {
    io.printf("\nexample usage:\n", .{});
    io.printf("\n  %s -m base-model.gguf --lora lora-file.gguf -o merged-model-f16.gguf\n", .{argv[0].ptr});
    io.printf("\nNOTE: output model is F16\n", .{});
    io.printf("\n", .{});
}

/// Ports `main` (export-lora.cpp:414 @c1d0e7a00).
fn run(a: std.mem.Allocator, argv: []const [:0]const u8) !u8 {
    var params = try tc.params.Params.init(a);

    params.out_file = "ggml-lora-merged-f16.gguf";

    tc.common.init();

    if (!try arg.parse(argv, &params, .export_lora, printUsage)) return 1;

    g_verbose = params.verbosity > 1;
    const out_file = try a.dupeSentinel(u8, params.out_file, 0);
    blk: {
        var ctx = MergeCtx.init(a, try a.dupeSentinel(u8, params.model.path, 0), params.lora_adapters.items, out_file, params.cpuparams.n_threads) catch |e| {
            if (e == error.Throw) break :blk;
            return e;
        };
        defer ctx.deinit();
        ctx.runMerge() catch |e| {
            if (e == error.Throw) break :blk;
            return e;
        };
        io.printf("done, output file is %s\n", .{out_file.ptr});
        return 0;
    }
    io.eprintf("%.*s\n", .{ @as(c_int, @intCast(arg.what().len)), arg.what().ptr });
    return 1;
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return run(arena, argv);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
