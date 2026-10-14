//! Compares `src/llama/adapter.zig` and `src/llama/api/adapter.zig`, and
//! the context's adapter setters, with llama.cpp's `llama-adapter.cpp` and
//! `llama-context.cpp`.
//!
//! **Not a port.** The harness behind `scripts/adapter-diff`. It writes a
//! set of LoRA files for the model under test -- good ones over attention,
//! linear-attention, SSM, FFN and embedding weights, F32 and F16, with and
//! without an alpha, an aLoRA with invocation tokens, and one bad file per
//! error `llama_adapter_lora_init_impl` can throw -- then loads the model
//! twice in one process, through the C++ in `libllama.a` and through the
//! port, and runs the same script through `llama.h` and through the port's
//! `api` functions of the same names. Each side writes a transcript, and
//! the two must be identical:
//!
//! - **everything logged** -- the metadata dump, the buffer types chosen,
//!   the buffer sizes, every error -- and every adapter API result: the
//!   metadata by key and by index, truncated and not, the invocation
//!   tokens;
//! - **every graph node computed**, through `cb_eval`, with a hash of its
//!   data, over decodes with no adapter, one, several at different scales,
//!   zero scales, the same set re-applied (which must not reserve again), a
//!   control vector over all layers, over a range, partially overwritten,
//!   rejected and cleared;
//! - every logit row's bits.
//!
//! **The active adapters' order is the pointers' libc++ hash order**, so
//! the C++'s and the port's -- different pointers -- would agree only by
//! chance. The harness loads each of the port's adapters several times and
//! keeps the copies whose order *under the real `std::unordered_map`*
//! (`harness/adapter_ref.cpp`) matches the C++'s at every step; the port's
//! own emulation of that order then decides its graph, so a wrong emulation
//! shows as a node diff. Each step also records the order directly, and a
//! sweep over synthetic pointers compares `adapter.Loras` with the template.
//!
//! Usage: adapter_diff <tmpdir> <model.gguf> [--quick]

const std = @import("std");
const staged = @import("staged");

const c = staged.impl.c;
const M = staged.model;
const L = staged.adapter;
const Pctx = staged.api.context;
const Pad = staged.api.adapter;

const gpa = std.heap.c_allocator;

extern fn adapter_ref_loras_order(keys: [*]const ?*anyopaque, scales: [*]const f32, n: usize, out_keys: [*]?*anyopaque, out_scales: [*]f32, out_bc: *usize) usize;
extern fn adapter_ref_loras_find(keys: [*]const ?*anyopaque, scales: [*]const f32, n: usize, key: ?*anyopaque, out_scale: *f32) c_int;
extern fn truncate(path: [*:0]const u8, len: i64) c_int;

var checks: usize = 0;
var failures: usize = 0;

fn fail(comptime fmt: []const u8, args: anytype) void {
    failures += 1;
    std.debug.print("FAIL " ++ fmt ++ "\n", args);
}

// -----------------------------------------------------------------------------
// The transcript

var rec: ?*std.ArrayList(u8) = null;
var n_nodes_seen: usize = 0;

fn note(comptime fmt: []const u8, args: anytype) void {
    const r = rec orelse return;
    r.print(gpa, fmt ++ "\n", args) catch unreachable;
}

fn logCb(level: c.enum_ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    const r = rec orelse return;
    _ = level;
    r.appendSlice(gpa, std.mem.span(@as([*:0]const u8, @ptrCast(text)))) catch unreachable;
}

fn hashBytes(p: [*]const u8, n: usize) u64 {
    return std.hash.Wyhash.hash(0, p[0..n]);
}

var scratch: std.ArrayList(u8) = .empty;

fn tensorHash(t: *const c.ggml_tensor) u64 {
    const n = c.ggml_nbytes(t);
    scratch.resize(gpa, n) catch unreachable;
    c.ggml_backend_tensor_get(t, scratch.items.ptr, 0, n);
    return hashBytes(scratch.items.ptr, n);
}

fn nodeCb(t: [*c]c.ggml_tensor, ask: bool, _: ?*anyopaque) callconv(.c) bool {
    if (ask) return true;
    const node: *c.ggml_tensor = t;
    n_nodes_seen += 1;
    const r = rec orelse return true;
    r.print(gpa, "node {s} {s} {s} [{d},{d},{d},{d}] [{d},{d},{d},{d}] p", .{
        std.mem.sliceTo(&node.name, 0), c.ggml_op_desc(node), c.ggml_type_name(node.type),
        node.ne[0],                     node.ne[1],           node.ne[2],
        node.ne[3],                     node.nb[0],           node.nb[1],
        node.nb[2],                     node.nb[3],
    }) catch unreachable;
    for (node.op_params) |w| r.print(gpa, " {x}", .{@as(u32, @bitCast(w))}) catch unreachable;
    r.appendSlice(gpa, " src") catch unreachable;
    for (node.src) |s| {
        if (s == null) continue;
        const src: *c.ggml_tensor = s;
        r.print(gpa, " {s}:{s}", .{ std.mem.sliceTo(&src.name, 0), c.ggml_op_desc(src) }) catch unreachable;
        // A LoRA weight is a leaf the callback never sees: hash it here.
        if (src.op == c.GGML_OP_NONE and src.buffer != null and std.mem.find(u8, std.mem.sliceTo(&src.name, 0), ".lora_") != null) {
            r.print(gpa, "={x}", .{tensorHash(src)}) catch unreachable;
        }
    }
    if (node.view_src != null) {
        const v: *c.ggml_tensor = node.view_src;
        r.print(gpa, " view {s}+{d}", .{ std.mem.sliceTo(&v.name, 0), node.view_offs }) catch unreachable;
    }
    r.print(gpa, " flags {x} h {x}\n", .{ node.flags, tensorHash(node) }) catch unreachable;
    return true;
}

/// Lines whose figures move with time, with what the process has
/// allocated, or with an address, and not with what either side computed.
fn normalize(log: []const u8) std.ArrayList(u8) {
    var out: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, log, '\n');
    var first = true;
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "ggml_metal_library_compile_pipeline:")) continue;
        if (!first) out.append(gpa, '\n') catch unreachable;
        first = false;
        if (std.mem.find(u8, line, "reserve took ")) |at| {
            out.appendSlice(gpa, line[0 .. at + "reserve took ".len]) catch unreachable;
            out.appendSlice(gpa, "N ms") catch unreachable;
            if (std.mem.find(u8, line, ", sched copies")) |rest| out.appendSlice(gpa, line[rest..]) catch unreachable;
            continue;
        }
        if (std.mem.startsWith(u8, line, "ggml_metal_log_allocated_size:")) {
            if (std.mem.findScalar(u8, line, '(')) |open| {
                out.appendSlice(gpa, line[0..open]) catch unreachable;
                out.appendSlice(gpa, "(N)") catch unreachable;
                continue;
            }
        }
        // The adapters array is the harness's, a different one per side.
        if (std.mem.find(u8, line, ": adapters = ")) |at| {
            out.appendSlice(gpa, line[0 .. at + ": adapters = ".len]) catch unreachable;
            out.appendSlice(gpa, "P") catch unreachable;
            continue;
        }
        out.appendSlice(gpa, line) catch unreachable;
    }
    return out;
}

fn firstDiff(name: []const u8, a: []const u8, b: []const u8) void {
    var ia = std.mem.splitScalar(u8, a, '\n');
    var ib = std.mem.splitScalar(u8, b, '\n');
    var line: usize = 1;
    while (true) : (line += 1) {
        const la = ia.next();
        const lb = ib.next();
        if (la == null and lb == null) return;
        checks += 1;
        if (la == null or lb == null or !std.mem.eql(u8, la.?, lb.?)) {
            fail("{s}: transcript line {d}:\n    ref  '{s}'\n    port '{s}'", .{ name, line, la orelse "<end>", lb orelse "<end>" });
            return;
        }
    }
}

// -----------------------------------------------------------------------------
// The model's shapes

const Info = struct {
    arch: []u8,
    n_embd: i64,
    n_layer: i64,
    names: std.ArrayList([]u8) = .empty,
    shapes: std.ArrayList([2]i64) = .empty,
    dims: std.ArrayList(u32) = .empty,

    fn shape(self: *const Info, name: []const u8) ?[2]i64 {
        for (self.names.items, 0..) |n, i| if (std.mem.eql(u8, n, name)) return self.shapes.items[i];
        return null;
    }
};

fn readInfo(path: [:0]const u8) Info {
    var mctx: ?*c.ggml_context = null;
    const g = c.gguf_init_from_file(path.ptr, .{ .no_alloc = true, .ctx = @ptrCast(&mctx) }) orelse {
        std.debug.print("FAIL: cannot read {s}\n", .{path});
        std.process.exit(1);
    };
    defer c.gguf_free(g);
    defer c.ggml_free(mctx);
    const arch = std.mem.span(c.gguf_get_val_str(g, c.gguf_find_key(g, "general.architecture")));
    var key: [128]u8 = undefined;
    const ke = std.fmt.bufPrintSentinel(&key, "{s}.embedding_length", .{arch}, 0) catch unreachable;
    const kl = std.fmt.bufPrintSentinel(&key, "{s}.block_count", .{arch}, 0) catch unreachable;
    _ = ke;
    var info: Info = .{ .arch = gpa.dupe(u8, arch) catch unreachable, .n_embd = 0, .n_layer = 0 };
    info.n_layer = c.gguf_get_val_u32(g, c.gguf_find_key(g, kl.ptr));
    const ke2 = std.fmt.bufPrintSentinel(&key, "{s}.embedding_length", .{arch}, 0) catch unreachable;
    info.n_embd = c.gguf_get_val_u32(g, c.gguf_find_key(g, ke2.ptr));
    var cur: ?*c.ggml_tensor = c.ggml_get_first_tensor(mctx);
    while (cur) |t| : (cur = c.ggml_get_next_tensor(mctx, t)) {
        info.names.append(gpa, gpa.dupe(u8, std.mem.sliceTo(&t.name, 0)) catch unreachable) catch unreachable;
        info.shapes.append(gpa, .{ t.ne[0], t.ne[1] }) catch unreachable;
        info.dims.append(gpa, @intCast(c.ggml_n_dims(t))) catch unreachable;
    }
    return info;
}

// -----------------------------------------------------------------------------
// Writing LoRA files

var rng_state: u64 = 0x9e3779b97f4a7c15;

fn rnd() f32 {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    const u: u32 = @truncate(rng_state >> 40);
    return @as(f32, @floatFromInt(u)) / @as(f32, 1 << 24) * 2.0 - 1.0;
}

const File = struct {
    g: *c.gguf_context,
    ctx: *c.ggml_context,

    fn init(general_type: ?[:0]const u8, arch: ?[]const u8, adapter_type: ?[:0]const u8, alpha: ?f32) File {
        const ctx = c.ggml_init(.{ .mem_size = 96 << 20, .mem_buffer = null, .no_alloc = false }).?;
        const g = c.gguf_init_empty().?;
        if (general_type) |t| c.gguf_set_val_str(g, "general.type", t.ptr);
        if (arch) |a| {
            var buf: [64]u8 = undefined;
            const z = std.fmt.bufPrintSentinel(&buf, "{s}", .{a}, 0) catch unreachable;
            c.gguf_set_val_str(g, "general.architecture", z.ptr);
        }
        if (adapter_type) |t| c.gguf_set_val_str(g, "adapter.type", t.ptr);
        if (alpha) |a| c.gguf_set_val_f32(g, "adapter.lora.alpha", a);
        return .{ .g = g, .ctx = ctx };
    }

    fn tensor(self: *File, name: []const u8, ty: c.enum_ggml_type, ne0: i64, ne1: i64, amp: f32) void {
        const t = c.ggml_new_tensor_2d(self.ctx, ty, ne0, ne1);
        var buf: [128]u8 = undefined;
        const z = std.fmt.bufPrintSentinel(&buf, "{s}", .{name}, 0) catch unreachable;
        _ = c.ggml_set_name(t, z.ptr);
        const n: usize = @intCast(ne0 * ne1);
        if (ty == c.GGML_TYPE_F32) {
            const d: [*]f32 = @ptrCast(@alignCast(t.*.data));
            for (0..n) |i| d[i] = rnd() * amp;
        } else {
            const d: [*]f16 = @ptrCast(@alignCast(t.*.data));
            for (0..n) |i| d[i] = @floatCast(rnd() * amp);
        }
        c.gguf_add_tensor(self.g, t);
    }

    /// A pair adapting `target`, whose shape is `ne`.
    fn pair(self: *File, target: []const u8, ne: [2]i64, rank: i64, ta: c.enum_ggml_type, tb: c.enum_ggml_type) void {
        var na: [128]u8 = undefined;
        var nb: [128]u8 = undefined;
        if (std.mem.endsWith(u8, target, "token_embd.weight")) {
            self.tensor(std.fmt.bufPrint(&na, "{s}.lora_a", .{target}) catch unreachable, ta, rank, ne[1], 0.5);
            self.tensor(std.fmt.bufPrint(&nb, "{s}.lora_b", .{target}) catch unreachable, tb, rank, ne[0], 0.05);
        } else {
            self.tensor(std.fmt.bufPrint(&na, "{s}.lora_a", .{target}) catch unreachable, ta, ne[0], rank, 0.05);
            self.tensor(std.fmt.bufPrint(&nb, "{s}.lora_b", .{target}) catch unreachable, tb, rank, ne[1], 0.05);
        }
    }

    fn write(self: *File, path: [:0]const u8) void {
        if (!c.gguf_write_to_file(self.g, path.ptr, false)) {
            std.debug.print("FAIL: cannot write {s}\n", .{path});
            std.process.exit(1);
        }
        c.gguf_free(self.g);
        c.ggml_free(self.ctx);
    }
};

/// `good`: one of the four the script sets. `keep`: loaded on both sides,
/// queried, and left for the model to free.
const Lf = struct { name: []const u8, path: [:0]u8, good: bool, keep: bool = false };

fn pathFor(dir: []const u8, name: []const u8) [:0]u8 {
    return std.fmt.allocPrintSentinel(gpa, "{s}/{s}.gguf", .{ dir, name }, 0) catch unreachable;
}

/// Every 2-D weight of block `bid` a LoRA can adapt.
fn blockTargets(info: *const Info, bid: u32, out: *std.ArrayList([]const u8)) void {
    var pre: [32]u8 = undefined;
    const p = std.fmt.bufPrint(&pre, "blk.{d}.", .{bid}) catch unreachable;
    for (info.names.items, 0..) |n, i| {
        if (!std.mem.startsWith(u8, n, p) or info.dims.items[i] != 2) continue;
        if (!std.mem.endsWith(u8, n, ".weight") or std.mem.find(u8, n, "norm") != null) continue;
        if (std.mem.find(u8, n, "conv1d") != null) continue;
        out.append(gpa, n) catch unreachable;
    }
}

fn writeFiles(dir: []const u8, info: *const Info, files: *std.ArrayList(Lf)) void {
    const arch = info.arch;
    const F32 = c.GGML_TYPE_F32;
    const F16 = c.GGML_TYPE_F16;

    // A: block 0 (linear attention, SSM, FFN), F32, rank 4, alpha 8. Its
    // ssm_out is named with `.lora_a` twice, which `replace_all` strips.
    {
        var f = File.init("adapter", arch, "lora", 8.0);
        var ts: std.ArrayList([]const u8) = .empty;
        defer ts.deinit(gpa);
        blockTargets(info, 0, &ts);
        for (ts.items) |t| {
            const ne = info.shape(t).?;
            if (std.mem.endsWith(u8, t, "ssm_out.weight")) {
                var na: [128]u8 = undefined;
                var nb: [128]u8 = undefined;
                const stem = t[0 .. t.len - ".weight".len];
                f.tensor(std.fmt.bufPrint(&na, "{s}.lora_a.weight.lora_a", .{stem}) catch unreachable, F32, ne[0], 4, 0.05);
                f.tensor(std.fmt.bufPrint(&nb, "{s}.lora_b", .{t}) catch unreachable, F32, 4, ne[1], 0.05);
            } else {
                f.pair(t, ne, 4, F32, F32);
            }
        }
        const p = pathFor(dir, "a");
        f.write(p);
        files.append(gpa, .{ .name = "a", .path = p, .good = true }) catch unreachable;
    }

    // B: block 3 (full attention, FFN) plus block 0's ffn_up, which A also
    // adapts; F16 and mixed; no alpha.
    {
        var f = File.init("adapter", arch, "lora", null);
        var ts: std.ArrayList([]const u8) = .empty;
        defer ts.deinit(gpa);
        blockTargets(info, 3, &ts);
        for (ts.items, 0..) |t, i| f.pair(t, info.shape(t).?, 2, F16, if (i % 2 == 0) F16 else F32);
        if (info.shape("blk.0.ffn_up.weight")) |ne| f.pair("blk.0.ffn_up.weight", ne, 2, F16, F16);
        const p = pathFor(dir, "b");
        f.write(p);
        files.append(gpa, .{ .name = "b", .path = p, .good = true }) catch unreachable;
    }

    // C: an aLoRA over block 1, rank 8, alpha 16, with metadata of every
    // kind and a norm tensor, which is skipped.
    {
        var f = File.init("adapter", arch, "lora", 16.0);
        const toks = [_]u32{ 151644, 77091, 198, 7 };
        c.gguf_set_arr_data(f.g, "adapter.alora.invocation_tokens", c.GGUF_TYPE_UINT32, &toks, toks.len);
        c.gguf_set_val_str(f.g, "general.name", "synthetic aLoRA");
        c.gguf_set_val_str(f.g, "adapter.lora.task_name", "a task name that is quite a bit longer than forty bytes\nand has a newline");
        c.gguf_set_val_u32(f.g, "training.epochs", 3);
        c.gguf_set_val_f32(f.g, "training.lr", 0.000125);
        c.gguf_set_val_bool(f.g, "training.flag", true);
        var strs = [_][*c]const u8{ "x", "y\"z" };
        c.gguf_set_arr_str(f.g, "training.list", &strs, strs.len);
        const ints = [_]i32{ -1, 2, 3 };
        c.gguf_set_arr_data(f.g, "training.ints", c.GGUF_TYPE_INT32, &ints, ints.len);
        c.gguf_set_val_str(f.g, "z.last", "");
        var ts: std.ArrayList([]const u8) = .empty;
        defer ts.deinit(gpa);
        blockTargets(info, 1, &ts);
        for (ts.items) |t| f.pair(t, info.shape(t).?, 8, F32, F16);
        f.tensor("blk.1.attn_norm.weight", F32, info.n_embd, 1, 1.0);
        const p = pathFor(dir, "c");
        f.write(p);
        files.append(gpa, .{ .name = "c", .path = p, .good = true }) catch unreachable;
    }

    // E: block 2, F32, rank 3, and the output head where the model has
    // one.
    {
        var f = File.init("adapter", arch, "lora", 4.0);
        if (info.shape("output.weight")) |ne| f.pair("output.weight", ne, 2, F32, F32);
        var ts: std.ArrayList([]const u8) = .empty;
        defer ts.deinit(gpa);
        blockTargets(info, 2, &ts);
        for (ts.items) |t| f.pair(t, info.shape(t).?, 3, F32, F32);
        const p = pathFor(dir, "e");
        f.write(p);
        files.append(gpa, .{ .name = "e", .path = p, .good = true }) catch unreachable;
    }

    // The embeddings, loaded but never set: on a model whose output head is
    // the embedding matrix (Qwen3.5-2B ties them), the head's
    // `build_lora_mm` finds the flipped embedding pair by name and the C++
    // aborts in `ggml_mul_mat`. It is left for the model to free.
    {
        var f = File.init("adapter", arch, "lora", 4.0);
        f.pair("token_embd.weight", info.shape("token_embd.weight").?, 2, F16, F32);
        const p = pathFor(dir, "embd");
        f.write(p);
        files.append(gpa, .{ .name = "embd", .path = p, .good = false, .keep = true }) catch unreachable;
    }

    const q = info.shape("blk.3.attn_q.weight") orelse info.shape("blk.0.ffn_up.weight").?;
    const qname: []const u8 = if (info.shape("blk.3.attn_q.weight") != null) "blk.3.attn_q.weight" else "blk.0.ffn_up.weight";

    const Bad = struct {
        fn one(d: []const u8, fs: *std.ArrayList(Lf), name: []const u8, f: *File) void {
            const p = pathFor(d, name);
            f.write(p);
            fs.append(gpa, .{ .name = name, .path = p, .good = false }) catch unreachable;
        }
    };

    {
        var f = File.init("model", arch, "lora", 8.0);
        f.pair(qname, q, 2, F32, F32);
        Bad.one(dir, files, "type-model", &f);
    }
    {
        var f = File.init(null, arch, "lora", 8.0);
        f.pair(qname, q, 2, F32, F32);
        Bad.one(dir, files, "type-missing", &f);
    }
    {
        var f = File.init("adapter", "llama", "lora", 8.0);
        f.pair(qname, q, 2, F32, F32);
        Bad.one(dir, files, "arch", &f);
    }
    {
        var f = File.init("adapter", null, "lora", 8.0);
        f.pair(qname, q, 2, F32, F32);
        Bad.one(dir, files, "arch-missing", &f);
    }
    {
        var f = File.init("adapter", arch, "control_vector", 8.0);
        f.pair(qname, q, 2, F32, F32);
        Bad.one(dir, files, "adapter-type", &f);
    }
    {
        var f = File.init("adapter", arch, "lora", 8.0);
        c.gguf_set_val_str(f.g, "adapter.alora.invocation_tokens", "1 2 3");
        f.pair(qname, q, 2, F32, F32);
        Bad.one(dir, files, "alora-str", &f);
    }
    {
        var f = File.init("adapter", arch, "lora", 8.0);
        const ints = [_]i32{ 1, 2 };
        c.gguf_set_arr_data(f.g, "adapter.alora.invocation_tokens", c.GGUF_TYPE_INT32, &ints, ints.len);
        f.pair(qname, q, 2, F32, F32);
        Bad.one(dir, files, "alora-i32", &f);
    }
    {
        var f = File.init("adapter", arch, "lora", 8.0);
        f.pair(qname, q, 2, F32, F32);
        f.tensor("blk.0.attn_qkv.weight.lora_c", F32, 4, 4, 0.1);
        Bad.one(dir, files, "suffix", &f);
    }
    {
        var f = File.init("adapter", arch, "lora", 8.0);
        f.pair(qname, q, 2, F32, F32);
        f.tensor("blk.1.ffn_up.weight.lora_a", F32, info.n_embd, 2, 0.1);
        Bad.one(dir, files, "missing-b", &f);
    }
    {
        var f = File.init("adapter", arch, "lora", 8.0);
        f.tensor("blk.1.ffn_up.weight.lora_b", F32, 2, 16, 0.1);
        f.pair(qname, q, 2, F32, F32);
        Bad.one(dir, files, "missing-a", &f);
    }
    {
        var f = File.init("adapter", arch, "lora", 8.0);
        f.pair("blk.999.attn_q.weight", q, 2, F32, F32);
        Bad.one(dir, files, "not-in-model", &f);
    }
    {
        // Two errors: the file lists blk.1's first, std::map visits blk.0's
        // first, and that is the one reported.
        var f = File.init("adapter", arch, "lora", 8.0);
        f.pair("blk.1.no_such.weight", q, 2, F32, F32);
        f.tensor("blk.0.ffn_up.weight.lora_a", F32, info.n_embd, 2, 0.1);
        Bad.one(dir, files, "two-errors", &f);
    }
    {
        var f = File.init("adapter", arch, "lora", 8.0);
        f.pair(qname, .{ q[0] + 1, q[1] }, 2, F32, F32);
        Bad.one(dir, files, "shape-a", &f);
    }
    {
        var f = File.init("adapter", arch, "lora", 8.0);
        f.pair(qname, .{ q[0], q[1] - 1 }, 2, F32, F32);
        Bad.one(dir, files, "shape-b", &f);
    }
    {
        var f = File.init("adapter", arch, "lora", 8.0);
        f.tensor("blk.3.attn_q.weight.lora_a", F32, q[0], 2, 0.1);
        f.tensor("blk.3.attn_q.weight.lora_b", F32, 3, q[1], 0.1);
        Bad.one(dir, files, "not-transposed", &f);
    }
    {
        var f = File.init("adapter", arch, "lora", 8.0);
        const e = info.shape("token_embd.weight").?;
        f.pair("token_embd.weight", .{ e[0] - 1, e[1] }, 2, F32, F32);
        Bad.one(dir, files, "embd-shape", &f);
    }
    {
        var f = File.init("adapter", arch, "lora", 8.0);
        f.pair(qname, q, 2, F32, F32);
        const p = pathFor(dir, "truncated");
        f.write(p);
        const fp = c.fopen(p.ptr, "rb");
        _ = c.fseek(fp, 0, c.SEEK_END);
        const size = c.ftell(fp);
        _ = c.fclose(fp);
        _ = truncate(p.ptr, size - 64);
        files.append(gpa, .{ .name = "truncated", .path = p, .good = false }) catch unreachable;
    }
    {
        const p = pathFor(dir, "not-gguf");
        const fp = c.fopen(p.ptr, "wb");
        _ = c.fwrite("this is not a gguf file\n", 1, 24, fp);
        _ = c.fclose(fp);
        files.append(gpa, .{ .name = "not-gguf", .path = p, .good = false }) catch unreachable;
    }
    files.append(gpa, .{ .name = "missing-file", .path = pathFor(dir, "does-not-exist"), .good = false }) catch unreachable;
}

// -----------------------------------------------------------------------------
// The two sides

fn Api(comptime port: bool) type {
    return struct {
        const init_from_model = if (port) Pctx.llama_init_from_model else c.llama_init_from_model;
        const free = if (port) Pctx.llama_free else c.llama_free;
        const get_memory = if (port) Pctx.llama_get_memory else c.llama_get_memory;
        const memory_clear = if (port) Pctx.llama_memory_clear else c.llama_memory_clear;
        const decode = if (port) Pctx.llama_decode else c.llama_decode;
        const get_logits_ith = if (port) Pctx.llama_get_logits_ith else c.llama_get_logits_ith;
        const lora_init = if (port) Pad.llama_adapter_lora_init else c.llama_adapter_lora_init;
        const lora_free = if (port) Pad.llama_adapter_lora_free else c.llama_adapter_lora_free;
        const meta_count = if (port) Pad.llama_adapter_meta_count else c.llama_adapter_meta_count;
        const meta_key_by_index = if (port) Pad.llama_adapter_meta_key_by_index else c.llama_adapter_meta_key_by_index;
        const meta_val_str = if (port) Pad.llama_adapter_meta_val_str else c.llama_adapter_meta_val_str;
        const meta_val_str_by_index = if (port) Pad.llama_adapter_meta_val_str_by_index else c.llama_adapter_meta_val_str_by_index;
        const alora_n = if (port) Pad.llama_adapter_get_alora_n_invocation_tokens else c.llama_adapter_get_alora_n_invocation_tokens;
        const alora_tokens = if (port) Pad.llama_adapter_get_alora_invocation_tokens else c.llama_adapter_get_alora_invocation_tokens;
        const set_loras = if (port) Pad.llama_set_adapters_lora else c.llama_set_adapters_lora;
        const set_cvec = if (port) Pad.llama_set_adapter_cvec else c.llama_set_adapter_cvec;
    };
}

const Handle = ?*c.struct_llama_adapter_lora;

fn query(comptime A: type, ad: Handle, name: []const u8) void {
    if (ad == null) {
        note("adapter {s}: not loaded", .{name});
        return;
    }
    var buf: [512]u8 = undefined;
    const n = A.meta_count(ad);
    note("adapter {s}: meta_count {d}", .{ name, n });
    var i: i32 = -1;
    while (i <= n) : (i += 1) {
        @memset(&buf, 0xaa);
        const rk = A.meta_key_by_index(ad, i, &buf, buf.len);
        const key = std.mem.sliceTo(&buf, 0);
        note("  key {d}: {d} '{s}'", .{ i, rk, key });
        @memset(&buf, 0xaa);
        const rv = A.meta_val_str_by_index(ad, i, &buf, buf.len);
        note("  val {d}: {d} '{s}'", .{ i, rv, std.mem.sliceTo(&buf, 0) });
    }
    @memset(&buf, 0xaa);
    note("  key 0 in 6: {d} '{s}'", .{ A.meta_key_by_index(ad, 0, &buf, 6), std.mem.sliceTo(&buf, 0) });
    note("  key 0 in 0: {d}", .{A.meta_key_by_index(ad, 0, null, 0)});
    note("  key -1 in 0: {d}", .{A.meta_key_by_index(ad, -1, null, 0)});
    @memset(&buf, 0xaa);
    note("  val 1 in 3: {d} '{s}'", .{ A.meta_val_str_by_index(ad, 1, &buf, 3), std.mem.sliceTo(&buf, 0) });
    for ([_][*:0]const u8{ "general.type", "adapter.lora.alpha", "adapter.lora.task_name", "training.list", "nope", "" }) |k| {
        @memset(&buf, 0xaa);
        const r = A.meta_val_str(ad, k, &buf, buf.len);
        note("  get '{s}': {d} '{s}'", .{ k, r, std.mem.sliceTo(&buf, 0) });
    }
    @memset(&buf, 0xaa);
    note("  get type in 4: {d} '{s}'", .{ A.meta_val_str(ad, "general.type", &buf, 4), std.mem.sliceTo(&buf, 0) });
    const na = A.alora_n(ad);
    const toks = A.alora_tokens(ad);
    note("  alora {d} null {}", .{ na, toks == null });
    for (0..na) |k| note("    {d}", .{toks[k]});
}

// -----------------------------------------------------------------------------
// The script

const Set = struct { idx: usize, scale: f32 };

const Cv = enum { full, short, bad_embd, clear };

const Step = union(enum) {
    decode,
    lora: []const Set,
    lora_null,
    cvec: struct { mode: Cv, start: i32, end: i32 },
};

/// The adapters the script sets: a, b, c, e.
const n_good = 4;
const last_layer: i32 = -100;

const steps = [_]Step{
    .decode,
    .{ .lora = &.{.{ .idx = 0, .scale = 1.0 }} },
    .decode,
    // The same set again: no new reserve.
    .{ .lora = &.{.{ .idx = 0, .scale = 1.0 }} },
    .decode,
    .{ .lora = &.{ .{ .idx = 0, .scale = 0.5 }, .{ .idx = 1, .scale = 0.0 }, .{ .idx = 2, .scale = 2.0 } } },
    .decode,
    .{ .lora = &.{ .{ .idx = 2, .scale = 2.0 }, .{ .idx = 0, .scale = 0.5 } } },
    .decode,
    .{ .lora = &.{ .{ .idx = 0, .scale = 0.3 }, .{ .idx = 1, .scale = 1.5 }, .{ .idx = 2, .scale = 2.0 }, .{ .idx = 3, .scale = 0.3 } } },
    .decode,
    .{ .lora = &.{.{ .idx = 1, .scale = 0.0 }} },
    .decode,
    .{ .cvec = .{ .mode = .full, .start = 1, .end = last_layer } },
    .decode,
    .{ .cvec = .{ .mode = .full, .start = 5, .end = 10 } },
    .decode,
    .{ .lora = &.{ .{ .idx = 0, .scale = 1.0 }, .{ .idx = 1, .scale = -1.0 } } },
    .decode,
    .{ .cvec = .{ .mode = .bad_embd, .start = 1, .end = 4 } },
    .{ .cvec = .{ .mode = .short, .start = 0, .end = 100 } },
    .decode,
    .{ .cvec = .{ .mode = .clear, .start = 0, .end = 0 } },
    .decode,
    .lora_null,
    .decode,
    .{ .lora = &.{ .{ .idx = 0, .scale = 1.0 }, .{ .idx = 0, .scale = 2.0 } } },
    .decode,
    .{ .lora = &.{.{ .idx = 0, .scale = 1.0 }} },
    .decode,
    .{ .lora = &.{ .{ .idx = 3, .scale = 1.0 }, .{ .idx = 1, .scale = 0.25 }, .{ .idx = 2, .scale = 0.125 } } },
    .decode,
};

const prompt_text = "The quick brown fox jumps over the lazy dog. The capital of France is";

fn argmax(row: [*]const f32, n: usize) i32 {
    var best: usize = 0;
    for (1..n) |i| {
        if (row[i] > row[best]) best = i;
    }
    return @intCast(best);
}

/// The order the real `std::unordered_map` iterates `keys` in, as indices
/// into `handles`.
fn trueOrder(handles: []const Handle, sets: []const Set, out: []usize) usize {
    var keys: [16]?*anyopaque = undefined;
    var scales: [16]f32 = undefined;
    for (sets, 0..) |s, i| {
        keys[i] = handles[s.idx];
        scales[i] = s.scale;
    }
    var ok: [16]?*anyopaque = undefined;
    var os: [16]f32 = undefined;
    var bc: usize = 0;
    const n = adapter_ref_loras_order(&keys, &scales, sets.len, &ok, &os, &bc);
    for (0..n) |k| {
        out[k] = for (handles, 0..) |h, j| {
            if (@as(?*anyopaque, h) == ok[k]) break j;
        } else 99;
    }
    return n;
}

/// For each step, the sets the context's map was last rebuilt from:
/// `set_adapters_lora` keeps the map it has when the new sets are the same
/// adapters at the same scales, so its order is that earlier insertion's.
var effective: [steps.len][]const Set = undefined;

fn computeEffective() void {
    var cur: []const Set = &.{};
    for (steps, 0..) |st, si| {
        switch (st) {
            .lora => |sets| {
                if (!sameAsMap(cur, sets)) cur = sets;
            },
            .lora_null => {
                if (!sameAsMap(cur, &.{})) cur = &.{};
            },
            else => {},
        }
        effective[si] = cur;
    }
}

/// `adapters_lora_are_same` against the map `cur` built.
fn sameAsMap(cur: []const Set, sets: []const Set) bool {
    // The map: the non-zero entries, the first of a repeated adapter kept.
    var map: [16]Set = undefined;
    var n_map: usize = 0;
    for (cur) |s| {
        if (s.scale == 0.0) continue;
        const dup = for (map[0..n_map]) |m| {
            if (m.idx == s.idx) break true;
        } else false;
        if (!dup) {
            map[n_map] = s;
            n_map += 1;
        }
    }
    var n_non_zero: usize = 0;
    for (sets) |s| {
        if (s.scale == 0.0) continue;
        n_non_zero += 1;
        const hit = for (map[0..n_map]) |m| {
            if (m.idx == s.idx) break m;
        } else return false;
        if (hit.scale != s.scale) return false;
    }
    return n_non_zero == n_map;
}

var cvec_full: []f32 = &.{};
var cvec_short: []f32 = &.{};

fn script(comptime port: bool, model: ?*c.struct_llama_model, handles: []const Handle, tokens: []const i32, n_vocab: usize, n_layer: i32) void {
    const A = Api(port);
    var params = c.llama_context_default_params();
    params.cb_eval = nodeCb;
    params.cb_eval_user_data = null;
    params.no_perf = true;
    params.n_ctx = 512;
    const ctx = A.init_from_model(model, params) orelse {
        note("init failed", .{});
        return;
    };
    defer A.free(ctx);
    const mem = A.get_memory(ctx);
    var b = c.llama_batch_init(@intCast(tokens.len + 8), 0, 1);
    defer c.llama_batch_free(b);

    for (steps, 0..) |st, si| {
        note("step {d}", .{si});
        switch (st) {
            .decode => {
                A.memory_clear(mem, true);
                b.n_tokens = 0;
                for (tokens, 0..) |t, i| {
                    b.token[i] = t;
                    b.pos[i] = @intCast(i);
                    b.n_seq_id[i] = 1;
                    b.seq_id[i][0] = 0;
                    b.logits[i] = @intFromBool(i + 1 == tokens.len);
                }
                b.n_tokens = @intCast(tokens.len);
                var ret = A.decode(ctx, b);
                note("decode prompt: {d}", .{ret});
                if (ret != 0) return;
                var pos: i32 = @intCast(tokens.len);
                for (0..2) |_| {
                    const r = A.get_logits_ith(ctx, -1) orelse {
                        note("logits null", .{});
                        return;
                    };
                    note("logits {x}", .{hashBytes(@ptrCast(r), n_vocab * @sizeOf(f32))});
                    const next = argmax(r, n_vocab);
                    note("token {d}", .{next});
                    b.n_tokens = 1;
                    b.token[0] = next;
                    b.pos[0] = pos;
                    b.n_seq_id[0] = 1;
                    b.seq_id[0][0] = 0;
                    b.logits[0] = 1;
                    ret = A.decode(ctx, b);
                    note("decode step: {d}", .{ret});
                    if (ret != 0) return;
                    pos += 1;
                }
                if (A.get_logits_ith(ctx, -1)) |r| note("logits {x}", .{hashBytes(@ptrCast(r), n_vocab * @sizeOf(f32))});
            },
            .lora => |sets| {
                var ads: [16]Handle = undefined;
                var scales: [16]f32 = undefined;
                for (sets, 0..) |s, i| {
                    ads[i] = handles[s.idx];
                    scales[i] = s.scale;
                }
                note("set loras: {d}", .{A.set_loras(ctx, &ads, sets.len, &scales)});
                // The order the context's map iterates, as adapter indices.
                var order: [16]usize = undefined;
                var n: usize = 0;
                if (port) {
                    for (Pctx.ctxOf(ctx).loras.items.items) |e| {
                        order[n] = for (handles, 0..) |h, j| {
                            if (@as(?*anyopaque, h) == @as(?*anyopaque, e.lora)) break j;
                        } else 99;
                        n += 1;
                    }
                } else {
                    n = trueOrder(handles, effective[si], &order);
                }
                note("order {any}", .{order[0..n]});
            },
            .lora_null => note("set loras null: {d}", .{A.set_loras(ctx, null, 0, null)}),
            .cvec => |cv| {
                const end = if (cv.end == last_layer) n_layer - 1 else cv.end;
                const n_embd: i32 = @intCast(cvec_full.len / @as(usize, @intCast(n_layer - 1)));
                const r = switch (cv.mode) {
                    .full => A.set_cvec(ctx, cvec_full.ptr, cvec_full.len, n_embd, cv.start, end),
                    .short => A.set_cvec(ctx, cvec_short.ptr, cvec_short.len, n_embd, cv.start, end),
                    .bad_embd => A.set_cvec(ctx, cvec_full.ptr, cvec_full.len, n_embd - 1, cv.start, end),
                    .clear => A.set_cvec(ctx, null, 0, n_embd, cv.start, end),
                };
                note("set cvec {s}: {d}", .{ @tagName(cv.mode), r });
            },
        }
    }
}

// -----------------------------------------------------------------------------
// The order of the active adapters, against the real template

fn sweepOrder() void {
    var prng = std.Random.DefaultPrng.init(42);
    const R = prng.random();
    var trial: usize = 0;
    while (trial < 3000) : (trial += 1) {
        const n = R.intRangeAtMost(usize, 1, 40);
        var keys: [64]?*anyopaque = undefined;
        var scales: [64]f32 = undefined;
        // Heap-like addresses, sometimes close together, sometimes reused.
        const base: usize = 0x6000_0000_0000 + R.intRangeAtMost(usize, 0, 1 << 32) * 16;
        for (0..n) |i| {
            const reuse = i > 0 and R.uintLessThan(u8, 8) == 0;
            keys[i] = if (reuse) keys[R.uintLessThan(usize, i)] else @ptrFromInt(base + R.intRangeAtMost(usize, 0, if (trial % 2 == 0) 64 else 1 << 20) * 16);
            scales[i] = switch (R.uintLessThan(u8, 6)) {
                0 => 0.0,
                1 => -0.0,
                else => R.float(f32) * 4 - 2,
            };
        }
        var l: L.Loras = .{};
        defer l.deinit();
        for (0..n) |i| if (scales[i] != 0.0) l.insert(@ptrCast(@alignCast(keys[i].?)), scales[i]);
        var ok: [64]?*anyopaque = undefined;
        var os: [64]f32 = undefined;
        var bc: usize = 0;
        const m = adapter_ref_loras_order(&keys, &scales, n, &ok, &os, &bc);
        checks += 1;
        var same = m == l.items.items.len and (l.map == null or l.map.?.bucketCount() == bc) and (l.map != null or bc == 0);
        if (same) {
            for (0..m) |k| {
                if (@as(?*anyopaque, l.items.items[k].lora) != ok[k] or @as(u32, @bitCast(l.items.items[k].scale)) != @as(u32, @bitCast(os[k]))) same = false;
            }
        }
        if (!same) {
            fail("order sweep trial {d}: n {d}, size {d} vs {d}, buckets {d}", .{ trial, n, l.items.items.len, m, bc });
            return;
        }
        // find, present and absent.
        for (0..n + 2) |i| {
            const key: ?*anyopaque = if (i < n) keys[i] else @ptrFromInt(base + (i + 1) * 4096 * 16);
            var s: f32 = 0;
            const found = adapter_ref_loras_find(&keys, &scales, n, key, &s) != 0;
            const mine = l.find(@ptrCast(@alignCast(key)));
            checks += 1;
            if (found != (mine != null) or (found and @as(u32, @bitCast(s)) != @as(u32, @bitCast(mine.?)))) {
                fail("find sweep trial {d}: key {d}", .{ trial, i });
                return;
            }
        }
    }
}

// -----------------------------------------------------------------------------

const ModelCfg = struct { name: []const u8, n_gpu_layers: i32 };

const model_cfgs = [_]ModelCfg{
    .{ .name = "gpu", .n_gpu_layers = -1 },
    .{ .name = "cpu", .n_gpu_layers = 0 },
    .{ .name = "partial", .n_gpu_layers = 10 },
};

/// How many copies of each good adapter the port loads to find pointers
/// whose order matches the C++'s.
const n_cand = 6;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: adapter_diff <tmpdir> <model.gguf> [--quick]\n", .{});
        std.process.exit(2);
    }
    const dir = args[1];
    const path = try gpa.dupeSentinel(u8, args[2], 0);
    var quick = false;
    for (args[3..]) |a| {
        if (std.mem.eql(u8, a, "--quick")) quick = true;
    }
    const out_dir: ?[]const u8 = if (std.c.getenv("ADAPTER_DIFF_OUT")) |d| std.mem.span(d) else null;

    c.llama_log_set(logCb, null);
    c.ggml_log_set(logCb, null);
    c.llama_backend_init();

    computeEffective();
    sweepOrder();
    std.debug.print("  {s:<40} {s:<4}\n", .{ "order sweep", if (failures == 0) "ok" else "FAIL" });

    const info = readInfo(path);
    var files: std.ArrayList(Lf) = .empty;
    writeFiles(dir, &info, &files);

    // Control-vector data: layers 1.. of n_embd each, and a short one.
    const n_layer: i32 = @intCast(info.n_layer);
    const ne: usize = @intCast(info.n_embd);
    cvec_full = try gpa.alloc(f32, ne * @as(usize, @intCast(n_layer - 1)));
    for (cvec_full) |*v| v.* = rnd() * 0.25;
    cvec_short = try gpa.alloc(f32, ne * 3 + 7);
    for (cvec_short) |*v| v.* = rnd() * 0.5;

    for (model_cfgs) |mc| {
        if (quick and !std.mem.eql(u8, mc.name, "gpu")) continue;
        var mp = c.llama_model_default_params();
        mp.n_gpu_layers = mc.n_gpu_layers;
        const rm = c.llama_model_load_from_file(path.ptr, mp) orelse {
            std.debug.print("FAIL: the C++ cannot load {s}\n", .{path});
            std.process.exit(1);
        };
        var splits: std.ArrayList([:0]const u8) = .empty;
        const pm = M.loadFromFileImpl(.{ .path = path }, &splits, null, null, mp) orelse {
            std.debug.print("FAIL: the port cannot load {s}\n", .{path});
            std.process.exit(1);
        };
        const pmc: ?*c.struct_llama_model = @ptrCast(pm);

        const vocab = c.llama_model_get_vocab(rm);
        var tokens: [256]i32 = undefined;
        const nt = c.llama_tokenize(vocab, prompt_text, prompt_text.len, &tokens, tokens.len, true, false);
        const n_vocab: usize = @intCast(c.llama_vocab_n_tokens(vocab));

        // A first context initializes what each backend initializes once.
        {
            var p = c.llama_context_default_params();
            p.n_ctx = 256;
            if (c.llama_init_from_model(rm, p)) |ctx| c.llama_free(ctx);
        }

        // ---- loading
        var ref: std.ArrayList(u8) = .empty;
        defer ref.deinit(gpa);
        var port: std.ArrayList(u8) = .empty;
        defer port.deinit(gpa);

        var ref_h: [n_good]Handle = undefined;
        rec = &ref;
        {
            const A = Api(false);
            var k: usize = 0;
            for (files.items) |f| {
                note("load {s}", .{f.name});
                const h = A.lora_init(rm, f.path.ptr);
                note("load {s}: {s}", .{ f.name, if (h == null) "null" else "ok" });
                if (f.good) {
                    ref_h[k] = h;
                    k += 1;
                } else if (f.keep) {
                    query(A, h, f.name);
                } else if (h != null) A.lora_free(h);
            }
            for (files.items[0..n_good], 0..) |f, i| query(A, ref_h[i], f.name);
            note("alora null: {d}", .{A.alora_n(null)});
            A.lora_free(null);
            // One more, freed by hand, which detaches it from the model.
            const d = A.lora_init(rm, files.items[0].path.ptr);
            A.lora_free(d);
            note("freed d", .{});
        }

        var cand: [n_good][n_cand]Handle = undefined;
        rec = &port;
        {
            const A = Api(true);
            var k: usize = 0;
            for (files.items) |f| {
                note("load {s}", .{f.name});
                if (f.good) {
                    var logs: [n_cand]std.ArrayList(u8) = @splat(.empty);
                    for (0..n_cand) |j| {
                        rec = &logs[j];
                        cand[k][j] = A.lora_init(pmc, f.path.ptr);
                    }
                    rec = &port;
                    port.appendSlice(gpa, logs[0].items) catch unreachable;
                    for (1..n_cand) |j| {
                        checks += 1;
                        if (!std.mem.eql(u8, logs[0].items, logs[j].items)) fail("{s}: loading {s} again logged differently", .{ mc.name, f.name });
                    }
                    for (&logs) |*l| l.deinit(gpa);
                    note("load {s}: {s}", .{ f.name, if (cand[k][0] == null) "null" else "ok" });
                    k += 1;
                } else {
                    const h = A.lora_init(pmc, f.path.ptr);
                    note("load {s}: {s}", .{ f.name, if (h == null) "null" else "ok" });
                    if (f.keep) {
                        query(A, h, f.name);
                    } else if (h != null) A.lora_free(h);
                }
            }
        }

        // Choose one copy of each whose real-template order matches the
        // C++'s at every step that sets several adapters.
        var pick: [n_good]usize = @splat(0);
        var found = false;
        var combo: usize = 0;
        const n_combo = std.math.pow(usize, n_cand, n_good);
        search: while (combo < n_combo) : (combo += 1) {
            var x = combo;
            for (&pick) |*p| {
                p.* = x % n_cand;
                x /= n_cand;
            }
            var ph: [n_good]Handle = undefined;
            for (0..n_good) |i| ph[i] = cand[i][pick[i]];
            for (steps, 0..) |st, si| switch (st) {
                .lora => {
                    var o1: [16]usize = undefined;
                    var o2: [16]usize = undefined;
                    const n1 = trueOrder(&ref_h, effective[si], &o1);
                    const n2 = trueOrder(&ph, effective[si], &o2);
                    if (n1 != n2 or !std.mem.eql(usize, o1[0..n1], o2[0..n2])) continue :search;
                },
                else => {},
            };
            found = true;
            break;
        }
        checks += 1;
        if (!found) fail("{s}: no combination of the port's adapters orders as the C++'s", .{mc.name});
        var port_h: [n_good]Handle = undefined;
        for (0..n_good) |i| port_h[i] = cand[i][pick[i]];
        {
            rec = null;
            const A = Api(true);
            for (0..n_good) |i| for (0..n_cand) |j| if (j != pick[i]) A.lora_free(cand[i][j]);
            rec = &port;
            for (files.items[0..n_good], 0..) |f, i| query(A, port_h[i], f.name);
            note("alora null: {d}", .{A.alora_n(null)});
            A.lora_free(null);
            const d = A.lora_init(pmc, files.items[0].path.ptr);
            A.lora_free(d);
            note("freed d", .{});
            checks += 1;
            if (pm.loras.items.len != n_good + 1) fail("{s}: the port's model holds {d} adapters, not {d}", .{ mc.name, pm.loras.items.len, n_good + 1 });
        }
        rec = null;
        compare(init.io, out_dir, mc.name, "load", ref.items, port.items, 0);

        // ---- the script
        // An adapter one side failed to load has already failed the load
        // comparison; the script would only dereference it.
        const loaded = for (0..n_good) |i| {
            if (ref_h[i] == null or port_h[i] == null) break false;
        } else true;
        checks += 1;
        if (!loaded) {
            fail("{s}: an adapter did not load; the script is skipped", .{mc.name});
            c.llama_model_free(rm);
            M.destroy(pm);
            continue;
        }
        ref.clearRetainingCapacity();
        port.clearRetainingCapacity();
        n_nodes_seen = 0;
        rec = &ref;
        script(false, rm, &ref_h, tokens[0..@intCast(nt)], n_vocab, n_layer);
        const ref_nodes = n_nodes_seen;
        rec = &port;
        script(true, pmc, &port_h, tokens[0..@intCast(nt)], n_vocab, n_layer);
        rec = null;
        compare(init.io, out_dir, mc.name, "script", ref.items, port.items, ref_nodes);

        // The script must have seen the adapters change the result: the
        // C++'s logits differ between the first decode and each adapted one.
        vacuity(mc.name, ref.items);

        c.llama_model_free(rm);
        M.destroy(pm);
        for (splits.items) |s| gpa.free(s);
        splits.deinit(gpa);
    }

    std.debug.print("adapter-diff: {d} checks, {d} failures\n", .{ checks, failures });
    if (failures != 0) std.process.exit(1);
}

fn compare(io: std.Io, out_dir: ?[]const u8, cfg: []const u8, what: []const u8, ref: []const u8, port: []const u8, nodes: usize) void {
    var rn = normalize(ref);
    defer rn.deinit(gpa);
    var pn = normalize(port);
    defer pn.deinit(gpa);
    if (out_dir) |d| {
        var pb: [512]u8 = undefined;
        const rp = std.fmt.bufPrint(&pb, "{s}/{s}-{s}.ref", .{ d, cfg, what }) catch unreachable;
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = rp, .data = rn.items }) catch {};
        const pp = std.fmt.bufPrint(&pb, "{s}/{s}-{s}.port", .{ d, cfg, what }) catch unreachable;
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = pp, .data = pn.items }) catch {};
    }
    var nb: [128]u8 = undefined;
    const name = std.fmt.bufPrint(&nb, "{s}: {s}", .{ cfg, what }) catch unreachable;
    const before = failures;
    firstDiff(name, rn.items, pn.items);
    const lines = std.mem.count(u8, rn.items, "\n");
    std.debug.print("  {s:<40} {s:<4} {d:>8} lines {d:>8} nodes\n", .{ name, if (failures == before) "ok" else "FAIL", lines, nodes });
}

/// Every adapted decode must give logits the unadapted one did not, or the
/// adapters are not reaching the graph and the comparison proves nothing.
fn vacuity(cfg: []const u8, ref: []const u8) void {
    var it = std.mem.splitScalar(u8, ref, '\n');
    var step: usize = 0;
    var baseline: ?[]const u8 = null;
    var seen: std.ArrayList([]const u8) = .empty;
    defer seen.deinit(gpa);
    var first_in_step = true;
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, line, "step ")) {
            step = std.fmt.parseInt(usize, line[5..], 10) catch 0;
            first_in_step = true;
            continue;
        }
        if (!std.mem.startsWith(u8, line, "logits ") or !first_in_step) continue;
        first_in_step = false;
        if (baseline == null) {
            baseline = line;
            continue;
        }
        seen.append(gpa, line) catch unreachable;
        // Steps 12 and 25 decode with nothing active -- after the all-zero
        // set, and after the null set with the control vector cleared --
        // and must give the unadapted logits back; every other must not.
        const expect_changed = step != 12 and step != 25;
        checks += 1;
        if (expect_changed == std.mem.eql(u8, line, baseline.?)) {
            fail("{s}: step {d} decoded {s} the unadapted decode", .{ cfg, step, if (expect_changed) "exactly as" else "differently from" });
        }
    }
}
