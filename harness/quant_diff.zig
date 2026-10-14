//! Compares `src/llama/quant.zig` and `src/llama/model_saver.zig` with
//! llama.cpp's `llama-quant.cpp` and `llama-model-saver.cpp`.
//!
//! **Not a port.** The harness behind `scripts/quant-diff`. Everything runs
//! in one process, the C++ in `libllama.a` beside the port compiled through
//! `harness_root.zig`, both converting through the same `ggml_quantize_chunk`:
//!
//! 1. **The saver's layer walk**: every entry of `layer_fields` is at
//!    `8 * index` in the C++'s `llama_layer`, and there are
//!    `sizeof(llama_layer) / 8` of them.
//! 2. **The type rules, on every architecture.** `llama-ext.h`'s helpers
//!    build a model from a descriptor and compute every tensor's type,
//!    so the rules run on architectures whose hyperparameters the port
//!    cannot load: every architecture, every file type, descriptors that
//!    reach each branch (GQA, expert counts, 70B), tensor lists with odd
//!    shapes, tied and untied embeddings, experts, and the parameters
//!    that steer selection. Compared: each type, what was thrown, and
//!    everything logged.
//! 3. **Whole files.** `llama_model_quantize` through the C++ and through
//!    the port on synthesized Qwen3.5 files and the local models, into
//!    separate outputs: the return value, everything logged, and every
//!    output file byte for byte.
//! 4. **The saver.** Each local model loaded through both and written by
//!    `llama_model_save_to_file`, byte for byte.
//!
//! Usage: quant_diff <tmpdir> [--quick] [--skip-big] [--only section] [model.gguf...]

const std = @import("std");
const harness_root = @import("harness_root");

const c = harness_root.impl.c;
const quant = harness_root.quant;
const saver = harness_root.model_saver;
const api = harness_root.api.quant;
const M = harness_root.model;

const gpa = std.heap.c_allocator;

var checks: usize = 0;
var failures: usize = 0;
const max_reported = 40;

fn fail(comptime fmt: []const u8, args: anytype) void {
    failures += 1;
    if (failures <= max_reported) std.debug.print("FAIL " ++ fmt ++ "\n", args);
}

// -----------------------------------------------------------------------------
// The C++ side

const QDesc = quant.ModelDesc;
extern fn qref_model(desc: *const QDesc, err: [*]u8, n_err: usize) ?*c.struct_llama_model;
extern fn qref_compute(model: *const c.struct_llama_model, params: *const c.llama_model_quantize_params, ftype: c_int, tensors: [*]const *c.ggml_tensor, result: [*]c_int, n: usize, err: [*]u8, n_err: usize) c_int;
extern fn qref_allows(model: *const c.struct_llama_model, params: *const c.llama_model_quantize_params, t: *const c.ggml_tensor) c_int;
extern fn qref_default_type(ftype: c_int) c_int;
extern fn qref_layer_size() usize;
extern fn qref_layer_offset(i: usize) usize;
extern fn qref_n_layer_offsets() usize;

// -----------------------------------------------------------------------------
// The transcript

var rec: ?*std.ArrayList(u8) = null;

fn logCb(level: c.enum_ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    const r = rec orelse return;
    r.print(gpa, "{d}|", .{level}) catch unreachable;
    r.appendSlice(gpa, std.mem.span(@as([*:0]const u8, @ptrCast(text)))) catch unreachable;
}

fn firstDiff(name: []const u8, a: []const u8, b: []const u8) bool {
    var ia = std.mem.splitScalar(u8, a, '\n');
    var ib = std.mem.splitScalar(u8, b, '\n');
    var line: usize = 1;
    while (true) : (line += 1) {
        const la = ia.next();
        const lb = ib.next();
        if (la == null and lb == null) return true;
        checks += 1;
        if (la == null or lb == null or !std.mem.eql(u8, la.?, lb.?)) {
            fail("{s}: transcript line {d}:\n    ref  '{s}'\n    port '{s}'", .{ name, line, la orelse "<end>", lb orelse "<end>" });
            return false;
        }
    }
}

// -----------------------------------------------------------------------------
// 1. The layer walk

fn checkLayerWalk() void {
    const n = saver.layer_fields.len;
    checks += 1;
    if (qref_n_layer_offsets() != n) fail("layer_fields has {d} entries, the generated offsets {d}", .{ n, qref_n_layer_offsets() });
    checks += 1;
    if (qref_layer_size() != n * @sizeOf(usize)) fail("sizeof(llama_layer) is {d} pointers, layer_fields names {d}", .{ qref_layer_size() / 8, n });
    for (0..@min(n, qref_n_layer_offsets())) |i| {
        checks += 1;
        if (qref_layer_offset(i) != i * @sizeOf(usize)) fail("llama_layer.{s} is at {d}, layer_fields puts it at {d}", .{ saver.layer_fields[i], qref_layer_offset(i), i * 8 });
    }
}

// -----------------------------------------------------------------------------
// 2. The type rules

const valid_ftypes = [_]c_int{ 0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 40, 41 };

const Shape = struct { name: []const u8, ne: [4]i64, ty: c.enum_ggml_type = c.GGML_TYPE_F16 };

/// A list of tensors with no data, shaped like a model's.
const TensorList = struct {
    ctx: *c.ggml_context,
    items: std.ArrayList(*c.ggml_tensor) = .empty,

    fn init() TensorList {
        return .{ .ctx = c.ggml_init(.{ .mem_size = 64 << 20, .mem_buffer = null, .no_alloc = true }).? };
    }

    fn deinit(self: *TensorList) void {
        self.items.deinit(gpa);
        c.ggml_free(self.ctx);
    }

    fn add(self: *TensorList, name: []const u8, ne: [4]i64, ty: c.enum_ggml_type) void {
        const t = c.ggml_new_tensor(self.ctx, ty, 4, &ne).?;
        var buf: [64]u8 = undefined;
        const z = std.fmt.bufPrintSentinel(&buf, "{s}", .{name}, 0) catch unreachable;
        _ = c.ggml_set_name(t, z.ptr);
        self.items.append(gpa, t) catch unreachable;
    }

    fn addf(self: *TensorList, comptime fmt: []const u8, args: anytype, ne: [4]i64, ty: c.enum_ggml_type) void {
        var buf: [64]u8 = undefined;
        self.add(std.fmt.bufPrint(&buf, fmt, args) catch unreachable, ne, ty);
    }
};

const Variant = enum { dense_tied, untied_odd, moe, no_layer, bad_layer };

fn buildList(v: Variant, n_layer: u32, n_expert: u32) TensorList {
    var l = TensorList.init();
    const e: i64 = @max(2, n_expert);
    const n_embd: i64 = 4096;
    const h16 = c.GGML_TYPE_F16;
    const f32_ = c.GGML_TYPE_F32;
    // Few enough layers that the 80-layer descriptors stay quick.
    const L: u32 = n_layer;
    switch (v) {
        .dense_tied, .no_layer, .bad_layer => {
            l.add("token_embd.weight", .{ n_embd, 32000, 1, 1 }, h16);
            l.add("output_norm.weight", .{ n_embd, 1, 1, 1 }, f32_);
        },
        .untied_odd, .moe => {
            l.add("output.weight", .{ n_embd, 32000, 1, 1 }, h16);
            l.add("output_norm.weight", .{ n_embd, 1, 1, 1 }, f32_);
            l.add("token_embd.weight", .{ n_embd, 32000, 1, 1 }, h16);
            l.add("per_layer_token_embd.weight", .{ 8960, 262144, 1, 1 }, h16);
            l.add("position_embd.weight", .{ n_embd, 2048, 1, 1 }, h16);
            l.add("token_types.weight", .{ n_embd, 2, 1, 1 }, h16);
            l.add("abcde", .{ 256, 256, 1, 1 }, h16);
            l.add("w", .{ 256, 64, 1, 1 }, f32_);
        },
    }
    var i: u32 = 0;
    while (i < L) : (i += 1) {
        l.addf("blk.{d}.attn_norm.weight", .{i}, .{ n_embd, 1, 1, 1 }, f32_);
        switch (v) {
            .dense_tied, .no_layer, .bad_layer => {
                l.addf("blk.{d}.attn_q.weight", .{i}, .{ n_embd, n_embd, 1, 1 }, h16);
                l.addf("blk.{d}.attn_k.weight", .{i}, .{ n_embd, 1024, 1, 1 }, h16);
                l.addf("blk.{d}.attn_v.weight", .{i}, .{ n_embd, 1024, 1, 1 }, h16);
                l.addf("blk.{d}.attn_output.weight", .{i}, .{ n_embd, n_embd, 1, 1 }, h16);
                l.addf("blk.{d}.ffn_gate.weight", .{i}, .{ n_embd, 11008, 1, 1 }, h16);
                l.addf("blk.{d}.ffn_up.weight", .{i}, .{ n_embd, 11008, 1, 1 }, h16);
                l.addf("blk.{d}.ffn_down.weight", .{i}, .{ 11008, n_embd, 1, 1 }, h16);
            },
            .untied_odd => {
                if (i % 2 == 0) {
                    l.addf("blk.{d}.attn_qkv.weight", .{i}, .{ n_embd, 6144, 1, 1 }, h16);
                } else {
                    l.addf("blk.{d}.attn_q.weight", .{i}, .{ n_embd, n_embd, 1, 1 }, h16);
                    l.addf("blk.{d}.attn_k.weight", .{i}, .{ n_embd, 1024, 1, 1 }, h16);
                    l.addf("blk.{d}.attn_v.weight", .{i}, .{ 1056, 1024, 1, 1 }, h16);
                }
                if (i == 0) {
                    l.addf("blk.{d}.attn_kv_b.weight", .{i}, .{ 512, 4096, 1, 1 }, h16);
                    l.addf("blk.{d}.ffn_gate_inp.weight", .{i}, .{ n_embd, 8, 1, 1 }, f32_);
                    l.addf("blk.{d}.ssm_conv1d.weight", .{i}, .{ 4, 8192, 1, 1 }, f32_);
                    l.addf("blk.{d}.altup_proj.weight", .{i}, .{ 2048, 2048, 1, 1 }, h16);
                    l.addf("blk.{d}.time_mix_w1.weight", .{i}, .{ 2048, 160, 1, 1 }, h16);
                    l.addf("blk.{d}.attn_q_norm.weight", .{i}, .{ 128, 1, 1, 1 }, f32_);
                    l.addf("blk.{d}.post_ffw_norm.weight", .{i}, .{ 256, 2, 1, 1 }, f32_);
                    l.addf("blk.{d}.ffn_down.bias", .{i}, .{ 256, 2, 1, 1 }, f32_);
                    l.addf("blk.{d}.attn_output.weight.x", .{i}, .{ 256, 2, 1, 1 }, h16);
                    l.addf("blk.{d}.int.weight", .{i}, .{ 256, 2, 1, 1 }, c.GGML_TYPE_I32);
                }
                l.addf("blk.{d}.attn_output.weight", .{i}, .{ if (i % 3 == 1) 40 else n_embd, n_embd, 1, 1 }, h16);
                l.addf("blk.{d}.ffn_gate.weight", .{i}, .{ n_embd, 13824, 1, 1 }, h16);
                l.addf("blk.{d}.ffn_up.weight", .{i}, .{ if (i % 4 == 3) 2000 else n_embd, 13824, 1, 1 }, h16);
                l.addf("blk.{d}.ffn_down.weight", .{i}, .{ if (i % 2 == 0) 13824 else 1000, n_embd, 1, 1 }, h16);
            },
            .moe => {
                l.addf("blk.{d}.attn_q.weight", .{i}, .{ n_embd, n_embd, 1, 1 }, h16);
                l.addf("blk.{d}.attn_k.weight", .{i}, .{ n_embd, 512, 1, 1 }, h16);
                l.addf("blk.{d}.attn_v.weight", .{i}, .{ n_embd, 512, 1, 1 }, h16);
                l.addf("blk.{d}.attn_output.weight", .{i}, .{ n_embd, n_embd, 1, 1 }, h16);
                l.addf("blk.{d}.ffn_gate_inp.weight", .{i}, .{ n_embd, e, 1, 1 }, f32_);
                l.addf("blk.{d}.ffn_gate_exps.weight", .{i}, .{ n_embd, 1536, e, 1 }, h16);
                l.addf("blk.{d}.ffn_up_exps.weight", .{i}, .{ n_embd, 1536, e, 1 }, h16);
                l.addf("blk.{d}.ffn_down_exps.weight", .{i}, .{ 1536, n_embd, e, 1 }, h16);
                l.addf("blk.{d}.ffn_gate_shexp.weight", .{i}, .{ n_embd, 1536, 1, 1 }, h16);
                l.addf("blk.{d}.ffn_down_shexp.weight", .{i}, .{ 1536, n_embd, 1, 1 }, h16);
                l.addf("blk.{d}.attn_kv_b.weight", .{i}, .{ 512, 4096, 1, 1 }, h16);
                l.addf("blk.{d}.attn_q_b.weight", .{i}, .{ 1536, 4096, 2, 1 }, h16);
            },
        }
    }
    switch (v) {
        .no_layer => l.add("ffn_down_extra.weight", .{ 11008, n_embd, 1, 1 }, h16),
        .bad_layer => l.addf("blk.{d}.ffn_up.weight", .{L}, .{ n_embd, 11008, 1, 1 }, h16),
        else => {},
    }
    return l;
}

const ParamSet = struct {
    name: []const u8,
    edit: *const fn (*c.llama_model_quantize_params) void,
};

const tt_main = [_]c.struct_llama_model_tensor_override{
    .{ .pattern = "attn_v", .type = c.GGML_TYPE_Q8_0 },
    .{ .pattern = "ffn_down.*\\.weight$", .type = c.GGML_TYPE_Q5_K },
    .{ .pattern = "blk\\.1\\.", .type = c.GGML_TYPE_IQ4_XS },
    .{ .pattern = null, .type = 0 },
};
const tt_bad = [_]c.struct_llama_model_tensor_override{
    .{ .pattern = "attn_v", .type = c.GGML_TYPE_Q8_0 },
    .{ .pattern = "attn_(", .type = c.GGML_TYPE_Q8_0 },
    .{ .pattern = null, .type = 0 },
};
const tt_same = [_]c.struct_llama_model_tensor_override{
    .{ .pattern = "ffn_up", .type = c.GGML_TYPE_Q4_K },
    .{ .pattern = "nothing-matches-this", .type = c.GGML_TYPE_Q2_K },
    .{ .pattern = null, .type = 0 },
};

const Edits = struct {
    fn none(_: *c.llama_model_quantize_params) void {}
    fn pure(p: *c.llama_model_quantize_params) void {
        p.pure = true;
    }
    fn outEmb(p: *c.llama_model_quantize_params) void {
        p.output_tensor_type = c.GGML_TYPE_Q8_0;
        p.token_embedding_type = c.GGML_TYPE_Q4_K;
    }
    fn outF16(p: *c.llama_model_quantize_params) void {
        p.output_tensor_type = c.GGML_TYPE_F16;
    }
    fn noOut(p: *c.llama_model_quantize_params) void {
        p.quantize_output_tensor = false;
    }
    fn tt(p: *c.llama_model_quantize_params) void {
        p.tt_overrides = &tt_main;
    }
    fn onlyCopy(p: *c.llama_model_quantize_params) void {
        p.only_copy = true;
    }
    fn ttBad(p: *c.llama_model_quantize_params) void {
        p.tt_overrides = &tt_bad;
    }
    fn ttSame(p: *c.llama_model_quantize_params) void {
        p.tt_overrides = &tt_same;
        p.pure = true;
    }
};

const param_sets = [_]ParamSet{
    .{ .name = "default", .edit = Edits.none },
    .{ .name = "pure", .edit = Edits.pure },
    .{ .name = "out+emb", .edit = Edits.outEmb },
    .{ .name = "outF16", .edit = Edits.outF16 },
    .{ .name = "noOut", .edit = Edits.noOut },
    .{ .name = "tt", .edit = Edits.tt },
    .{ .name = "onlyCopy", .edit = Edits.onlyCopy },
    .{ .name = "ttBad", .edit = Edits.ttBad },
    .{ .name = "ttSame", .edit = Edits.ttSame },
};

var type_calls: usize = 0;
var type_rc = [3]usize{ 0, 0, 0 };
var type_logged: usize = 0;

/// One `llama_quant_compute_types` on both sides.
fn computeBoth(name: []const u8, rm: *c.struct_llama_model, pm: *M.Model, params: *const c.llama_model_quantize_params, ftype: c_int, list: *const TensorList) void {
    const n = list.items.items.len;
    type_calls += 1;

    var ref: std.ArrayList(u8) = .empty;
    defer ref.deinit(gpa);
    var port: std.ArrayList(u8) = .empty;
    defer port.deinit(gpa);

    const types = gpa.alloc(c_int, n) catch unreachable;
    defer gpa.free(types);
    var err: [1024]u8 = undefined;

    rec = &ref;
    @memset(types, -1);
    const rc = qref_compute(rm, params, ftype, list.items.items.ptr, types.ptr, n, &err, err.len);
    type_rc[@intCast(rc)] += 1;
    if (std.mem.find(u8, ref.items, "|") != null) type_logged += 1;
    ref.print(gpa, "\nrc {d}", .{rc}) catch unreachable;
    if (rc != 0) ref.print(gpa, " {s}", .{std.mem.sliceTo(&err, 0)}) catch unreachable;
    for (types) |t| ref.print(gpa, " {d}", .{t}) catch unreachable;

    rec = &port;
    @memset(types, -1);
    var exc: harness_root.impl.Exception = .{};
    var prc: c_int = 0;
    if (quant.quantInit(pm, params, &exc)) |qs| {
        defer quant.quantFree(qs);
        quant.quantComputeTypes(qs, @intCast(ftype), list.items.items, @ptrCast(types), &exc) catch {
            prc = 1;
        };
    } else |_| prc = 2;
    port.print(gpa, "\nrc {d}", .{prc}) catch unreachable;
    if (prc != 0) port.print(gpa, " {s}", .{exc.what()}) catch unreachable;
    for (types) |t| port.print(gpa, " {d}", .{t}) catch unreachable;
    rec = null;

    checks += n;
    if (!std.mem.eql(u8, ref.items, port.items)) _ = firstDiff(name, ref.items, port.items);
}

fn allowsBoth(name: []const u8, rm: *c.struct_llama_model, pm: *M.Model, params: *const c.llama_model_quantize_params, list: *const TensorList) void {
    var exc: harness_root.impl.Exception = .{};
    const qs = quant.quantInit(pm, params, &exc) catch return;
    defer quant.quantFree(qs);
    for (list.items.items) |t| {
        checks += 1;
        const r = qref_allows(rm, params, t);
        const p: c_int = @intFromBool(quant.quantTensorAllowsQuantization(qs, t));
        if (r != p) fail("{s}: allows {s}: ref {d} port {d}", .{ name, std.mem.sliceTo(&t.name, 0), r, p });
    }
}

const DescCfg = struct { n_layer: u32, n_head: u32, n_head_kv: u32, n_expert: u32 };

fn makeModels(arch_name: [:0]const u8, d: DescCfg, rm_out: *?*c.struct_llama_model, pm_out: *?*M.Model) bool {
    const desc: QDesc = .{
        .architecture = arch_name.ptr,
        .n_embd = 4096,
        .n_ff = 11008,
        .n_layer = d.n_layer,
        .n_head = d.n_head,
        .n_head_kv = d.n_head_kv,
        .n_expert = d.n_expert,
        .n_embd_head_k = 128,
        .n_embd_head_v = 128,
    };
    var err: [1024]u8 = undefined;
    err[0] = 0;
    rm_out.* = qref_model(&desc, &err, err.len);
    var exc: harness_root.impl.Exception = .{};
    pm_out.* = quant.modelFromMetadata(&desc, &exc) catch null;
    checks += 1;
    if ((rm_out.* == null) != (pm_out.* == null)) {
        fail("model {s}: ref {s} port {s}", .{ arch_name, if (rm_out.* == null) std.mem.sliceTo(&err, 0) else "ok", if (pm_out.* == null) exc.what() else "ok" });
        return false;
    }
    if (rm_out.* == null) {
        checks += 1;
        if (!std.mem.eql(u8, std.mem.sliceTo(&err, 0), exc.what())) fail("model {s}: ref '{s}' port '{s}'", .{ arch_name, std.mem.sliceTo(&err, 0), exc.what() });
        return false;
    }
    return true;
}

fn freeModels(rm: ?*c.struct_llama_model, pm: ?*M.Model) void {
    if (rm) |m| c.llama_model_free(m);
    if (pm) |m| {
        m.deinit();
        gpa.destroy(m);
    }
}

fn typeRules(quick: bool) void {
    // Every architecture, one descriptor, every file type, two parameter
    // sets.
    {
        var lists = [_]TensorList{ buildList(.dense_tied, 8, 0), buildList(.untied_odd, 8, 0), buildList(.moe, 8, 8) };
        defer for (&lists) |*l| l.deinit();
        var n_arch: usize = 0;
        for (harness_root.arch.archAll()) |a| {
            const an = harness_root.arch.archName(a);
            var rm: ?*c.struct_llama_model = null;
            var pm: ?*M.Model = null;
            const ok = makeModels(an, .{ .n_layer = 8, .n_head = 32, .n_head_kv = 8, .n_expert = 0 }, &rm, &pm);
            defer freeModels(rm, pm);
            if (!ok) continue;
            n_arch += 1;
            for (valid_ftypes) |ft| {
                for ([_]usize{ 0, 1 }) |ps| {
                    for (&lists, 0..) |*l, li| {
                        if (quick and li != 1) continue;
                        var params = c.llama_model_quantize_default_params();
                        param_sets[ps].edit(&params);
                        var nb: [128]u8 = undefined;
                        const nm = std.fmt.bufPrint(&nb, "types {s} ftype {d} {s} list {d}", .{ an, ft, param_sets[ps].name, li }) catch unreachable;
                        computeBoth(nm, rm.?, pm.?, &params, ft, l);
                    }
                }
            }
        }
        std.debug.print("  type rules, every architecture: {d} architectures\n", .{n_arch});
    }

    // The architectures the rules name, under every descriptor, parameter
    // set and list.
    const archs = [_][:0]const u8{ "llama", "falcon", "bailingmoe3", "qwen35", "qwen2", "deepseek2" };
    const layer_counts = [_]u32{ 1, 8, 32, 80 };
    const heads = [_][2]u32{ .{ 32, 32 }, .{ 32, 8 }, .{ 32, 4 }, .{ 32, 0 } };
    const experts = [_]u32{ 0, 4, 8, 16 };
    const variants = [_]Variant{ .dense_tied, .untied_odd, .moe, .no_layer, .bad_layer };
    for (archs) |an| {
        for (layer_counts) |nl| {
            if (quick and nl != 8 and nl != 80) continue;
            for (heads) |h| {
                for (experts) |ne| {
                    var rm: ?*c.struct_llama_model = null;
                    var pm: ?*M.Model = null;
                    const ok = makeModels(an, .{ .n_layer = nl, .n_head = h[0], .n_head_kv = h[1], .n_expert = ne }, &rm, &pm);
                    defer freeModels(rm, pm);
                    if (!ok) continue;
                    for (variants) |v| {
                        var l = buildList(v, nl, ne);
                        defer l.deinit();
                        for (param_sets, 0..) |ps, psi| {
                            var params = c.llama_model_quantize_default_params();
                            ps.edit(&params);
                            if (psi == 0 or psi == 4 or psi == 6) {
                                var nb: [128]u8 = undefined;
                                allowsBoth(std.fmt.bufPrint(&nb, "allows {s} {s}", .{ an, ps.name }) catch unreachable, rm.?, pm.?, &params, &l);
                            }
                            for (valid_ftypes) |ft| {
                                if (quick and psi > 1 and ft != 15 and ft != 23) continue;
                                if (v == .no_layer or v == .bad_layer) {
                                    if (psi != 0) continue;
                                }
                                var nb: [160]u8 = undefined;
                                const nm = std.fmt.bufPrint(&nb, "types {s} L{d} h{d}/{d} e{d} {s} {s} ftype {d}", .{ an, nl, h[0], h[1], ne, @tagName(v), ps.name, ft }) catch unreachable;
                                computeBoth(nm, rm.?, pm.?, &params, ft, &l);
                            }
                        }
                    }
                }
            }
        }
    }
    std.debug.print("  type rules: {d} type computations: {d} computed, {d} threw, {d} could not compile a pattern; {d} logged\n", .{ type_calls, type_rc[0], type_rc[1], type_rc[2], type_logged });
}

// -----------------------------------------------------------------------------
// 3. Whole files

const Rng = struct {
    s: u64,
    fn next(self: *Rng) u64 {
        self.s = self.s *% 6364136223846793005 +% 1442695040888963407;
        return self.s >> 11;
    }
    fn float(self: *Rng) f32 {
        const u: f32 = @floatFromInt(self.next() & 0xffffff);
        var x = (u / 16777216.0) * 2.0 - 1.0;
        if (self.next() % 97 == 0) x *= 40.0;
        return x;
    }
};

const FileSpec = struct {
    name: []const u8,
    /// The type 2D weights are stored as; `mixed` varies it per tensor.
    wtype: c.enum_ggml_type,
    mixed: bool = false,
    n_head: u32 = 4,
    n_head_kv: u32 = 1,
    n_expert: u32 = 0,
    tied: bool = true,
    n_layer: u32 = 8,
    /// Extra tensors for error paths.
    huge_blk: bool = false,
    odd_shape: bool = false,
    arch: [:0]const u8 = "qwen35",
};

const n_embd_f: i64 = 256;
const n_ff_f: i64 = 288;

fn kvKey(a: harness_root.arch.Arch, kv: harness_root.arch.Kv, buf: *[256]u8) [*:0]const u8 {
    const n = harness_root.arch.kvName(a, kv, null);
    @memcpy(buf[0..n.len], n.slice());
    buf[n.len] = 0;
    return @ptrCast(buf);
}

fn setMeta(g: *c.gguf_context, s: FileSpec) void {
    const a = harness_root.arch.archFromString(s.arch);
    var b: [256]u8 = undefined;
    c.gguf_set_val_str(g, "general.architecture", s.arch.ptr);
    c.gguf_set_val_str(g, "general.name", "synthetic");
    c.gguf_set_val_u32(g, kvKey(a, .context_length, &b), 4096);
    c.gguf_set_val_u32(g, kvKey(a, .embedding_length, &b), @intCast(n_embd_f));
    c.gguf_set_val_u32(g, kvKey(a, .block_count, &b), s.n_layer);
    c.gguf_set_val_u32(g, kvKey(a, .feed_forward_length, &b), @intCast(n_ff_f));
    c.gguf_set_val_u32(g, kvKey(a, .attention_head_count, &b), s.n_head);
    c.gguf_set_val_u32(g, kvKey(a, .attention_head_count_kv, &b), s.n_head_kv);
    c.gguf_set_val_f32(g, kvKey(a, .attention_layernorm_rms_eps, &b), 1e-6);
    const sections = [_]i32{ 11, 11, 10, 0 };
    c.gguf_set_arr_data(g, kvKey(a, .rope_dimension_sections, &b), c.GGUF_TYPE_INT32, &sections, 4);
    c.gguf_set_val_u32(g, kvKey(a, .ssm_conv_kernel, &b), 4);
    c.gguf_set_val_u32(g, kvKey(a, .ssm_inner_size, &b), 256);
    c.gguf_set_val_u32(g, kvKey(a, .ssm_state_size, &b), 32);
    c.gguf_set_val_u32(g, kvKey(a, .ssm_time_step_rank, &b), 8);
    c.gguf_set_val_u32(g, kvKey(a, .ssm_group_count, &b), 4);
    if (s.n_expert > 0) {
        c.gguf_set_val_u32(g, kvKey(a, .expert_count, &b), s.n_expert);
        c.gguf_set_val_u32(g, kvKey(a, .expert_used_count, &b), 2);
    }
    c.gguf_set_val_u32(g, "general.file_type", 0);
}

const SynthTensor = struct { name: []const u8, ne: [4]i64, ty: c.enum_ggml_type };

fn specTensors(s: FileSpec, out: *std.ArrayList(SynthTensor)) void {
    const w = s.wtype;
    const f32_ = c.GGML_TYPE_F32;
    const add = struct {
        fn f(o: *std.ArrayList(SynthTensor), name: []const u8, ne: [4]i64, ty: c.enum_ggml_type) void {
            o.append(gpa, .{ .name = gpa.dupe(u8, name) catch unreachable, .ne = ne, .ty = ty }) catch unreachable;
        }
    }.f;
    var nb: [64]u8 = undefined;
    add(out, "token_embd.weight", .{ n_embd_f, 64, 1, 1 }, w);
    if (!s.tied) add(out, "output.weight", .{ n_embd_f, 64, 1, 1 }, w);
    add(out, "output_norm.weight", .{ n_embd_f, 1, 1, 1 }, f32_);
    add(out, "abcde", .{ 64, 4, 1, 1 }, w);
    add(out, "position_embd.weight", .{ n_embd_f, 8, 1, 1 }, w);
    var i: u32 = 0;
    while (i < s.n_layer) : (i += 1) {
        const mixed_ty: c.enum_ggml_type = if (!s.mixed) w else switch (i % 4) {
            0 => c.GGML_TYPE_Q8_0,
            1 => c.GGML_TYPE_Q4_K,
            2 => c.GGML_TYPE_F16,
            else => c.GGML_TYPE_BF16,
        };
        add(out, std.fmt.bufPrint(&nb, "blk.{d}.attn_norm.weight", .{i}) catch unreachable, .{ n_embd_f, 1, 1, 1 }, f32_);
        if (i == 1) {
            add(out, std.fmt.bufPrint(&nb, "blk.{d}.attn_qkv.weight", .{i}) catch unreachable, .{ n_embd_f, 384, 1, 1 }, mixed_ty);
        } else {
            add(out, std.fmt.bufPrint(&nb, "blk.{d}.attn_q.weight", .{i}) catch unreachable, .{ n_embd_f, 256, 1, 1 }, mixed_ty);
            add(out, std.fmt.bufPrint(&nb, "blk.{d}.attn_k.weight", .{i}) catch unreachable, .{ n_embd_f, 64, 1, 1 }, mixed_ty);
            add(out, std.fmt.bufPrint(&nb, "blk.{d}.attn_v.weight", .{i}) catch unreachable, .{ n_embd_f, 64, 1, 1 }, mixed_ty);
        }
        add(out, std.fmt.bufPrint(&nb, "blk.{d}.attn_output.weight", .{i}) catch unreachable, .{ n_embd_f, 256, 1, 1 }, mixed_ty);
        if (s.n_expert > 0) {
            add(out, std.fmt.bufPrint(&nb, "blk.{d}.ffn_gate_inp.weight", .{i}) catch unreachable, .{ n_embd_f, s.n_expert, 1, 1 }, f32_);
            add(out, std.fmt.bufPrint(&nb, "blk.{d}.ffn_up_exps.weight", .{i}) catch unreachable, .{ n_embd_f, 64, s.n_expert, 1 }, mixed_ty);
            add(out, std.fmt.bufPrint(&nb, "blk.{d}.ffn_down_exps.weight", .{i}) catch unreachable, .{ 64 * 4, 64, s.n_expert, 1 }, mixed_ty);
        }
        add(out, std.fmt.bufPrint(&nb, "blk.{d}.ffn_gate.weight", .{i}) catch unreachable, .{ n_embd_f, n_ff_f, 1, 1 }, mixed_ty);
        add(out, std.fmt.bufPrint(&nb, "blk.{d}.ffn_up.weight", .{i}) catch unreachable, .{ n_embd_f, n_ff_f, 1, 1 }, mixed_ty);
        // n_ff is a multiple of 32 and not of 256: every K-type falls back.
        add(out, std.fmt.bufPrint(&nb, "blk.{d}.ffn_down.weight", .{i}) catch unreachable, .{ n_ff_f, n_embd_f, 1, 1 }, if (s.mixed) c.GGML_TYPE_F16 else w);
        if (i == 2) {
            add(out, std.fmt.bufPrint(&nb, "blk.{d}.ssm_conv1d.weight", .{i}) catch unreachable, .{ 4, 256, 1, 1 }, f32_);
            // A multiple of 32 and not of 256; with `odd_shape` not even of
            // 32, which only the K and IQ types can fall back from.
            add(out, std.fmt.bufPrint(&nb, "blk.{d}.ssm_out.weight", .{i}) catch unreachable, .{ if (s.odd_shape) 40 else 96, 256, 1, 1 }, w);
        }
    }
    if (s.huge_blk) add(out, "blk.99999999999.ffn_up.weight", .{ n_embd_f, 8, 1, 1 }, w);
}

fn writeTensorData(t: *c.ggml_tensor, rng: *Rng) void {
    const n: usize = @intCast(c.ggml_nelements(t));
    const tmp = gpa.alloc(f32, n) catch unreachable;
    defer gpa.free(tmp);
    for (tmp) |*x| x.* = rng.float();
    switch (t.type) {
        c.GGML_TYPE_F32 => @memcpy(@as([*]f32, @ptrCast(@alignCast(t.data)))[0..n], tmp),
        c.GGML_TYPE_F16 => c.ggml_fp32_to_fp16_row(tmp.ptr, @ptrCast(@alignCast(t.data)), @intCast(n)),
        c.GGML_TYPE_BF16 => c.ggml_fp32_to_bf16_row(tmp.ptr, @ptrCast(@alignCast(t.data)), @intCast(n)),
        c.GGML_TYPE_I32 => @memset(@as([*]u8, @ptrCast(t.data))[0..c.ggml_nbytes(t)], 1),
        else => _ = c.ggml_quantize_chunk(t.type, tmp.ptr, t.data, 0, c.ggml_nrows(t), t.ne[0], null),
    }
}

/// Writes `s` as one file, or as `n_split` splits; returns the path to pass.
fn writeSpec(dir: []const u8, s: FileSpec, n_split: u32) [:0]const u8 {
    var tl: std.ArrayList(SynthTensor) = .empty;
    defer {
        for (tl.items) |t| gpa.free(t.name);
        tl.deinit(gpa);
    }
    specTensors(s, &tl);
    var rng: Rng = .{ .s = std.hash.Wyhash.hash(0, s.name) };

    const ctx = c.ggml_init(.{ .mem_size = 256 << 20, .mem_buffer = null, .no_alloc = false }).?;
    defer c.ggml_free(ctx);
    var tensors: std.ArrayList(*c.ggml_tensor) = .empty;
    defer tensors.deinit(gpa);
    for (tl.items) |st| {
        const t = c.ggml_new_tensor(ctx, st.ty, 4, &st.ne).?;
        var nb: [64]u8 = undefined;
        _ = c.ggml_set_name(t, (std.fmt.bufPrintSentinel(&nb, "{s}", .{st.name}, 0) catch unreachable).ptr);
        writeTensorData(t, &rng);
        tensors.append(gpa, t) catch unreachable;
    }

    if (n_split <= 1) {
        const g = c.gguf_init_empty().?;
        defer c.gguf_free(g);
        setMeta(g, s);
        for (tensors.items) |t| c.gguf_add_tensor(g, t);
        const path = gpa.printSentinel("{s}/{s}.gguf", .{ dir, s.name }, 0) catch unreachable;
        if (!c.gguf_write_to_file(g, path.ptr, false)) @panic("cannot write a synthetic file");
        return path;
    }

    var first: [:0]const u8 = undefined;
    for (0..n_split) |i| {
        const g = c.gguf_init_empty().?;
        defer c.gguf_free(g);
        if (i == 0) setMeta(g, s);
        c.gguf_set_val_u16(g, "split.no", @intCast(i));
        c.gguf_set_val_u16(g, "split.count", @intCast(n_split));
        c.gguf_set_val_i32(g, "split.tensors.count", @intCast(tensors.items.len));
        // Tensors are dealt to the splits in turn, so every split holds
        // tensors from every part of the model.
        for (tensors.items, 0..) |t, k| if (k % n_split == i) c.gguf_add_tensor(g, t);
        var buf: [1024]u8 = undefined;
        const prefix = gpa.printSentinel("{s}/{s}", .{ dir, s.name }, 0) catch unreachable;
        defer gpa.free(prefix);
        const n = c.llama_split_path(&buf, buf.len, prefix.ptr, @intCast(i), @intCast(n_split));
        const path = gpa.dupeSentinel(u8, buf[0..@intCast(n)], 0) catch unreachable;
        if (!c.gguf_write_to_file(g, path.ptr, false)) @panic("cannot write a split");
        if (i == 0) first = path else gpa.free(path);
    }
    return first;
}

/// Synthesized importance data for every tensor of a file: one positive
/// value per column (per expert).
const Imatrix = struct {
    entries: std.ArrayList(c.struct_llama_model_imatrix_data) = .empty,
    names: std.ArrayList([:0]u8) = .empty,
    data: std.ArrayList([]f32) = .empty,

    fn fromFile(path: [:0]const u8, edit: ?*const fn (*Imatrix) void) Imatrix {
        var self: Imatrix = .{};
        var ctx: ?*c.ggml_context = null;
        const g = c.gguf_init_from_file(path.ptr, .{ .no_alloc = true, .ctx = &ctx }).?;
        defer c.gguf_free(g);
        defer c.ggml_free(ctx);
        var rng: Rng = .{ .s = 7 };
        var t: ?*c.ggml_tensor = c.ggml_get_first_tensor(ctx);
        while (t) |tt| : (t = c.ggml_get_next_tensor(ctx, tt)) {
            const n: usize = @intCast(tt.ne[0] * tt.ne[2]);
            const d = gpa.alloc(f32, n) catch unreachable;
            for (d) |*x| x.* = @abs(rng.float()) + 0.01;
            self.names.append(gpa, gpa.dupeSentinel(u8, std.mem.sliceTo(&tt.name, 0), 0) catch unreachable) catch unreachable;
            self.data.append(gpa, d) catch unreachable;
        }
        if (edit) |e| e(&self);
        for (self.names.items, self.data.items) |n, d| {
            self.entries.append(gpa, .{ .name = n.ptr, .data = d.ptr, .size = d.len }) catch unreachable;
        }
        self.entries.append(gpa, .{ .name = null, .data = null, .size = 0 }) catch unreachable;
        return self;
    }

    fn deinit(self: *Imatrix) void {
        for (self.names.items) |n| gpa.free(n);
        for (self.data.items) |d| gpa.free(d);
        self.names.deinit(gpa);
        self.data.deinit(gpa);
        self.entries.deinit(gpa);
    }

    fn find(self: *Imatrix, name: []const u8) ?usize {
        for (self.names.items, 0..) |n, i| if (std.mem.eql(u8, n, name)) return i;
        return null;
    }

    fn nan(self: *Imatrix) void {
        const i = self.find("blk.3.ffn_up.weight").?;
        self.data.items[i][5] = std.math.nan(f32);
    }
    fn inf(self: *Imatrix) void {
        const i = self.find("blk.0.attn_q.weight").?;
        self.data.items[i][0] = -std.math.inf(f32);
    }
    fn short(self: *Imatrix) void {
        const i = self.find("blk.4.attn_k.weight").?;
        self.data.items[i] = gpa.realloc(self.data.items[i], 10) catch unreachable;
    }
    fn shortEmbd(self: *Imatrix) void {
        const i = self.find("token_embd.weight").?;
        self.data.items[i] = gpa.realloc(self.data.items[i], 10) catch unreachable;
    }
    fn missing(self: *Imatrix) void {
        const i = self.find("blk.5.ffn_gate.weight").?;
        gpa.free(self.names.items[i]);
        self.names.items[i] = gpa.dupeSentinel(u8, "blk.5.ffn_gate.weightX", 0) catch unreachable;
    }
    fn duplicate(self: *Imatrix) void {
        self.names.append(gpa, gpa.dupeSentinel(u8, "blk.0.attn_q.weight", 0) catch unreachable) catch unreachable;
        const d = gpa.alloc(f32, 3) catch unreachable;
        @memset(d, 1);
        self.data.append(gpa, d) catch unreachable;
    }
};

var file_runs: usize = 0;

fn fileBytes(path: [:0]const u8) ?[]u8 {
    const f = c.fopen(path.ptr, "rb") orelse return null;
    defer _ = c.fclose(f);
    _ = c.fseeko(f, 0, c.SEEK_END);
    const size: usize = @intCast(c.ftello(f));
    _ = c.fseeko(f, 0, c.SEEK_SET);
    const buf = gpa.alloc(u8, size) catch unreachable;
    if (size > 0 and c.fread(buf.ptr, 1, size, f) != size) @panic("short read");
    return buf;
}

fn compareFiles(name: []const u8, ref_path: [:0]const u8, port_path: [:0]const u8) void {
    const a = fileBytes(ref_path);
    defer if (a) |x| gpa.free(x);
    const b = fileBytes(port_path);
    defer if (b) |x| gpa.free(x);
    checks += 1;
    if ((a == null) != (b == null)) {
        fail("{s}: output {s} exists on one side only (ref {}, port {})", .{ name, ref_path, a != null, b != null });
        return;
    }
    if (a == null) return;
    checks += a.?.len / 4096 + 1;
    if (!std.mem.eql(u8, a.?, b.?)) {
        const n = @min(a.?.len, b.?.len);
        var at: usize = n;
        for (0..n) |i| if (a.?[i] != b.?[i]) {
            at = i;
            break;
        };
        fail("{s}: {s} differs: sizes {d}/{d}, first difference at byte {d}", .{ name, ref_path, a.?.len, b.?.len, at });
    }
}

fn removePath(path: [:0]const u8) void {
    _ = std.c.unlink(path.ptr);
}

/// Every output a run may write: the file itself, or its splits.
fn outputPaths(base: [:0]const u8, out: *std.ArrayList([:0]u8)) void {
    out.append(gpa, gpa.dupeSentinel(u8, base, 0) catch unreachable) catch unreachable;
    var n: u32 = 1;
    while (n <= 4) : (n += 1) {
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            var buf: [1024]u8 = undefined;
            const k = c.llama_split_path(&buf, buf.len, base.ptr, @intCast(i), @intCast(n));
            out.append(gpa, gpa.dupeSentinel(u8, buf[0..@intCast(k)], 0) catch unreachable) catch unreachable;
        }
    }
}

fn runQuant(dir: []const u8, name: []const u8, inp: [:0]const u8, params: *const c.llama_model_quantize_params, out_override: ?[:0]const u8) void {
    file_runs += 1;
    const ref_out = out_override orelse (gpa.printSentinel("{s}/out-ref.gguf", .{dir}, 0) catch unreachable);
    const port_out = if (out_override) |o| o else (gpa.printSentinel("{s}/out-port.gguf", .{dir}, 0) catch unreachable);
    var ref_paths: std.ArrayList([:0]u8) = .empty;
    var port_paths: std.ArrayList([:0]u8) = .empty;
    defer {
        for (ref_paths.items) |p| gpa.free(p);
        for (port_paths.items) |p| gpa.free(p);
        ref_paths.deinit(gpa);
        port_paths.deinit(gpa);
    }
    outputPaths(ref_out, &ref_paths);
    outputPaths(port_out, &port_paths);
    for (ref_paths.items) |p| removePath(p);
    for (port_paths.items) |p| removePath(p);

    var ref: std.ArrayList(u8) = .empty;
    defer ref.deinit(gpa);
    var port: std.ArrayList(u8) = .empty;
    defer port.deinit(gpa);

    rec = &ref;
    const rc_ref = c.llama_model_quantize(inp.ptr, ref_out.ptr, params);
    ref.print(gpa, "\nrc {d}", .{rc_ref}) catch unreachable;
    // A path the C++ could not write is not there for the port to fail on
    // either; keep both outputs apart only where they can be.
    var ref_files: std.ArrayList(?[]u8) = .empty;
    defer {
        for (ref_files.items) |f| if (f) |x| gpa.free(x);
        ref_files.deinit(gpa);
    }
    if (out_override != null) {
        for (ref_paths.items) |p| ref_files.append(gpa, fileBytes(p)) catch unreachable;
        for (ref_paths.items) |p| removePath(p);
    }
    rec = &port;
    const rc_port = api.llama_model_quantize(inp.ptr, port_out.ptr, params);
    port.print(gpa, "\nrc {d}", .{rc_port}) catch unreachable;
    rec = null;

    checks += 1;
    if (rc_ref != rc_port) fail("{s}: rc ref {d} port {d}", .{ name, rc_ref, rc_port });
    _ = firstDiff(name, ref.items, port.items);
    if (out_override != null) {
        for (port_paths.items, ref_files.items) |p, rf| {
            const pf = fileBytes(p);
            defer if (pf) |x| gpa.free(x);
            checks += 1;
            if ((rf == null) != (pf == null) or (rf != null and !std.mem.eql(u8, rf.?, pf.?))) fail("{s}: {s} differs", .{ name, p });
        }
    } else {
        for (ref_paths.items, port_paths.items) |a, b| compareFiles(name, a, b);
    }
    for (ref_paths.items) |p| removePath(p);
    for (port_paths.items) |p| removePath(p);
    if (out_override == null) {
        gpa.free(ref_out);
        gpa.free(port_out);
    }
    if (std.c.getenv("QUANT_DIFF_VERBOSE") != null) {
        var why: []const u8 = "";
        if (std.mem.find(u8, ref.items, "failed to quantize: ")) |at| {
            const rest = ref.items[at + "failed to quantize: ".len ..];
            why = rest[0 .. std.mem.findScalar(u8, rest, '\n') orelse rest.len];
        }
        std.debug.print("    {s}: rc {d}, {d} log bytes {s}\n", .{ name, rc_ref, ref.items.len, why });
    }
}

const kv_overrides = blk: {
    var o: [6]c.struct_llama_model_kv_override = @splat(std.mem.zeroes(c.struct_llama_model_kv_override));
    const set = struct {
        fn key(x: *c.struct_llama_model_kv_override, k: []const u8) void {
            @memcpy(x.key[0..k.len], k);
        }
    };
    o[0].tag = c.LLAMA_KV_OVERRIDE_TYPE_FLOAT;
    set.key(&o[0], "general.some_float");
    o[0].unnamed_0.val_f64 = 0.1;
    o[1].tag = c.LLAMA_KV_OVERRIDE_TYPE_INT;
    set.key(&o[1], "general.some_int");
    o[1].unnamed_0.val_i64 = -5000000000;
    o[2].tag = c.LLAMA_KV_OVERRIDE_TYPE_BOOL;
    set.key(&o[2], "general.some_bool");
    o[2].unnamed_0.val_bool = true;
    o[3].tag = c.LLAMA_KV_OVERRIDE_TYPE_STR;
    set.key(&o[3], "general.name");
    @memcpy(o[3].unnamed_0.val_str[0..8], "override");
    o[4].tag = 77;
    set.key(&o[4], "general.weird");
    break :blk o;
};

const prune_mid = [_]i32{ 2, 5, -1 };
const prune_last = [_]i32{ 7, -1 };
const prune_first = [_]i32{ 0, -1 };

fn fileRuns(dir: []const u8, quick: bool) void {
    const specs = [_]FileSpec{
        .{ .name = "f32-gqa4-tied", .wtype = c.GGML_TYPE_F32 },
        .{ .name = "f16-gqa1-untied-moe8", .wtype = c.GGML_TYPE_F16, .n_head_kv = 4, .n_expert = 8, .tied = false },
        .{ .name = "bf16-gqa4-untied", .wtype = c.GGML_TYPE_BF16, .tied = false },
        .{ .name = "mixed-gqa1", .wtype = c.GGML_TYPE_F32, .mixed = true, .n_head_kv = 4, .tied = false },
    };
    var paths: [specs.len][:0]const u8 = undefined;
    for (specs, 0..) |s, i| paths[i] = writeSpec(dir, s, 1);
    defer for (paths) |p| gpa.free(p);

    var nb: [256]u8 = undefined;

    // Every file type on every file, with importance data, four threads.
    for (specs, paths) |s, p| {
        var im = Imatrix.fromFile(p, null);
        defer im.deinit();
        for (valid_ftypes) |ft| {
            if (quick and ft != 15 and ft != 19 and ft != 38) continue;
            var params = c.llama_model_quantize_default_params();
            params.ftype = @intCast(ft);
            params.nthread = 4;
            params.allow_requantize = true;
            params.imatrix = im.entries.items.ptr;
            runQuant(dir, std.fmt.bufPrint(&nb, "{s} ftype {d} imatrix", .{ s.name, ft }) catch unreachable, p, &params, null);
        }
    }
    std.debug.print("  files, every type with an imatrix: {d} runs\n", .{file_runs});

    // Without importance data: a dry run of every type, and the real
    // thing for the types that need none.
    const p0 = paths[0];
    for (valid_ftypes) |ft| {
        var params = c.llama_model_quantize_default_params();
        params.ftype = @intCast(ft);
        params.nthread = 2;
        params.dry_run = true;
        runQuant(dir, std.fmt.bufPrint(&nb, "dry ftype {d}", .{ft}) catch unreachable, p0, &params, null);
        // The experts' file too: a dry run sizes every expert, which only a
        // 3D tensor can show (a size of ne[1] rows passed without it).
        runQuant(dir, std.fmt.bufPrint(&nb, "dry moe ftype {d}", .{ft}) catch unreachable, paths[1], &params, null);
        if (quick) continue;
        params.dry_run = false;
        params.nthread = if (@rem(ft, 2) == 0) 1 else 3;
        runQuant(dir, std.fmt.bufPrint(&nb, "no imatrix ftype {d}", .{ft}) catch unreachable, p0, &params, null);
    }

    // Parameters, each on a few file types.
    const Case = struct { name: []const u8, edit: *const fn (*c.llama_model_quantize_params) void, imatrix: bool = true };
    const E = struct {
        fn pure(p: *c.llama_model_quantize_params) void {
            p.pure = true;
        }
        fn outEmb(p: *c.llama_model_quantize_params) void {
            p.output_tensor_type = c.GGML_TYPE_Q8_0;
            p.token_embedding_type = c.GGML_TYPE_Q4_K;
        }
        fn outF16(p: *c.llama_model_quantize_params) void {
            p.output_tensor_type = c.GGML_TYPE_F16;
            p.token_embedding_type = c.GGML_TYPE_IQ2_XXS;
        }
        fn noOut(p: *c.llama_model_quantize_params) void {
            p.quantize_output_tensor = false;
        }
        fn onlyCopy(p: *c.llama_model_quantize_params) void {
            p.only_copy = true;
        }
        fn tt(p: *c.llama_model_quantize_params) void {
            p.tt_overrides = &tt_main;
        }
        fn ttBad(p: *c.llama_model_quantize_params) void {
            p.tt_overrides = &tt_bad;
        }
        fn kv(p: *c.llama_model_quantize_params) void {
            p.kv_overrides = &kv_overrides;
        }
        fn pruneMid(p: *c.llama_model_quantize_params) void {
            p.prune_layers = &prune_mid;
        }
        fn pruneLast(p: *c.llama_model_quantize_params) void {
            p.prune_layers = &prune_last;
        }
        fn pruneFirst(p: *c.llama_model_quantize_params) void {
            p.prune_layers = &prune_first;
        }
        fn noRequant(p: *c.llama_model_quantize_params) void {
            p.allow_requantize = false;
        }
        fn oneThread(p: *c.llama_model_quantize_params) void {
            p.nthread = 1;
        }
        fn allThreads(p: *c.llama_model_quantize_params) void {
            p.nthread = 0;
        }
    };
    const cases = [_]Case{
        .{ .name = "pure", .edit = E.pure },
        .{ .name = "out+emb", .edit = E.outEmb },
        .{ .name = "outF16+embIQ2", .edit = E.outF16 },
        .{ .name = "noOut", .edit = E.noOut },
        .{ .name = "onlyCopy", .edit = E.onlyCopy },
        .{ .name = "tt", .edit = E.tt },
        .{ .name = "ttBad", .edit = E.ttBad },
        .{ .name = "kv", .edit = E.kv },
        .{ .name = "pruneMid", .edit = E.pruneMid },
        .{ .name = "pruneMid-noim", .edit = E.pruneMid, .imatrix = false },
        .{ .name = "pruneLast", .edit = E.pruneLast },
        .{ .name = "pruneFirst", .edit = E.pruneFirst },
        .{ .name = "noRequant", .edit = E.noRequant },
        .{ .name = "oneThread", .edit = E.oneThread },
        .{ .name = "allThreads", .edit = E.allThreads },
    };
    const case_ftypes = [_]c_int{ 15, 10, 23, 2, 38, 1 };
    for (cases) |cs| {
        for ([_]usize{ 0, 3 }) |fi| {
            var im = Imatrix.fromFile(paths[fi], null);
            defer im.deinit();
            for (case_ftypes) |ft| {
                if (quick and ft != 15) continue;
                var params = c.llama_model_quantize_default_params();
                params.ftype = @intCast(ft);
                params.nthread = 4;
                params.allow_requantize = true;
                if (cs.imatrix) params.imatrix = im.entries.items.ptr;
                cs.edit(&params);
                runQuant(dir, std.fmt.bufPrint(&nb, "{s} {s} ftype {d}", .{ specs[fi].name, cs.name, ft }) catch unreachable, paths[fi], &params, null);
            }
        }
    }

    // Importance data that is wrong in each way the C++ checks.
    const im_edits = [_]struct { []const u8, *const fn (*Imatrix) void }{
        .{ "nan", Imatrix.nan },         .{ "inf", Imatrix.inf },
        .{ "short", Imatrix.short },     .{ "short token_embd", Imatrix.shortEmbd },
        .{ "missing", Imatrix.missing }, .{ "duplicate", Imatrix.duplicate },
    };
    for (im_edits) |e| {
        var im = Imatrix.fromFile(p0, e[1]);
        defer im.deinit();
        for ([_]c_int{ 19, 15 }) |ft| {
            var params = c.llama_model_quantize_default_params();
            params.ftype = @intCast(ft);
            params.nthread = 2;
            params.imatrix = im.entries.items.ptr;
            runQuant(dir, std.fmt.bufPrint(&nb, "imatrix {s} ftype {d}", .{ e[0], ft }) catch unreachable, p0, &params, null);
        }
    }

    // Errors before any tensor.
    for ([_]c_int{ 4, 5, 33, 39, 42, 1024, -1 }) |ft| {
        var params = c.llama_model_quantize_default_params();
        params.ftype = @bitCast(ft);
        runQuant(dir, std.fmt.bufPrint(&nb, "invalid ftype {d}", .{ft}) catch unreachable, p0, &params, null);
    }
    {
        var params = c.llama_model_quantize_default_params();
        const missing = gpa.printSentinel("{s}/does-not-exist.gguf", .{dir}, 0) catch unreachable;
        defer gpa.free(missing);
        runQuant(dir, "missing input", missing, &params, null);
        const bad_out = gpa.printSentinel("{s}/no-such-dir/out.gguf", .{dir}, 0) catch unreachable;
        defer gpa.free(bad_out);
        runQuant(dir, "unwritable output", p0, &params, bad_out);
        params.dry_run = true;
        runQuant(dir, "unwritable output, dry run", p0, &params, bad_out);
    }
    {
        // An IQ2 token embedding with no importance data: nothing checks,
        // and ggml_quantize_chunk asserts.
        var params = c.llama_model_quantize_default_params();
        params.ftype = c.LLAMA_FTYPE_MOSTLY_Q4_K_M;
        params.token_embedding_type = c.GGML_TYPE_IQ2_XXS;
        const out = gpa.printSentinel("{s}/death.gguf", .{dir}, 0) catch unreachable;
        defer gpa.free(out);
        const job: QuantJob = .{ .inp = p0, .out = out, .params = &params };
        sameDeath("IQ2 token_embd without imatrix", refQuantJob, portQuantJob, &job);
        removePath(out);
    }
    {
        // An unknown architecture.
        const s: FileSpec = .{ .name = "unknown-arch", .wtype = c.GGML_TYPE_F32, .arch = "no-such-arch", .n_layer = 2 };
        const p = writeSpec(dir, s, 1);
        defer gpa.free(p);
        var params = c.llama_model_quantize_default_params();
        runQuant(dir, "unknown arch", p, &params, null);
    }
    {
        // A block number past INT_MAX, with and without pruning.
        const s: FileSpec = .{ .name = "huge-blk", .wtype = c.GGML_TYPE_F32, .n_layer = 3, .huge_blk = true };
        const p = writeSpec(dir, s, 1);
        defer gpa.free(p);
        var params = c.llama_model_quantize_default_params();
        params.ftype = c.LLAMA_FTYPE_MOSTLY_Q8_0;
        runQuant(dir, "huge blk", p, &params, null);
        params.prune_layers = &prune_first;
        runQuant(dir, "huge blk pruned", p, &params, null);
    }
    {
        // A row length no 32-block type divides: F16 for the K and IQ
        // types, a throw for the rest.
        const s: FileSpec = .{ .name = "odd-shape", .wtype = c.GGML_TYPE_F32, .n_layer = 3, .odd_shape = true };
        const p = writeSpec(dir, s, 1);
        defer gpa.free(p);
        var im = Imatrix.fromFile(p, null);
        defer im.deinit();
        for (valid_ftypes) |ft| {
            var params = c.llama_model_quantize_default_params();
            params.ftype = @intCast(ft);
            params.imatrix = im.entries.items.ptr;
            runQuant(dir, std.fmt.bufPrint(&nb, "odd shape ftype {d}", .{ft}) catch unreachable, p, &params, null);
        }
    }
    {
        // Splits, kept and merged.
        const s: FileSpec = .{ .name = "split", .wtype = c.GGML_TYPE_F16, .tied = false };
        const p = writeSpec(dir, s, 3);
        defer gpa.free(p);
        var im = Imatrix.fromFile(paths[0], null);
        defer im.deinit();
        for ([_]bool{ false, true }) |keep| {
            for ([_]c_int{ 15, 7, 10 }) |ft| {
                var params = c.llama_model_quantize_default_params();
                params.ftype = @intCast(ft);
                params.nthread = 4;
                params.keep_split = keep;
                params.imatrix = im.entries.items.ptr;
                runQuant(dir, std.fmt.bufPrint(&nb, "split keep={} ftype {d}", .{ keep, ft }) catch unreachable, p, &params, null);
            }
        }
        var params = c.llama_model_quantize_default_params();
        params.keep_split = true;
        params.dry_run = true;
        runQuant(dir, "split keep dry", p, &params, null);
    }
    std.debug.print("  files: {d} runs\n", .{file_runs});
}

fn bigRuns(dir: []const u8, models: []const [:0]const u8) void {
    var nb: [256]u8 = undefined;
    for (models, 0..) |m, mi| {
        const Run = struct { ft: c_int, dry: bool };
        const runs: []const Run = if (mi == 0)
            &.{ .{ .ft = 15, .dry = false }, .{ .ft = 7, .dry = false }, .{ .ft = 30, .dry = false }, .{ .ft = 10, .dry = true }, .{ .ft = 24, .dry = true } }
        else
            &.{ .{ .ft = 12, .dry = false }, .{ .ft = 17, .dry = true } };
        for (runs) |r| {
            var params = c.llama_model_quantize_default_params();
            params.ftype = @intCast(r.ft);
            params.allow_requantize = true;
            params.dry_run = r.dry;
            const t0 = c.ggml_time_ms();
            runQuant(dir, std.fmt.bufPrint(&nb, "{s} ftype {d}{s}", .{ m, r.ft, if (r.dry) " dry" else "" }) catch unreachable, m, &params, null);
            std.debug.print("  {s} ftype {d}{s}: {d} s\n", .{ m, r.ft, if (r.dry) " dry" else "", @divTrunc(c.ggml_time_ms() - t0, 1000) });
        }
    }
}

// -----------------------------------------------------------------------------
// Aborts, in forked children

extern "c" fn fork() c_int;
extern "c" fn pipe(fds: *[2]c_int) c_int;
extern "c" fn dup2(old: c_int, new: c_int) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn waitpid(pid: c_int, status: *c_int, options: c_int) c_int;
extern "c" fn _exit(code: c_int) noreturn;

/// What a child that should have died printed: `ggml_abort`'s message
/// after its `file:line: `, which names the C++ source on one side and the
/// port's citation on the other. Null when it did not die of SIGABRT.
fn dies(body: *const fn (ctx: *const anyopaque) void, ctx: *const anyopaque, out: []u8) ?[]const u8 {
    var fds: [2]c_int = undefined;
    if (pipe(&fds) != 0) return null;
    const pid = fork();
    if (pid == 0) {
        _ = dup2(fds[1], 2);
        _ = close(fds[0]);
        body(ctx);
        _exit(0);
    }
    _ = close(fds[1]);
    var len: usize = 0;
    while (len < out.len) {
        const got = read(fds[0], out[len..].ptr, out.len - len);
        if (got <= 0) break;
        len += @intCast(got);
    }
    _ = close(fds[0]);
    var status: c_int = 0;
    _ = waitpid(pid, &status, 0);
    if (status & 0x7f != 6) return null;
    var lines = std.mem.splitScalar(u8, out[0..len], '\n');
    while (lines.next()) |line| {
        if (std.mem.find(u8, line, "terminating due to uncaught exception")) |at| return line[at..];
        const colon = std.mem.findScalar(u8, line, ':') orelse continue;
        const rest = line[colon + 1 ..];
        const colon2 = std.mem.find(u8, rest, ": ") orelse continue;
        if (colon2 == 0) continue;
        for (rest[0..colon2]) |ch| if (!std.ascii.isDigit(ch)) break else return rest[colon2 + 2 ..];
    }
    return "(died, no message)";
}

fn sameDeath(name: []const u8, ref: *const fn (*const anyopaque) void, port: *const fn (*const anyopaque) void, ctx: *const anyopaque) void {
    var a: [8192]u8 = undefined;
    var b: [8192]u8 = undefined;
    const da = dies(ref, ctx, &a);
    const db = dies(port, ctx, &b);
    if (std.c.getenv("QUANT_DIFF_VERBOSE") != null) std.debug.print("    death {s}: ref '{s}' port '{s}'\n", .{ name, da orelse "(lived)", db orelse "(lived)" });
    checks += 1;
    if (da == null) return fail("{s}: the C++ did not abort", .{name});
    if (db == null) return fail("{s}: the port did not abort (C++: {s})", .{ name, da.? });
    if (!std.mem.eql(u8, da.?, db.?)) fail("{s}: death differs:\n    ref  '{s}'\n    port '{s}'", .{ name, da.?, db.? });
}

const QuantJob = struct { inp: [:0]const u8, out: [:0]const u8, params: *const c.llama_model_quantize_params };

fn refQuantJob(p: *const anyopaque) void {
    const j: *const QuantJob = @ptrCast(@alignCast(p));
    _ = c.llama_model_quantize(j.inp.ptr, j.out.ptr, j.params);
}

fn portQuantJob(p: *const anyopaque) void {
    const j: *const QuantJob = @ptrCast(@alignCast(p));
    _ = api.llama_model_quantize(j.inp.ptr, j.out.ptr, j.params);
}

const SaveJob = struct { rm: *c.struct_llama_model, pm: *M.Model, out: [:0]const u8 };

fn refSaveJob(p: *const anyopaque) void {
    const j: *const SaveJob = @ptrCast(@alignCast(p));
    c.llama_model_save_to_file(j.rm, j.out.ptr);
}

fn portSaveJob(p: *const anyopaque) void {
    const j: *const SaveJob = @ptrCast(@alignCast(p));
    api.llama_model_save_to_file(@ptrCast(j.pm), j.out.ptr);
}

// -----------------------------------------------------------------------------
// 4. The saver

fn saverRuns(dir: []const u8, models: []const [:0]const u8, vocab_files: []const [:0]const u8) void {
    for (models) |m| saveBoth(dir, m, -1, false);
    for (models) |m| saveBoth(dir, m, 0, false);
    // Every upstream vocabulary, vocab-only: no tensors and default
    // hyperparameters, but each tokenizer's tokens, scores, types, merges
    // and special ids.
    for (vocab_files) |m| saveBoth(dir, m, 0, true);
}

fn hasCharsmap(path: [:0]const u8) bool {
    const g = c.gguf_init_from_file(path.ptr, .{ .no_alloc = true, .ctx = null }) orelse return false;
    defer c.gguf_free(g);
    return c.gguf_find_key(g, "tokenizer.ggml.precompiled_charsmap") >= 0;
}

fn saveBoth(dir: []const u8, m: [:0]const u8, ngl: i32, vocab_only: bool) void {
    {
        var mp = c.llama_model_default_params();
        mp.n_gpu_layers = ngl;
        mp.vocab_only = vocab_only;
        // CPU_REPACK buffers cannot be read back, and the C++ saver
        // crashes on them; keep CPU weights in plain buffers.
        if (ngl == 0) mp.use_extra_bufts = false;
        const rm = c.llama_model_load_from_file(m.ptr, mp) orelse {
            fail("the C++ cannot load {s}", .{m});
            return;
        };
        defer c.llama_model_free(rm);
        var splits: std.ArrayList([:0]const u8) = .empty;
        defer {
            for (splits.items) |s| gpa.free(s);
            splits.deinit(gpa);
        }
        const pm = M.loadFromFileImpl(.{ .path = m }, &splits, null, null, mp) orelse {
            fail("the port cannot load {s}", .{m});
            return;
        };
        defer M.destroy(pm);

        const ref_out = gpa.printSentinel("{s}/saved-ref.gguf", .{dir}, 0) catch unreachable;
        defer gpa.free(ref_out);
        const port_out = gpa.printSentinel("{s}/saved-port.gguf", .{dir}, 0) catch unreachable;
        defer gpa.free(port_out);
        var ref: std.ArrayList(u8) = .empty;
        defer ref.deinit(gpa);
        var port: std.ArrayList(u8) = .empty;
        defer port.deinit(gpa);
        if (hasCharsmap(m)) {
            // `std::vector<char>` matches no branch of add_kv: both abort.
            var nb0: [256]u8 = undefined;
            const job: SaveJob = .{ .rm = rm, .pm = pm, .out = ref_out };
            sameDeath(std.fmt.bufPrint(&nb0, "save {s} (charsmap)", .{m}) catch unreachable, refSaveJob, portSaveJob, &job);
            removePath(ref_out);
            return;
        }
        const dbg = std.c.getenv("QUANT_DIFF_VERBOSE") != null;
        if (dbg) std.debug.print("    loaded {s} ngl {d}\n", .{ m, ngl });
        rec = &ref;
        c.llama_model_save_to_file(rm, ref_out.ptr);
        if (dbg) std.debug.print("    ref saved\n", .{});
        rec = &port;
        api.llama_model_save_to_file(@ptrCast(pm), port_out.ptr);
        if (dbg) std.debug.print("    port saved\n", .{});
        rec = null;
        var nb: [256]u8 = undefined;
        const name = std.fmt.bufPrint(&nb, "save {s} ngl {d}{s}", .{ m, ngl, if (vocab_only) " vocab-only" else "" }) catch unreachable;
        _ = firstDiff(name, ref.items, port.items);
        compareFiles(name, ref_out, port_out);
        // The written file must load: its metadata is what was saved.
        const f = c.gguf_init_from_file(ref_out.ptr, .{ .no_alloc = true, .ctx = null });
        checks += 1;
        if (f == null) fail("{s}: the saved file does not parse", .{name}) else {
            std.debug.print("  {s}: {d} keys, {d} tensors\n", .{ name, c.gguf_get_n_kv(f), c.gguf_get_n_tensors(f) });
            c.gguf_free(f);
        }
        removePath(ref_out);
        removePath(port_out);
    }
}

// -----------------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: quant_diff <tmpdir> [--quick] [--skip-big] [--only section] [model.gguf...]\n", .{});
        std.process.exit(2);
    }
    const dir = args[1];
    var quick = false;
    var skip_big = false;
    var only: ?[]const u8 = null;
    var models: std.ArrayList([:0]const u8) = .empty;
    var vocab_files: std.ArrayList([:0]const u8) = .empty;
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--quick")) {
            quick = true;
        } else if (std.mem.eql(u8, args[i], "--skip-big")) {
            skip_big = true;
        } else if (std.mem.eql(u8, args[i], "--only") and i + 1 < args.len) {
            only = args[i + 1];
            i += 1;
        } else if (std.mem.find(u8, args[i], "ggml-vocab-") != null) {
            try vocab_files.append(gpa, try gpa.dupeSentinel(u8, args[i], 0));
        } else {
            try models.append(gpa, try gpa.dupeSentinel(u8, args[i], 0));
        }
    }
    const want = struct {
        fn f(o: ?[]const u8, s: []const u8) bool {
            return o == null or std.mem.eql(u8, o.?, s);
        }
    }.f;

    c.llama_log_set(logCb, null);

    harness_root.api.impl.llama_log_set(logCb, null);
    c.ggml_log_set(logCb, null);
    c.llama_backend_init();

    // The default parameters, byte for byte.
    {
        const a = c.llama_model_quantize_default_params();
        const b = api.llama_model_quantize_default_params();
        // Field by field: the padding after `dry_run` is unspecified.
        inline for (@typeInfo(c.llama_model_quantize_params).@"struct".field_names) |f| {
            checks += 1;
            if (!std.mem.eql(u8, std.mem.asBytes(&@field(a, f)), std.mem.asBytes(&@field(b, f)))) fail("llama_model_quantize_default_params.{s} differs", .{f});
        }
        var ft: c_int = -2;
        while (ft < 1100) : (ft += 1) {
            if (ft == 64) ft = 1020;
            checks += 1;
            const r = qref_default_type(ft);
            const p: c_int = @intCast(quant.ftypeGetDefaultType(@bitCast(ft)));
            if (r != p) fail("llama_ftype_get_default_type({d}): ref {d} port {d}", .{ ft, r, p });
        }
    }

    if (want(only, "layout")) checkLayerWalk();
    if (want(only, "types")) typeRules(quick);
    if (want(only, "files")) fileRuns(dir, quick);
    if (want(only, "big") and !skip_big and models.items.len != 0) bigRuns(dir, if (quick) models.items[0..1] else models.items);
    if (want(only, "saver")) saverRuns(dir, if (quick and models.items.len != 0) models.items[0..1] else models.items, vocab_files.items);

    std.debug.print("quant-diff: {d} checks, {d} failures\n", .{ checks, failures });
    if (failures != 0) std.process.exit(1);
}
