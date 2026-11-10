//! Derives a `prism.hadamard` model from a plain one, for `scripts/context-diff`
//! and `scripts/adapter-diff`.
//!
//! **Not a port.** The weights are not really folded, so a derived model
//! computes nonsense; both sides compute the same nonsense through the
//! transforms the graph must apply, which is what the gates compare.

const std = @import("std");
const harness_root = @import("harness_root");

const c = harness_root.impl.c;
const gpa = std.heap.c_allocator;

/// The weight kinds `load_hparams_hadamard` accepts as folded
/// (llama-model.cpp:2145 @cd959bae9).
const foldable_kinds = [_][]const u8{
    "attn_q",           "attn_k",         "attn_v",       "attn_qkv",       "attn_gate",   "attn_output",
    "ffn_gate",         "ffn_up",         "ffn_down",     "ffn_gate_exps",  "ffn_up_exps", "ffn_down_exps",
    "ffn_gate_up_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ssm_out",
};

fn foldable(name: []const u8) bool {
    if (std.mem.eql(u8, name, "output.weight")) return true;
    if (!std.mem.startsWith(u8, name, "blk.")) return false;
    var pos: usize = 4;
    while (pos < name.len and std.ascii.isDigit(name[pos])) pos += 1;
    if (pos == 4 or pos >= name.len or name[pos] != '.') return false;
    const rest = name[pos + 1 ..];
    for (foldable_kinds) |k| {
        if (rest.len == k.len + 7 and std.mem.startsWith(u8, rest, k) and std.mem.endsWith(u8, rest, ".weight")) return true;
    }
    return false;
}

/// `--hadamard`: `src` with `prism.hadamard` metadata added, written to
/// `dst` -- every foldable weight whose input width the block of 8 divides
/// marked folded, explicit signs per width, the token embedding as the
/// latent table, and for a gated delta net the grouped value heads. The
/// weights are not really folded, so the model computes nonsense; both
/// sides compute the same nonsense, through the transforms the graph must
/// apply, which is what is compared.
///
/// With `drop_output`, `output.weight` is left out, so the head is the
/// token embedding: version 2's tied latent output.
pub fn deriveHadamard(src: [:0]const u8, dst: [:0]const u8, drop_output: bool) void {
    var data_ctx: ?*c.ggml_context = null;
    const g = c.gguf_init_from_file(src.ptr, .{ .no_alloc = false, .ctx = &data_ctx }) orelse @panic("cannot read the model for --hadamard");
    defer c.gguf_free(g);
    defer c.ggml_free(data_ctx);
    const out = c.gguf_init_empty().?;
    defer c.gguf_free(out);
    c.gguf_set_kv(out, g);

    const block: u32 = 8;
    var names: std.ArrayList([*:0]const u8) = .empty;
    defer names.deinit(gpa);
    var widths: std.ArrayList(i32) = .empty;
    defer widths.deinit(gpa);
    var has_output = false;
    var inverse = false;
    // The file's tensors in its order, not the context's: that also holds
    // the data blob.
    var ti: i64 = 0;
    while (ti < c.gguf_get_n_tensors(g)) : (ti += 1) {
        const t = c.ggml_get_tensor(data_ctx, c.gguf_get_tensor_name(g, ti)).?;
        const name = std.mem.sliceTo(&t.*.name, 0);
        if (drop_output and std.mem.eql(u8, name, "output.weight")) continue;
        c.gguf_add_tensor(out, t);
        const w: i32 = @intCast(t.*.ne[0]);
        if (std.mem.eql(u8, name, "output.weight")) has_output = true;
        const as_inverse = std.mem.eql(u8, name, "token_embd.weight");
        if (!(foldable(name) or as_inverse) or @rem(w, @as(i32, block)) != 0) continue;
        if (as_inverse) inverse = true else names.append(gpa, @ptrCast(&t.*.name)) catch unreachable;
        if (std.mem.indexOfScalar(i32, widths.items, w) == null) widths.append(gpa, w) catch unreachable;
    }
    // A tied output (no output.weight) needs version 2 and tied_output.
    const tied = inverse and !has_output;
    c.gguf_set_val_u32(out, "prism.hadamard.version", if (tied) 2 else 1);
    c.gguf_set_val_bool(out, "prism.hadamard.tied_output", tied);
    c.gguf_set_val_u32(out, "prism.hadamard.block_size", block);
    c.gguf_set_val_str(out, "prism.hadamard.transform", "normalized-sylvester-walsh-hadamard");
    c.gguf_set_val_str(out, "prism.hadamard.axis", "input-last-dimension");
    c.gguf_set_val_str(out, "prism.hadamard.sign_mode", "explicit");
    c.gguf_set_arr_data(out, "prism.hadamard.sign_widths", c.GGUF_TYPE_INT32, widths.items.ptr, widths.items.len);
    var values: std.ArrayList(i32) = .empty;
    defer values.deinit(gpa);
    for (widths.items) |w| {
        for (0..@intCast(w)) |i| values.append(gpa, if ((i * 7 + @as(usize, @intCast(w))) % 3 == 0) -1 else 1) catch unreachable;
    }
    c.gguf_set_arr_data(out, "prism.hadamard.sign_values", c.GGUF_TYPE_INT32, values.items.ptr, values.items.len);
    c.gguf_set_arr_str(out, "prism.hadamard.weight_names", @ptrCast(names.items.ptr), names.items.len);
    if (inverse) {
        const inv = [_][*:0]const u8{"token_embd.weight"};
        c.gguf_set_arr_str(out, "prism.hadamard.inverse_weight_names", @ptrCast(@constCast(&inv)), inv.len);
    }
    // The gated delta net's value heads in grouped order, where the
    // geometry allows it.
    const arch_k = c.gguf_find_key(g, "general.architecture");
    const arch = if (arch_k >= 0) std.mem.span(c.gguf_get_val_str(g, arch_k)) else "";
    var kb: [96]u8 = undefined;
    const dt_k = c.gguf_find_key(g, (std.fmt.bufPrintSentinel(&kb, "{s}.ssm.time_step_rank", .{arch}, 0) catch unreachable).ptr);
    const ng_k = c.gguf_find_key(g, (std.fmt.bufPrintSentinel(&kb, "{s}.ssm.group_count", .{arch}, 0) catch unreachable).ptr);
    if (dt_k >= 0 and ng_k >= 0) {
        const n_v = c.gguf_get_val_u32(g, dt_k);
        const n_k = c.gguf_get_val_u32(g, ng_k);
        if (n_k > 0 and n_v > 0 and n_v % n_k == 0) c.gguf_set_val_bool(out, "prism.hadamard.gdn_v_grouped", true);
    }
    if (!c.gguf_write_to_file(out, dst.ptr, false)) @panic("cannot write the --hadamard model");
    std.debug.print("  --hadamard: {d} folded weights, {d} widths, inverse {}, tied {}, written to {s}\n", .{ names.items.len, widths.items.len, inverse, tied, dst });
}
