//! Model quantization: read a GGUF, choose each tensor's type, convert it,
//! and write the result.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-quant.cpp` -- the whole file: the tensor
//!   categories, `tensor_allows_quantization`, the type-selection rules,
//!   the fallback for shapes a type cannot hold, the threaded conversion,
//!   the imatrix and prune handling, `llama_model_quantize_impl`, and the
//!   helpers `llama-ext.h` declares for the type-selection test
//! - `llama.cpp/src/llama-ext.h` -- `llama_quant_model_desc`
//! - `llama.cpp/src/llama-impl.cpp` -- `llama_format_tensor_shape`
//!
//! All at v0.3.0 (`c1d0e7a00`). The C entry points are `api/quant.zig`'s.
//!
//! # The contract
//!
//! `llama-quant.o` exports two C functions, `llama_model_quantize` and
//! `llama_model_quantize_default_params`, and six C++-linkage helpers that
//! only `tests/test-quant-type-selection.cpp` calls. Those six are ported
//! too, as plain Zig functions, because they are what lets the
//! type-selection rules be diffed on every architecture: a file can only be
//! quantized end to end where the model's `load_arch_hparams` is ported,
//! and the helpers build their model from a descriptor instead.
//!
//! # What the C++ leaves to its platform
//!
//! - **The load mode is `LLAMA_LOAD_MODE_NONE`** on this target, the
//!   `#else` arm of `llama_model_quantize_impl`: tensors are read into a
//!   buffer, never mapped.
//! - **Data is loaded by the tensor's current name**, through a lookup
//!   keyed by the names the loader read. After `prune_layers` renames a
//!   block, that lookup finds the *original* tensor of the new name, as
//!   the C++'s `weights_map` does -- so a pruned model takes the data of
//!   the block whose number it inherited. Reproduced, not fixed.
//! - **Threads only split the work.** Every row is converted on its own,
//!   so the bytes do not depend on how many threads there are.

const std = @import("std");
const impl = @import("impl.zig");
const arch = @import("arch.zig");
const libcxx = @import("libcxx.zig");
const loader = @import("loader.zig");
const mmap = @import("mmap.zig");
const model_mod = @import("model.zig");
const regex = @import("regex.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Tensor = c.ggml_tensor;
const GgmlType = c.enum_ggml_type;
const Ftype = c.enum_llama_ftype;

fn oom() noreturn {
    impl.abort("llama-quant.cpp", 0, "out of memory");
}

fn nameOf(t: *const Tensor) [:0]const u8 {
    return std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&t.name)), 0);
}

fn typeName(ty: GgmlType) [*:0]const u8 {
    return c.ggml_type_name(ty);
}

fn contains(haystack: []const u8, needle: []const u8) bool {
    return std.mem.find(u8, haystack, needle) != null;
}

/// A key's name for the gguf API, NUL-terminated.
const KeyZ = struct {
    buf: [arch.max_name + 1]u8 = undefined,

    fn init(a: arch.Arch, kv: arch.Kv) KeyZ {
        var self: KeyZ = .{};
        const n = arch.kvName(a, kv, null);
        @memcpy(self.buf[0..n.len], n.slice());
        self.buf[n.len] = 0;
        return self;
    }

    fn ptr(self: *const KeyZ) [*:0]const u8 {
        return @ptrCast(&self.buf);
    }
};

// -----------------------------------------------------------------------------
// Tensor categories

/// Mirrors `tensor_category` (llama-quant.cpp:26 @c1d0e7a00).
pub const Category = enum {
    token_embd,
    attention_q,
    attention_v,
    attention_k,
    attention_qkv,
    attention_kv_b,
    attention_output,
    ffn_up,
    ffn_gate,
    ffn_down,
    output,
    other,
};

/// Ports `tensor_name_match_token_embd` (llama-quant.cpp:101 @c1d0e7a00).
fn nameMatchTokenEmbd(name: []const u8) bool {
    return std.mem.eql(u8, name, "token_embd.weight") or std.mem.eql(u8, name, "per_layer_token_embd.weight");
}

/// Ports `tensor_name_match_output_weight` (llama-quant.cpp:106 @c1d0e7a00).
fn nameMatchOutputWeight(name: []const u8) bool {
    return std.mem.eql(u8, name, "output.weight");
}

/// Ports `tensor_get_category` (llama-quant.cpp:116 @c1d0e7a00).
///
/// Parameters:
/// - `name`: the tensor's name.
///
/// Return: its broad category.
pub fn getCategory(name: []const u8) Category {
    if (nameMatchOutputWeight(name)) return .output;
    if (nameMatchTokenEmbd(name)) return .token_embd;
    if (contains(name, "attn_qkv.weight")) return .attention_qkv;
    if (contains(name, "attn_kv_b.weight")) return .attention_kv_b;
    if (contains(name, "attn_v.weight")) return .attention_v;
    if (contains(name, "attn_k.weight")) return .attention_k;
    if (contains(name, "attn_q.weight")) return .attention_q;
    if (contains(name, "attn_output.weight")) return .attention_output;
    if (contains(name, "ffn_up")) return .ffn_up;
    if (contains(name, "ffn_gate")) return .ffn_gate;
    if (contains(name, "ffn_down")) return .ffn_down;
    return .other;
}

/// Ports `category_is_attn_v` (llama-quant.cpp:154 @c1d0e7a00).
fn categoryIsAttnV(cat: Category) bool {
    return cat == .attention_v or cat == .attention_qkv or cat == .attention_kv_b;
}

// -----------------------------------------------------------------------------
// Quantization state

/// One compiled `tt_overrides` entry.
const Pattern = struct {
    re: regex.Regex,
    type: GgmlType,
};

/// Ports `quantize_state_impl` (llama-quant.cpp:164 @c1d0e7a00).
pub const State = struct {
    model: *const Model,
    params: *const c.llama_model_quantize_params,

    n_attention_wv: c_int = 0,
    n_ffn_down: c_int = 0,
    n_ffn_gate: c_int = 0,
    n_ffn_up: c_int = 0,
    i_attention_wv: c_int = 0,
    i_ffn_down: c_int = 0,
    i_ffn_gate: c_int = 0,
    i_ffn_up: c_int = 0,

    n_fallback: c_int = 0,

    has_imatrix: bool = false,

    /// Tied until `output.weight` is seen.
    has_tied_embeddings: bool = true,

    /// The tensor-type overrides, compiled once.
    tensor_type_patterns: std.ArrayList(Pattern) = .empty,

    /// Ports the constructor of `quantize_state_impl` (llama-quant.cpp:187
    /// @c1d0e7a00): every override pattern is compiled, in order, and the
    /// first that fails throws its `regex_error`.
    ///
    /// Parameters:
    /// - `model`: the model; borrowed for the state's life.
    /// - `params`: the parameters; borrowed for the state's life.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the state; free it with `deinit`.
    pub fn init(model: *const Model, params: *const c.llama_model_quantize_params, exc: *Exception) Thrown!State {
        var self: State = .{ .model = model, .params = params };
        errdefer self.deinit();
        if (params.tt_overrides != null) {
            var p = params.tt_overrides;
            while (p[0].pattern != null) : (p += 1) {
                const re = regex.Regex.compile(std.mem.span(p[0].pattern)) catch |e|
                    return exc.throw(.runtime_error, "{s}", .{regex.errorText(e)});
                self.tensor_type_patterns.append(allocator, .{ .re = re, .type = p[0].type }) catch oom();
            }
        }
        return self;
    }

    /// Return: nothing.
    pub fn deinit(self: *State) void {
        for (self.tensor_type_patterns.items) |*p| p.re.deinit();
        self.tensor_type_patterns.deinit(allocator);
    }
};

/// Mirrors `tensor_metadata` (llama-quant.cpp:200 @c1d0e7a00).
const Metadata = struct {
    name: [:0]u8 = undefined,
    target_type: GgmlType = 0,
    category: Category = .other,
    remapped_imatrix_name: []u8 = &[_]u8{},
    allows_quantization: bool = false,
    requires_imatrix: bool = false,

    fn deinit(self: *Metadata) void {
        allocator.free(self.name);
        allocator.free(self.remapped_imatrix_name);
    }
};

// -----------------------------------------------------------------------------
// Layer remapping for prune_layers

const blk_pattern = "blk\\.(\\d+)\\.";

/// `std::stoi` on a run of digits: `stoi: out of range` past `INT_MAX`.
fn stoi(digits: []const u8, exc: *Exception) Thrown!c_int {
    var v: i64 = 0;
    for (digits) |d| {
        v = v * 10 + (d - '0');
        if (v > std.math.maxInt(c_int)) return exc.throw(.out_of_range, "stoi: out of range", .{});
    }
    return @intCast(v);
}

/// The `std::map<int, std::string>` `remap_layer` fills: sorted by key.
const LayerMap = struct {
    keys: std.ArrayList(c_int) = .empty,
    vals: std.ArrayList([]u8) = .empty,

    fn deinit(self: *LayerMap) void {
        for (self.vals.items) |v| allocator.free(v);
        self.keys.deinit(allocator);
        self.vals.deinit(allocator);
    }

    fn get(self: *const LayerMap, k: c_int) ?[]const u8 {
        for (self.keys.items, 0..) |key, i| if (key == k) return self.vals.items[i];
        return null;
    }

    fn put(self: *LayerMap, k: c_int, v: []u8) void {
        var at: usize = 0;
        while (at < self.keys.items.len and self.keys.items[at] < k) at += 1;
        self.keys.insert(allocator, at, k) catch oom();
        self.vals.insert(allocator, at, v) catch oom();
    }

    fn empty(self: *const LayerMap) bool {
        return self.keys.items.len == 0;
    }
};

/// Ports `remap_layer` (llama-quant.cpp:48 @c1d0e7a00).
///
/// Return: the new name, owned by the caller: empty when the block is
/// pruned, `orig_name` when nothing moves.
fn remapLayer(re: *const regex.Regex, orig_name: []const u8, prune: []const c_int, mapped: *LayerMap, next_id: *c_int, exc: *Exception) Thrown![]u8 {
    if (prune.len == 0) return allocator.dupe(u8, orig_name) catch oom();

    var m = regex.Match.init();
    defer m.deinit();
    if (re.trySearch(orig_name, &m) catch |e| return exc.throw(.runtime_error, "{s}", .{regex.errorText(e)})) {
        const pos = m.position(1);
        const len = m.length(1);
        const blk = try stoi(orig_name[pos..][0..len], exc);

        if (mapped.get(blk) != null) {
            // Already mapped, do nothing.
        } else if (std.mem.findScalar(c_int, prune, blk) != null) {
            mapped.put(blk, allocator.dupe(u8, "") catch oom());
        } else if (blk < prune[0]) {
            mapped.put(blk, std.fmt.allocPrint(allocator, "{d}", .{blk}) catch oom());
            next_id.* = blk + 1;
        } else {
            mapped.put(blk, std.fmt.allocPrint(allocator, "{d}", .{next_id.*}) catch oom());
            next_id.* += 1;
        }

        const to = mapped.get(blk).?;
        if (to.len == 0) return allocator.dupe(u8, "") catch oom();
        return std.mem.concat(allocator, u8, &.{ orig_name[0..pos], to, orig_name[pos + len ..] }) catch oom();
    }

    return allocator.dupe(u8, orig_name) catch oom();
}

/// Ports `remap_imatrix` (llama-quant.cpp:76 @c1d0e7a00).
///
/// Return: the name the imatrix knows the tensor by, owned by the caller.
fn remapImatrix(re: *const regex.Regex, orig_name: [:0]const u8, mapped: *const LayerMap, exc: *Exception) Thrown![]u8 {
    if (mapped.empty()) return allocator.dupe(u8, orig_name) catch oom();

    var m = regex.Match.init();
    defer m.deinit();
    if (re.trySearch(orig_name, &m) catch |e| return exc.throw(.runtime_error, "{s}", .{regex.errorText(e)})) {
        const pos = m.position(1);
        const len = m.length(1);
        const blk = orig_name[pos..][0..len];

        for (mapped.keys.items, mapped.vals.items) |k, v| {
            if (std.mem.eql(u8, v, blk)) {
                return std.fmt.allocPrint(allocator, "{s}{d}{s}", .{ orig_name[0..pos], k, orig_name[pos + len ..] }) catch oom();
            }
        }
        c.ggml_abort("llama-quant.cpp", 91, "\n%s: imatrix mapping error for %s\n", "remap_imatrix", orig_name.ptr);
    }

    return allocator.dupe(u8, orig_name) catch oom();
}

// -----------------------------------------------------------------------------
// Dequantization

/// Ports `llama_tensor_dequantize_impl` (llama-quant.cpp:213 @c1d0e7a00):
/// `tensor`'s data as f32 in `output`, split across `nthread` threads.
fn dequantize(tensor: *Tensor, output: *std.ArrayList(f32), nelements: usize, nthread: c_int, exc: *Exception) Thrown!void {
    if (output.items.len < nelements) output.resize(allocator, nelements) catch oom();
    const f32_output = output.items.ptr;

    const qtype = c.ggml_get_type_traits(tensor.type);
    if (c.ggml_is_quantized(tensor.type)) {
        if (qtype.*.to_float == null) {
            return exc.throw(.runtime_error, "type {s} unsupported for integer quantization: no dequantization available", .{typeName(tensor.type)});
        }
    } else if (tensor.type != c.GGML_TYPE_F16 and tensor.type != c.GGML_TYPE_BF16) {
        return exc.throw(.runtime_error, "cannot dequantize/convert tensor type {s}", .{typeName(tensor.type)});
    }

    if (nthread < 2) {
        convertRun(tensor.type, qtype.*.to_float, @ptrCast(tensor.data), f32_output, @intCast(nelements));
        return;
    }

    const block_size: usize = if (tensor.type == c.GGML_TYPE_F16 or tensor.type == c.GGML_TYPE_BF16)
        1
    else
        @intCast(c.ggml_blck_size(tensor.type));
    const block_size_bytes = c.ggml_type_size(tensor.type);

    impl.assert(nelements % block_size == 0, "llama-quant.cpp", 255, "nelements % block_size == 0");
    const nt: usize = @intCast(nthread);
    const nblocks = nelements / block_size;
    const blocks_per_thread = nblocks / nt;
    const spare_blocks = nblocks - blocks_per_thread * nt;

    var in_buff_offs: usize = 0;
    var out_buff_offs: usize = 0;

    const workers = allocator.alloc(std.Thread, nt) catch oom();
    defer allocator.free(workers);
    const base: [*]u8 = @ptrCast(tensor.data);
    for (0..nt) |tnum| {
        const thr_blocks = blocks_per_thread + (if (tnum == nt - 1) spare_blocks else 0);
        const thr_elems = thr_blocks * block_size;
        const thr_block_bytes = thr_blocks * block_size_bytes;
        workers[tnum] = std.Thread.spawn(.{}, convertRun, .{
            tensor.type, qtype.*.to_float, base + in_buff_offs, f32_output + out_buff_offs, @as(i64, @intCast(thr_elems)),
        }) catch oom();
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (workers) |w| w.join();
}

/// The `compute` lambda of `llama_tensor_dequantize_impl`
/// (llama-quant.cpp:213 @c1d0e7a00).
fn convertRun(ty: GgmlType, to_float: c.ggml_to_float_t, inbuf: [*]u8, outbuf: [*]f32, nels: i64) void {
    if (ty == c.GGML_TYPE_F16) {
        c.ggml_fp16_to_fp32_row(@ptrCast(@alignCast(inbuf)), outbuf, nels);
    } else if (ty == c.GGML_TYPE_BF16) {
        c.ggml_bf16_to_fp32_row(@ptrCast(@alignCast(inbuf)), outbuf, nels);
    } else {
        to_float.?(inbuf, outbuf, nels);
    }
}

// -----------------------------------------------------------------------------
// Which tensors are quantized

/// Ports `tensor_allows_quantization` (llama-quant.cpp:289 @c1d0e7a00).
///
/// Parameters:
/// - `params`: the quantization parameters.
/// - `a`: the model's architecture.
/// - `tensor`: the tensor.
///
/// Return: whether the tensor may be quantized.
pub fn tensorAllowsQuantization(params: *const c.llama_model_quantize_params, a: arch.Arch, tensor: *const Tensor) bool {
    _ = a; // LLM_TN's names do not depend on the architecture.
    if (params.only_copy) return false;

    // Only 2D and 3D tensors (experts).
    if (c.ggml_n_dims(tensor) < 2) return false;

    const name = nameOf(tensor);

    // `name.rfind("weight") == name.size() - 6`, with `npos` and the
    // unsigned subtraction both as the C++ has them: a name shorter than
    // six bytes compares equal.
    const at = std.mem.findLast(u8, name, "weight") orelse std.math.maxInt(usize);
    var allowed = at == name.len -% 6;

    allowed = allowed and !contains(name, "_norm.weight");
    allowed = allowed and (params.quantize_output_tensor or !std.mem.eql(u8, name, "output.weight"));
    allowed = allowed and !contains(name, "ffn_gate_inp.weight");
    allowed = allowed and !contains(name, "ffn_gate_tid2eid.weight");
    allowed = allowed and !contains(name, "altup");
    allowed = allowed and !contains(name, "laurel");
    allowed = allowed and !contains(name, "per_layer_model_proj");

    // Positional embeddings and token types (BERT).
    const pos_embd = arch.tensorName(.pos_embd, "weight", -1, -1);
    const token_types = arch.tensorName(.token_types, "weight", -1, -1);
    allowed = allowed and !std.mem.eql(u8, name, pos_embd.slice());
    allowed = allowed and !std.mem.eql(u8, name, token_types.slice());

    allowed = allowed and !contains(name, "ssm_conv1d");
    allowed = allowed and !contains(name, "shortconv.conv.weight");
    allowed = allowed and !contains(name, "indexer.k_proj.weight");
    allowed = allowed and !contains(name, "indexer.q_proj.weight");

    inline for (.{
        "time_mix_first.weight",    "time_mix_w0.weight",       "time_mix_w1.weight",         "time_mix_w2.weight",
        "time_mix_v0.weight",       "time_mix_v1.weight",       "time_mix_v2.weight",         "time_mix_a0.weight",
        "time_mix_a1.weight",       "time_mix_a2.weight",       "time_mix_g1.weight",         "time_mix_g2.weight",
        "time_mix_decay_w1.weight", "time_mix_decay_w2.weight", "time_mix_lerp_fused.weight",
    }) |s| allowed = allowed and !contains(name, s);

    allowed = allowed and !contains(name, "attn_rel_b.weight");

    inline for (.{ ".position_embd", "sam.pos_embd", "sam.neck.", "sam.net_", ".rel_pos", ".patch_embd", ".patch_merger" }) |s|
        allowed = allowed and !contains(name, s);

    allowed = allowed and !contains(name, "a.rvq.codebook");
    allowed = allowed and !contains(name, "mm.a.code_embd");

    return allowed;
}

// -----------------------------------------------------------------------------
// Type selection

/// Ports `tensor_type_fallback` (llama-quant.cpp:374 @c1d0e7a00): a type
/// whose block the tensor's rows can hold.
fn typeFallback(qs: *State, t: *const Tensor, target_type: GgmlType, exc: *Exception) Thrown!GgmlType {
    var return_type = target_type;

    const ncols = t.ne[0];
    const qk_k = c.ggml_blck_size(target_type);

    if (@rem(ncols, qk_k) != 0) {
        impl.logWarn("warning: %-36s - ncols %6lld not divisible by %3lld (required for type %7s) ", .{
            @as([*:0]const u8, @ptrCast(&t.name)), @as(c_longlong, ncols), @as(c_longlong, qk_k), typeName(target_type),
        });
        qs.n_fallback += 1;

        return_type = switch (target_type) {
            // Block size 256 on the left, 32 on the right.
            c.GGML_TYPE_IQ1_S,
            c.GGML_TYPE_IQ1_M,
            c.GGML_TYPE_IQ2_XXS,
            c.GGML_TYPE_IQ2_XS,
            c.GGML_TYPE_IQ2_S,
            c.GGML_TYPE_IQ3_XXS,
            c.GGML_TYPE_IQ3_S,
            c.GGML_TYPE_IQ4_XS,
            => c.GGML_TYPE_IQ4_NL,
            c.GGML_TYPE_Q2_0,
            c.GGML_TYPE_Q2_K,
            c.GGML_TYPE_Q3_K,
            c.GGML_TYPE_TQ1_0,
            c.GGML_TYPE_TQ2_0,
            => c.GGML_TYPE_Q4_0,
            c.GGML_TYPE_Q4_K => c.GGML_TYPE_Q5_0,
            c.GGML_TYPE_Q5_K => c.GGML_TYPE_Q5_1,
            c.GGML_TYPE_Q6_K => c.GGML_TYPE_Q8_0,
            else => return exc.throw(.runtime_error, "no tensor type fallback is defined for type {s}", .{typeName(target_type)}),
        };
        if (@rem(ncols, c.ggml_blck_size(return_type)) != 0) {
            // Most likely the first dimension is not divisible by 32.
            impl.logWarn("(WARNING: must use F16 due to unusual shape) ", .{});
            return_type = c.GGML_TYPE_F16;
        }
        impl.logWarn("-> falling back to %7s\n", .{typeName(return_type)});
    }
    return return_type;
}

/// The `use_more_bits` lambda of `llama_tensor_get_type_impl`
/// (llama-quant.cpp:424 @c1d0e7a00), with C's truncating division.
fn useMoreBits(i_layer: c_int, n_layers: c_int) bool {
    return i_layer < @divTrunc(n_layers, 8) or i_layer >= @divTrunc(7 * n_layers, 8) or
        @rem(i_layer - @divTrunc(n_layers, 8), 3) == 2;
}

/// The `layer_info` lambda of `llama_tensor_get_type_impl`
/// (llama-quant.cpp:424 @c1d0e7a00): with experts, the layer is read from
/// the name.
fn layerInfo(n_expert: c_int, i_layer_in: c_int, n_layer: c_int, name: [:0]const u8, exc: *Exception) Thrown![2]c_int {
    var i_layer = i_layer_in;
    if (n_expert > 1) {
        if (c.sscanf(name.ptr, "blk.%d.", &i_layer) != 1) {
            return exc.throw(.runtime_error, "Failed to determine layer for tensor {s}", .{name});
        }
        if (i_layer < 0 or i_layer >= n_layer) {
            return exc.throw(.runtime_error, "Bad layer {d} for tensor {s}. Must be in [0, {d})", .{ i_layer, name, n_layer });
        }
    }
    return .{ i_layer, n_layer };
}

/// Ports `llama_tensor_get_type_impl` (llama-quant.cpp:424 @c1d0e7a00): the
/// standard mixture for a file type, by category, layer and architecture.
fn getTypeImpl(qs: *State, new_type_in: GgmlType, tensor: *const Tensor, ftype: Ftype, category: Category, exc: *Exception) Thrown!GgmlType {
    var new_type = new_type_in;
    const name = nameOf(tensor);
    const a = qs.model.arch;
    const hp = &qs.model.hparams;

    const n_expert: c_int = @max(1, @as(c_int, @bitCast(hp.n_expert)));

    const F = struct {
        fn in(f: Ftype, comptime list: anytype) bool {
            inline for (list) |x| if (f == x) return true;
            return false;
        }
    };

    if (category == .output or (qs.has_tied_embeddings and category == .token_embd)) {
        // Tied embeddings take the output tensor's quantization.
        if (qs.params.output_tensor_type < c.GGML_TYPE_COUNT) {
            new_type = qs.params.output_tensor_type;
        } else {
            const nx = tensor.ne[0];
            const qk_k = c.ggml_blck_size(new_type);

            if (ftype == c.LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
                new_type = c.GGML_TYPE_Q8_0;
            } else if (a == .falcon or @rem(nx, qk_k) != 0) {
                new_type = c.GGML_TYPE_Q8_0;
            } else if (F.in(ftype, .{
                c.LLAMA_FTYPE_MOSTLY_IQ2_XXS, c.LLAMA_FTYPE_MOSTLY_IQ2_XS, c.LLAMA_FTYPE_MOSTLY_IQ3_XXS,
                c.LLAMA_FTYPE_MOSTLY_IQ1_S,   c.LLAMA_FTYPE_MOSTLY_IQ2_S,  c.LLAMA_FTYPE_MOSTLY_IQ2_M,
                c.LLAMA_FTYPE_MOSTLY_IQ1_M,
            })) {
                new_type = c.GGML_TYPE_Q5_K;
            } else if (new_type != c.GGML_TYPE_Q8_0) {
                new_type = c.GGML_TYPE_Q6_K;
            }
        }
    } else if (ftype == c.LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
        // MoE tensors to MXFP4, the rest to Q8_0; MLA projections are 3D
        // too, so BailingMoE3 names its expert roles.
        const is_bailingmoe3_expert = a == .bailingmoe3 and
            (category == .ffn_up or category == .ffn_gate or category == .ffn_down);
        if (tensor.ne[2] > 1 and (a != .bailingmoe3 or is_bailingmoe3_expert)) {
            new_type = c.GGML_TYPE_MXFP4;
        } else {
            new_type = c.GGML_TYPE_Q8_0;
        }
    } else if (category == .token_embd) {
        if (qs.params.token_embedding_type < c.GGML_TYPE_COUNT) {
            new_type = qs.params.token_embedding_type;
        } else {
            if (F.in(ftype, .{ c.LLAMA_FTYPE_MOSTLY_IQ2_XXS, c.LLAMA_FTYPE_MOSTLY_IQ2_XS, c.LLAMA_FTYPE_MOSTLY_IQ1_S, c.LLAMA_FTYPE_MOSTLY_IQ1_M })) {
                new_type = c.GGML_TYPE_Q2_K;
            } else if (ftype == c.LLAMA_FTYPE_MOSTLY_IQ2_S or ftype == c.LLAMA_FTYPE_MOSTLY_IQ2_M) {
                new_type = c.GGML_TYPE_IQ3_S;
            } else if (ftype == c.LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
                new_type = c.GGML_TYPE_IQ3_S;
            } else if (F.in(ftype, .{ c.LLAMA_FTYPE_MOSTLY_TQ1_0, c.LLAMA_FTYPE_MOSTLY_TQ2_0, c.LLAMA_FTYPE_MOSTLY_Q2_0 })) {
                new_type = c.GGML_TYPE_Q4_K;
            }
        }
    } else if (F.in(ftype, .{
        c.LLAMA_FTYPE_MOSTLY_IQ2_XXS, c.LLAMA_FTYPE_MOSTLY_IQ2_XS, c.LLAMA_FTYPE_MOSTLY_IQ1_S,
        c.LLAMA_FTYPE_MOSTLY_IQ2_S,   c.LLAMA_FTYPE_MOSTLY_IQ2_M,  c.LLAMA_FTYPE_MOSTLY_IQ1_M,
    })) {
        const is_s_or_m = ftype == c.LLAMA_FTYPE_MOSTLY_IQ2_S or ftype == c.LLAMA_FTYPE_MOSTLY_IQ2_M;
        if (categoryIsAttnV(category)) {
            if (hp.nGqa(0) >= 4 or hp.n_expert >= 4) {
                new_type = c.GGML_TYPE_Q4_K;
            } else {
                new_type = if (is_s_or_m) c.GGML_TYPE_IQ3_S else c.GGML_TYPE_Q2_K;
            }
            qs.i_attention_wv += 1;
        } else if (hp.n_expert == 8 and category == .attention_k) {
            new_type = c.GGML_TYPE_Q4_K;
        } else if (category == .ffn_down) {
            if (qs.i_ffn_down < @divTrunc(qs.n_ffn_down, 8)) {
                new_type = if (is_s_or_m) c.GGML_TYPE_IQ3_S else c.GGML_TYPE_Q2_K;
            }
            qs.i_ffn_down += 1;
        } else if (category == .attention_output) {
            if (hp.n_expert == 8) {
                new_type = c.GGML_TYPE_Q5_K;
            } else {
                if (ftype == c.LLAMA_FTYPE_MOSTLY_IQ1_S or ftype == c.LLAMA_FTYPE_MOSTLY_IQ1_M) {
                    new_type = c.GGML_TYPE_IQ2_XXS;
                } else if (is_s_or_m) {
                    new_type = c.GGML_TYPE_IQ3_S;
                }
            }
        }
    } else if (categoryIsAttnV(category)) {
        if (ftype == c.LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = if (hp.nGqa(0) >= 4) c.GGML_TYPE_Q4_K else c.GGML_TYPE_Q3_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_Q2_K_S and hp.nGqa(0) >= 4) {
            new_type = c.GGML_TYPE_Q4_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = if (hp.nGqa(0) >= 4) c.GGML_TYPE_Q4_K else if (!qs.has_imatrix) c.GGML_TYPE_IQ3_S else c.GGML_TYPE_IQ3_XXS;
        } else if ((ftype == c.LLAMA_FTYPE_MOSTLY_IQ3_XS or ftype == c.LLAMA_FTYPE_MOSTLY_IQ3_S) and hp.nGqa(0) >= 4) {
            new_type = c.GGML_TYPE_Q4_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = c.GGML_TYPE_Q4_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = if (qs.i_attention_wv < 2) c.GGML_TYPE_Q5_K else c.GGML_TYPE_Q4_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = c.GGML_TYPE_Q5_K;
        } else if ((ftype == c.LLAMA_FTYPE_MOSTLY_IQ4_NL or ftype == c.LLAMA_FTYPE_MOSTLY_IQ4_XS) and hp.nGqa(0) >= 4) {
            new_type = c.GGML_TYPE_Q5_K;
        } else if ((ftype == c.LLAMA_FTYPE_MOSTLY_Q4_K_M or ftype == c.LLAMA_FTYPE_MOSTLY_Q5_K_M) and
            useMoreBits(qs.i_attention_wv, qs.n_attention_wv))
        {
            new_type = c.GGML_TYPE_Q6_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_Q4_K_S and qs.i_attention_wv < 4) {
            new_type = c.GGML_TYPE_Q5_K;
        }
        if (qs.model.type == .@"70b") {
            // Eight heads share attn_v in the 70B model, so it is 8x
            // smaller than attn_q: more bits cost almost nothing.
            if (new_type == c.GGML_TYPE_Q3_K or new_type == c.GGML_TYPE_Q4_K) new_type = c.GGML_TYPE_Q5_K;
        }
        if (hp.n_expert == 8) {
            // For the 8-expert model, Q8_0 trades just ~128MB.
            new_type = c.GGML_TYPE_Q8_0;
        }
        qs.i_attention_wv += 1;
    } else if (category == .attention_k) {
        if (hp.n_expert == 8) {
            new_type = c.GGML_TYPE_Q8_0;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = c.GGML_TYPE_IQ3_XXS;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = c.GGML_TYPE_IQ2_S;
        }
    } else if (category == .attention_q) {
        if (ftype == c.LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = c.GGML_TYPE_IQ3_XXS;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = c.GGML_TYPE_IQ2_S;
        }
    } else if (category == .ffn_down) {
        const info = try layerInfo(n_expert, qs.i_ffn_down, qs.n_ffn_down, name, exc);
        const i_layer = info[0];
        const n_layer = info[1];
        if (ftype == c.LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = c.GGML_TYPE_Q3_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < @divTrunc(n_layer, 8)) new_type = c.GGML_TYPE_Q4_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_IQ3_XXS and !qs.has_imatrix) {
            new_type = if (i_layer < @divTrunc(n_layer, 8)) c.GGML_TYPE_Q4_K else c.GGML_TYPE_Q3_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = if (i_layer < @divTrunc(n_layer, 16))
                c.GGML_TYPE_Q5_K
            else if (a != .falcon or useMoreBits(i_layer, n_layer))
                c.GGML_TYPE_Q4_K
            else
                c.GGML_TYPE_Q3_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_IQ3_M and (i_layer < @divTrunc(n_layer, 8) or
            (hp.n_expert == 8 and useMoreBits(i_layer, n_layer))))
        {
            new_type = c.GGML_TYPE_Q4_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = if (a == .falcon) c.GGML_TYPE_Q4_K else c.GGML_TYPE_Q5_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (a == .falcon) {
                new_type = if (i_layer < @divTrunc(n_layer, 16))
                    c.GGML_TYPE_Q6_K
                else if (useMoreBits(i_layer, n_layer))
                    c.GGML_TYPE_Q5_K
                else
                    c.GGML_TYPE_Q4_K;
            } else {
                if (useMoreBits(i_layer, n_layer)) new_type = c.GGML_TYPE_Q6_K;
            }
        } else if (i_layer < @divTrunc(n_layer, 8) and
            (ftype == c.LLAMA_FTYPE_MOSTLY_IQ4_NL or ftype == c.LLAMA_FTYPE_MOSTLY_IQ4_XS) and !qs.has_imatrix)
        {
            new_type = c.GGML_TYPE_Q5_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_Q5_K_M and useMoreBits(i_layer, n_layer)) {
            new_type = c.GGML_TYPE_Q6_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_Q4_K_S and a != .falcon and i_layer < @divTrunc(n_layer, 8)) {
            new_type = c.GGML_TYPE_Q5_K;
        } else if ((ftype == c.LLAMA_FTYPE_MOSTLY_Q4_0 or ftype == c.LLAMA_FTYPE_MOSTLY_Q5_0) and
            qs.has_imatrix and i_layer < @divTrunc(n_layer, 8))
        {
            // Guard the first ffn_down layers, only with an imatrix.
            new_type = if (ftype == c.LLAMA_FTYPE_MOSTLY_Q4_0) c.GGML_TYPE_Q4_1 else c.GGML_TYPE_Q5_1;
        }
        qs.i_ffn_down += 1;
    } else if (category == .attention_output) {
        if (a != .falcon) {
            if (hp.n_expert == 8) {
                if (F.in(ftype, .{
                    c.LLAMA_FTYPE_MOSTLY_Q2_K,   c.LLAMA_FTYPE_MOSTLY_IQ3_XS, c.LLAMA_FTYPE_MOSTLY_IQ3_XXS,
                    c.LLAMA_FTYPE_MOSTLY_Q3_K_S, c.LLAMA_FTYPE_MOSTLY_Q3_K_M, c.LLAMA_FTYPE_MOSTLY_IQ4_NL,
                    c.LLAMA_FTYPE_MOSTLY_Q4_K_S, c.LLAMA_FTYPE_MOSTLY_Q4_K_M, c.LLAMA_FTYPE_MOSTLY_IQ3_S,
                    c.LLAMA_FTYPE_MOSTLY_IQ3_M,  c.LLAMA_FTYPE_MOSTLY_IQ4_XS,
                })) {
                    new_type = c.GGML_TYPE_Q5_K;
                }
            } else {
                if (ftype == c.LLAMA_FTYPE_MOSTLY_Q2_K) {
                    new_type = c.GGML_TYPE_Q3_K;
                } else if (ftype == c.LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
                    new_type = c.GGML_TYPE_IQ3_S;
                } else if (ftype == c.LLAMA_FTYPE_MOSTLY_Q3_K_M) {
                    new_type = c.GGML_TYPE_Q4_K;
                } else if (ftype == c.LLAMA_FTYPE_MOSTLY_Q3_K_L) {
                    new_type = c.GGML_TYPE_Q5_K;
                } else if (ftype == c.LLAMA_FTYPE_MOSTLY_IQ3_M) {
                    new_type = c.GGML_TYPE_Q4_K;
                }
            }
        } else {
            if (ftype == c.LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = c.GGML_TYPE_Q4_K;
        }
    } else if (category == .attention_qkv) {
        if (F.in(ftype, .{ c.LLAMA_FTYPE_MOSTLY_Q3_K_M, c.LLAMA_FTYPE_MOSTLY_Q3_K_L, c.LLAMA_FTYPE_MOSTLY_IQ3_M })) {
            new_type = c.GGML_TYPE_Q4_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            new_type = c.GGML_TYPE_Q5_K;
        } else if (ftype == c.LLAMA_FTYPE_MOSTLY_Q5_K_M) {
            new_type = c.GGML_TYPE_Q6_K;
        }
    } else if (category == .ffn_gate) {
        const info = try layerInfo(n_expert, qs.i_ffn_gate, qs.n_ffn_gate, name, exc);
        const i_layer = info[0];
        const n_layer = info[1];
        if (ftype == c.LLAMA_FTYPE_MOSTLY_IQ3_XS and (i_layer >= @divTrunc(n_layer, 8) and i_layer < @divTrunc(7 * n_layer, 8))) {
            new_type = c.GGML_TYPE_IQ3_XXS;
        }
        qs.i_ffn_gate += 1;
    } else if (category == .ffn_up) {
        const info = try layerInfo(n_expert, qs.i_ffn_up, qs.n_ffn_up, name, exc);
        const i_layer = info[0];
        const n_layer = info[1];
        if (ftype == c.LLAMA_FTYPE_MOSTLY_IQ3_XS and (i_layer >= @divTrunc(n_layer, 8) and i_layer < @divTrunc(7 * n_layer, 8))) {
            new_type = c.GGML_TYPE_IQ3_XXS;
        }
        qs.i_ffn_up += 1;
    }

    return new_type;
}

/// Ports `llama_tensor_get_type` (llama-quant.cpp:679 @c1d0e7a00): the type
/// a tensor is written as -- the user's overrides, else the mixture, then
/// the fallback for its shape.
fn getType(qs: *State, params: *const c.llama_model_quantize_params, tensor: *const Tensor, default_type: GgmlType, category: Category, exc: *Exception) Thrown!GgmlType {
    if (!tensorAllowsQuantization(params, qs.model.arch, tensor)) return tensor.type;
    if (params.token_embedding_type < c.GGML_TYPE_COUNT and category == .token_embd) return params.token_embedding_type;
    if (params.output_tensor_type < c.GGML_TYPE_COUNT and category == .output) return params.output_tensor_type;

    var new_type = default_type;

    if (c.ggml_is_quantized(default_type)) {
        // The user's tensor types come first.
        var manual = false;
        if (qs.tensor_type_patterns.items.len != 0) {
            const tensor_name = nameOf(tensor);
            for (qs.tensor_type_patterns.items) |*p| {
                var m = regex.Match.init();
                defer m.deinit();
                const hit = p.re.trySearch(tensor_name, &m) catch |e| return exc.throw(.runtime_error, "{s}", .{regex.errorText(e)});
                if (hit) {
                    if (p.type != new_type) {
                        impl.logWarn("%s: %-36s - applying manual override: %s -> %s\n", .{
                            "llama_tensor_get_type", tensor_name.ptr, typeName(new_type), typeName(p.type),
                        });
                        new_type = p.type;
                    }
                    manual = true;
                    break;
                }
            }
        }

        if (!manual and !params.pure) {
            new_type = try getTypeImpl(qs, new_type, tensor, params.ftype, category, exc);
        }

        new_type = try typeFallback(qs, tensor, new_type, exc);
    }

    return new_type;
}

// -----------------------------------------------------------------------------
// Quantization

/// What `llama_tensor_quantize_impl`'s threads share.
const QuantizeJob = struct {
    counter: std.atomic.Value(i64) = .init(0),
    new_size: std.atomic.Value(usize) = .init(0),
    valid: std.atomic.Value(bool) = .init(true),
    new_type: GgmlType,
    f32_data: [*]const f32,
    new_data: [*]u8,
    chunk_size: i64,
    nrows: i64,
    n_per_row: i64,
    imatrix: ?[*]const f32,

    /// The `compute` lambda of `llama_tensor_quantize_impl`
    /// (llama-quant.cpp:727 @c1d0e7a00). The C++ takes its rows and adds
    /// its total under a mutex; an atomic add is the same exchange.
    fn run(job: *QuantizeJob) void {
        const nrows_per_chunk = @divTrunc(job.chunk_size, job.n_per_row);
        var local_size: usize = 0;
        while (true) {
            const first_row = job.counter.fetchAdd(nrows_per_chunk, .seq_cst);
            if (first_row >= job.nrows) {
                if (local_size > 0) _ = job.new_size.fetchAdd(local_size, .seq_cst);
                break;
            }
            const this_nrow = @min(job.nrows - first_row, nrows_per_chunk);
            const this_size = c.ggml_quantize_chunk(job.new_type, job.f32_data, job.new_data, first_row * job.n_per_row, this_nrow, job.n_per_row, job.imatrix);
            local_size += this_size;

            const row_size = c.ggml_row_size(job.new_type, job.n_per_row);
            const this_data = job.new_data + @as(usize, @intCast(first_row)) * row_size;
            if (!c.ggml_validate_row_data(job.new_type, this_data, this_size)) {
                job.valid.store(false, .seq_cst);
                break;
            }
        }
    }
};

/// Ports `llama_tensor_quantize_impl` (llama-quant.cpp:727 @c1d0e7a00):
/// `nrows` rows of `f32_data` as `new_type`, chunk by chunk on `nthread`
/// threads.
///
/// Return: the size written, or `error.Thrown` when validation fails.
fn quantizeRows(new_type: GgmlType, f32_data: [*]const f32, new_data: [*]u8, chunk_size: i64, nrows: i64, n_per_row: i64, imatrix: ?[*]const f32, nthread: i64, exc: *Exception) Thrown!usize {
    if (nthread < 2) {
        const new_size = c.ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix);
        if (!c.ggml_validate_row_data(new_type, new_data, new_size)) {
            return exc.throw(.runtime_error, "quantized data validation failed", .{});
        }
        return new_size;
    }

    var job: QuantizeJob = .{
        .new_type = new_type,
        .f32_data = f32_data,
        .new_data = new_data,
        .chunk_size = chunk_size,
        .nrows = nrows,
        .n_per_row = n_per_row,
        .imatrix = imatrix,
    };
    const workers = allocator.alloc(std.Thread, @intCast(nthread - 1)) catch oom();
    defer allocator.free(workers);
    for (workers) |*w| w.* = std.Thread.spawn(.{}, QuantizeJob.run, .{&job}) catch oom();
    job.run();
    for (workers) |w| w.join();
    if (!job.valid.load(.seq_cst)) {
        return exc.throw(.runtime_error, "quantized data validation failed", .{});
    }
    return job.new_size.load(.seq_cst);
}

/// Ports `tensor_requires_imatrix` (llama-quant.cpp:785 @c1d0e7a00).
fn requiresImatrix(tensor_name: []const u8, dst_type: GgmlType, ftype: Ftype) bool {
    if (nameMatchTokenEmbd(tensor_name) or nameMatchOutputWeight(tensor_name)) return false;
    return switch (dst_type) {
        c.GGML_TYPE_IQ3_XXS,
        c.GGML_TYPE_IQ2_XXS,
        c.GGML_TYPE_IQ2_XS,
        c.GGML_TYPE_IQ2_S,
        c.GGML_TYPE_IQ1_M,
        c.GGML_TYPE_IQ1_S,
        => true,
        // Only Q2_K tensors in a Q2_K_S file need one.
        c.GGML_TYPE_Q2_K => ftype == c.LLAMA_FTYPE_MOSTLY_Q2_K_S,
        else => false,
    };
}

/// Ports `llama_ftype_get_default_type` (llama-quant.cpp:810 @c1d0e7a00).
///
/// Parameters:
/// - `ftype`: the file type.
///
/// Return: its default tensor type, `GGML_TYPE_COUNT` when there is none.
pub fn ftypeGetDefaultType(ftype: Ftype) GgmlType {
    return switch (ftype) {
        c.LLAMA_FTYPE_MOSTLY_Q4_0 => c.GGML_TYPE_Q4_0,
        c.LLAMA_FTYPE_MOSTLY_Q4_1 => c.GGML_TYPE_Q4_1,
        c.LLAMA_FTYPE_MOSTLY_Q5_0 => c.GGML_TYPE_Q5_0,
        c.LLAMA_FTYPE_MOSTLY_Q5_1 => c.GGML_TYPE_Q5_1,
        c.LLAMA_FTYPE_MOSTLY_Q8_0 => c.GGML_TYPE_Q8_0,
        c.LLAMA_FTYPE_MOSTLY_F16 => c.GGML_TYPE_F16,
        c.LLAMA_FTYPE_MOSTLY_BF16 => c.GGML_TYPE_BF16,
        c.LLAMA_FTYPE_ALL_F32 => c.GGML_TYPE_F32,
        c.LLAMA_FTYPE_MOSTLY_Q1_0 => c.GGML_TYPE_Q1_0,
        c.LLAMA_FTYPE_MOSTLY_Q2_0 => c.GGML_TYPE_Q2_0,

        c.LLAMA_FTYPE_MOSTLY_MXFP4_MOE => c.GGML_TYPE_MXFP4,

        // K-quants
        c.LLAMA_FTYPE_MOSTLY_Q2_K_S, c.LLAMA_FTYPE_MOSTLY_Q2_K => c.GGML_TYPE_Q2_K,
        c.LLAMA_FTYPE_MOSTLY_IQ3_XS => c.GGML_TYPE_IQ3_S,
        c.LLAMA_FTYPE_MOSTLY_Q3_K_S, c.LLAMA_FTYPE_MOSTLY_Q3_K_M, c.LLAMA_FTYPE_MOSTLY_Q3_K_L => c.GGML_TYPE_Q3_K,
        c.LLAMA_FTYPE_MOSTLY_Q4_K_S, c.LLAMA_FTYPE_MOSTLY_Q4_K_M => c.GGML_TYPE_Q4_K,
        c.LLAMA_FTYPE_MOSTLY_Q5_K_S, c.LLAMA_FTYPE_MOSTLY_Q5_K_M => c.GGML_TYPE_Q5_K,
        c.LLAMA_FTYPE_MOSTLY_Q6_K => c.GGML_TYPE_Q6_K,
        c.LLAMA_FTYPE_MOSTLY_TQ1_0 => c.GGML_TYPE_TQ1_0,
        c.LLAMA_FTYPE_MOSTLY_TQ2_0 => c.GGML_TYPE_TQ2_0,
        c.LLAMA_FTYPE_MOSTLY_IQ2_XXS => c.GGML_TYPE_IQ2_XXS,
        c.LLAMA_FTYPE_MOSTLY_IQ2_XS => c.GGML_TYPE_IQ2_XS,
        c.LLAMA_FTYPE_MOSTLY_IQ2_S => c.GGML_TYPE_IQ2_XS,
        c.LLAMA_FTYPE_MOSTLY_IQ2_M => c.GGML_TYPE_IQ2_S,
        c.LLAMA_FTYPE_MOSTLY_IQ3_XXS => c.GGML_TYPE_IQ3_XXS,
        c.LLAMA_FTYPE_MOSTLY_IQ1_S => c.GGML_TYPE_IQ1_S,
        c.LLAMA_FTYPE_MOSTLY_IQ1_M => c.GGML_TYPE_IQ1_M,
        c.LLAMA_FTYPE_MOSTLY_IQ4_NL => c.GGML_TYPE_IQ4_NL,
        c.LLAMA_FTYPE_MOSTLY_IQ4_XS => c.GGML_TYPE_IQ4_XS,
        c.LLAMA_FTYPE_MOSTLY_IQ3_S, c.LLAMA_FTYPE_MOSTLY_IQ3_M => c.GGML_TYPE_IQ3_S,

        else => c.GGML_TYPE_COUNT,
    };
}

/// Ports `init_quantize_state_counters` (llama-quant.cpp:856 @c1d0e7a00).
fn initCounters(qs: *State, names: []const []const u8, categories: []Category) void {
    for (names, categories) |name, *cat_out| {
        const cat = getCategory(name);
        cat_out.* = cat;
        if (categoryIsAttnV(cat)) qs.n_attention_wv += 1;
        if (cat == .output) qs.has_tied_embeddings = false;
    }
    const n: c_int = @bitCast(qs.model.hparams.n_layer_all);
    qs.n_ffn_down = n;
    qs.n_ffn_gate = n;
    qs.n_ffn_up = n;
}

/// Ports `llama_format_tensor_shape` for a tensor (llama-impl.cpp:110 @c1d0e7a00).
fn formatTensorShape(t: *const Tensor, buf: *[256]u8) [*:0]const u8 {
    var len: usize = @intCast(c.snprintf(buf, buf.len, "%6lld", @as(c_longlong, t.ne[0])));
    var i: usize = 1;
    while (i < c.GGML_MAX_DIMS) : (i += 1) {
        len += @intCast(c.snprintf(buf[len..].ptr, buf.len - len, ", %6lld", @as(c_longlong, t.ne[i])));
    }
    return @ptrCast(buf);
}

// -----------------------------------------------------------------------------
// The output file

/// The `std::ofstream` of `llama_model_quantize_impl`, with
/// `exceptions(failbit)` set: a file that cannot be opened throws libc++'s
/// `ios_base::failure` the moment the mask is set.
const OutFile = struct {
    fp: ?*c.FILE = null,

    fn open(path: [*:0]const u8, exc: *Exception) Thrown!OutFile {
        const fp = c.fopen(path, "wb") orelse
            return exc.throw(.runtime_error, "ios_base::clear: unspecified iostream_category error", .{});
        return .{ .fp = fp };
    }

    fn isOpen(self: *const OutFile) bool {
        return self.fp != null;
    }

    fn write(self: *OutFile, data: [*]const u8, n: usize) void {
        if (n != 0) _ = c.fwrite(data, 1, n, self.fp);
    }

    /// Ports `zeros` (llama-quant.cpp:41 @c1d0e7a00).
    fn zeros(self: *OutFile, n: usize) void {
        const z: [4096]u8 = @splat(0);
        var left = n;
        while (left > 0) {
            const k = @min(left, z.len);
            self.write(&z, k);
            left -= k;
        }
    }

    fn seekStart(self: *OutFile) void {
        _ = c.fseek(self.fp, 0, c.SEEK_SET);
    }

    fn close(self: *OutFile) void {
        if (self.fp) |fp| _ = c.fclose(fp);
        self.fp = null;
    }
};

// -----------------------------------------------------------------------------
// The driver

/// A tensor's data loaded by the name it now has, looked up among the names
/// the loader read, as `load_data_for` does (llama-model-loader.cpp:1403
/// @c1d0e7a00) in the `LLAMA_LOAD_MODE_NONE` arm: read from the file into
/// `cur.data`.
fn loadDataFor(ml: *const loader.Loader, by_name: *const std.StringHashMapUnmanaged(*loader.Weight), cur: *Tensor, exc: *Exception) Thrown!void {
    const name = nameOf(cur);
    const w = by_name.get(name) orelse
        return exc.throw(.runtime_error, "{s}: tensor '{s}' not found", .{ "require_weight", name });
    impl.assert(cur.data != null, "llama-model-loader.cpp", 1414, "cur->data != nullptr");
    impl.assert(w.idx < ml.files.items.len, "llama-model-loader.cpp", 1415, "w.idx < files.size()");
    const file = &ml.files.items[w.idx];
    try file.seek(w.offs, c.SEEK_SET, exc);
    try file.readRaw(cur.data.?, c.ggml_nbytes(cur), exc);
    if (ml.check_tensors and !c.ggml_validate_row_data(cur.type, cur.data, c.ggml_nbytes(cur))) {
        return exc.throw(.runtime_error, "tensor '{s}' has invalid data", .{name});
    }
}

fn mib(n: usize) f64 {
    return @as(f64, @floatFromInt(n)) / 1024.0 / 1024.0;
}

/// Ports `llama_model_quantize_impl` (llama-quant.cpp:876 @c1d0e7a00).
///
/// Parameters:
/// - `fname_inp`: the model to read.
/// - `fname_out`: the file to write, or the prefix of the splits with
///   `keep_split`.
/// - `params`: the parameters.
/// - `exc`: receives what is thrown.
///
/// Return: nothing, or `error.Thrown`.
pub fn quantize(fname_inp: [:0]const u8, fname_out: [:0]const u8, params: *const c.llama_model_quantize_params, exc: *Exception) Thrown!void {
    var ftype: Ftype = params.ftype;

    var nthread: c_int = params.nthread;
    if (nthread <= 0) nthread = @intCast(std.Thread.getCpuCount() catch 1);

    const default_type = ftypeGetDefaultType(ftype);
    if (default_type == c.GGML_TYPE_COUNT) {
        return exc.throw(.runtime_error, "invalid output file type {d}\n", .{@as(c_int, @bitCast(ftype))});
    }

    // `#else` arm: mmap is not used on this target.
    const load_mode = c.LLAMA_LOAD_MODE_NONE;

    var splits: std.ArrayList([:0]const u8) = .empty;
    defer {
        for (splits.items) |s| allocator.free(s);
        splits.deinit(allocator);
    }
    var ml = try loader.Loader.init(.{ .path = fname_inp }, &splits, load_mode, true, false, true, params.kv_overrides, exc);
    defer ml.deinit();
    try ml.initMappings(false, null, exc);

    var model = try Model.create(&ml, model_mod.defaultParams(), exc);
    defer model.deinit();

    model.loadHparams(&ml, exc) catch |e| switch (e) {
        error.Thrown => return error.Thrown,
        error.Unported => return exc.throw(.runtime_error, "load_arch_hparams is not ported for architecture '{s}'", .{model.archName()}),
    };
    model.loadStats(&ml);

    var qs = try State.init(&model, params, exc);
    defer qs.deinit();

    if (params.only_copy) ftype = ml.ftype;

    // `std::unordered_map<std::string, std::vector<float>>`, iterated in
    // libc++'s order by the finiteness check.
    var i_data = libcxx.StringMap([]const f32).init(allocator) catch oom();
    defer i_data.deinit();
    var have_imatrix = false;
    if (params.imatrix != null) {
        var p = params.imatrix;
        while (p[0].name != null) : (p += 1) {
            const data: []const f32 = if (p[0].size == 0) &.{} else p[0].data[0..p[0].size];
            _ = i_data.getOrPut(std.mem.span(p[0].name), data) catch oom();
        }
        have_imatrix = true;
        impl.logInfo("\n%s: have importance matrix data with %d entries\n", .{ "llama_model_quantize_impl", @as(c_int, @intCast(i_data.count())) });
        qs.has_imatrix = true;
        var it = i_data.iterator();
        while (it.next()) |kv| {
            for (kv.value.*) |f| {
                if (!std.math.isFinite(f)) {
                    var buf: [64]u8 = undefined;
                    _ = c.snprintf(&buf, buf.len, "%f", @as(f64, f));
                    return exc.throw(.runtime_error, "imatrix contains non-finite value {s}\n", .{std.mem.sliceTo(&buf, 0)});
                }
            }
        }
    }

    const alignment: usize = c.GGUF_DEFAULT_ALIGNMENT;
    const ctx_out = c.gguf_init_empty().?;
    var ctx_out_owned = true;
    defer if (ctx_out_owned) c.gguf_free(ctx_out);

    var prune_list: std.ArrayList(c_int) = .empty;
    defer prune_list.deinit(allocator);
    if (params.prune_layers != null) {
        var p = params.prune_layers;
        while (p[0] != -1) : (p += 1) prune_list.append(allocator, p[0]) catch oom();
    }

    const a = ml.meta.arch;

    // The input's key-value pairs.
    c.gguf_set_kv(ctx_out, ml.meta.ctx);
    c.gguf_set_val_u32(ctx_out, KeyZ.init(a, .general_quantization_version).ptr(), c.GGML_QNT_VERSION);
    c.gguf_set_val_u32(ctx_out, KeyZ.init(a, .general_file_type).ptr(), ftype);

    // No split metadata.
    _ = c.gguf_remove_key(ctx_out, KeyZ.init(a, .split_no).ptr());
    _ = c.gguf_remove_key(ctx_out, KeyZ.init(a, .split_count).ptr());
    _ = c.gguf_remove_key(ctx_out, KeyZ.init(a, .split_tensors_count).ptr());

    if (params.kv_overrides != null) {
        var o = params.kv_overrides;
        while (o[0].key[0] != 0) : (o += 1) {
            const key: [*:0]const u8 = @ptrCast(&o[0].key);
            switch (o[0].tag) {
                c.LLAMA_KV_OVERRIDE_TYPE_FLOAT => c.gguf_set_val_f32(ctx_out, key, @floatCast(o[0].unnamed_0.val_f64)),
                // Written as UINT32 (llama.cpp#14182).
                c.LLAMA_KV_OVERRIDE_TYPE_INT => c.gguf_set_val_u32(ctx_out, key, @truncate(@abs(o[0].unnamed_0.val_i64))),
                c.LLAMA_KV_OVERRIDE_TYPE_BOOL => c.gguf_set_val_bool(ctx_out, key, o[0].unnamed_0.val_bool),
                c.LLAMA_KV_OVERRIDE_TYPE_STR => c.gguf_set_val_str(ctx_out, key, @ptrCast(&o[0].unnamed_0.val_str)),
                else => impl.logWarn("%s: unknown KV override type for key %s\n", .{ "llama_model_quantize_impl", key }),
            }
        }
    }

    var re_blk = regex.Regex.compile(blk_pattern) catch unreachable;
    defer re_blk.deinit();

    var mapped: LayerMap = .{};
    defer mapped.deinit();
    var blk_id: c_int = 0;

    // `weights_map` keyed by the names the loader read, before any rename.
    var by_name: std.StringHashMapUnmanaged(*loader.Weight) = .empty;
    defer {
        var kit = by_name.keyIterator();
        while (kit.next()) |k| allocator.free(k.*);
        by_name.deinit(allocator);
    }
    for (ml.weights.items) |*w| {
        by_name.put(allocator, allocator.dupe(u8, nameOf(w.tensor)) catch oom(), w) catch oom();
    }

    // The list of weights.
    var tensors: std.ArrayList(*loader.Weight) = .empty;
    defer tensors.deinit(allocator);
    tensors.ensureTotalCapacity(allocator, ml.weights.items.len) catch oom();
    for (ml.weights.items) |*w| {
        const orig = allocator.dupeSentinel(u8, nameOf(w.tensor), 0) catch oom();
        defer allocator.free(orig);
        const remapped = try remapLayer(&re_blk, orig, prune_list.items, &mapped, &blk_id, exc);
        defer allocator.free(remapped);
        if (remapped.len == 0) {
            impl.logDebug("%s: pruning tensor %s\n", .{ "llama_model_quantize_impl", orig.ptr });
            continue;
        }
        if (!std.mem.eql(u8, remapped, orig)) {
            const z = allocator.dupeSentinel(u8, remapped, 0) catch oom();
            defer allocator.free(z);
            _ = c.ggml_set_name(w.tensor, z.ptr);
            impl.logDebug("%s: tensor %s remapped to %s\n", .{ "llama_model_quantize_impl", orig.ptr, c.ggml_get_name(w.tensor) });
        }
        tensors.appendAssumeCapacity(w);
    }
    if (prune_list.items.len != 0) {
        c.gguf_set_val_u32(ctx_out, KeyZ.init(a, .block_count).ptr(), @bitCast(blk_id));
    }

    // keep_split needs the weights in split order.
    if (params.keep_split) {
        libcxx.sort(*loader.Weight, tensors.items, {}, struct {
            fn lt(_: void, x: *loader.Weight, y: *loader.Weight) bool {
                if (x.idx == y.idx) return x.offs < y.offs;
                return x.idx < y.idx;
            }
        }.lt);
    }

    // Each tensor's metadata, computed once.
    const metadata = allocator.alloc(Metadata, tensors.items.len) catch oom();
    for (metadata) |*m| m.* = .{};
    defer {
        for (metadata) |*m| m.deinit();
        allocator.free(metadata);
    }
    const names = allocator.alloc([]const u8, tensors.items.len) catch oom();
    defer allocator.free(names);
    const categories = allocator.alloc(Category, tensors.items.len) catch oom();
    defer allocator.free(categories);
    for (tensors.items, metadata, names) |w, *m, *n| {
        m.name = allocator.dupeSentinel(u8, nameOf(w.tensor), 0) catch oom();
        n.* = m.name;
    }

    initCounters(&qs, names, categories);
    for (metadata, categories) |*m, cat| m.category = cat;

    var idx: c_int = 0;
    var n_split: u16 = 1;

    // The split indices are assumed contiguous.
    if (params.keep_split) {
        for (tensors.items) |w| n_split = @max(w.idx +% 1, n_split);
    }
    const ctx_outs = allocator.alloc(?*c.gguf_context, n_split) catch oom();
    for (ctx_outs) |*x| x.* = null;
    defer {
        for (ctx_outs) |x| if (x) |ctx| c.gguf_free(ctx);
        allocator.free(ctx_outs);
    }
    ctx_outs[0] = ctx_out;
    ctx_out_owned = false;

    // For dry_run.
    var will_require_imatrix = false;

    // The preliminary pass over every weight.
    for (tensors.items, metadata) |w, *m| {
        const tensor = w.tensor;

        const i_split: u16 = if (params.keep_split) w.idx else 0;
        if (ctx_outs[i_split] == null) ctx_outs[i_split] = c.gguf_init_empty();
        c.gguf_add_tensor(ctx_outs[i_split], tensor);

        m.allows_quantization = tensorAllowsQuantization(params, model.arch, tensor);

        if (m.allows_quantization) {
            m.target_type = try getType(&qs, params, tensor, default_type, m.category, exc);
        } else {
            m.target_type = tensor.type;
        }

        m.requires_imatrix = requiresImatrix(nameOf(tensor), m.target_type, ftype);

        if (params.imatrix != null) {
            m.remapped_imatrix_name = try remapImatrix(&re_blk, nameOf(tensor), &mapped, exc);
        } else if (m.allows_quantization and m.requires_imatrix) {
            if (params.dry_run) {
                will_require_imatrix = true;
            } else {
                impl.logError("\n============================================================================\n" ++
                    " ERROR: this quantization requires an importance matrix!\n" ++
                    "        - offending tensor: %s\n" ++
                    "        - target type: %s\n" ++
                    "============================================================================\n\n", .{ m.name.ptr, typeName(m.target_type) });
                return exc.throw(.runtime_error, "this quantization requires an imatrix!", .{});
            }
        }
    }

    // Split info.
    if (n_split > 1) {
        for (ctx_outs, 0..) |ctx, i| {
            c.gguf_set_val_u16(ctx, KeyZ.init(a, .split_no).ptr(), @intCast(i));
            c.gguf_set_val_u16(ctx, KeyZ.init(a, .split_count).ptr(), n_split);
            c.gguf_set_val_i32(ctx, KeyZ.init(a, .split_tensors_count).ptr(), @intCast(tensors.items.len));
        }
    }

    var total_size_org: usize = 0;
    var total_size_new: usize = 0;

    var read_data: std.ArrayList(u8) = .empty;
    defer read_data.deinit(allocator);
    var work: std.ArrayList(u8) = .empty;
    defer work.deinit(allocator);
    var f32_conv_buf: std.ArrayList(f32) = .empty;
    defer f32_conv_buf.deinit(allocator);

    var cur_split: c_int = -1;
    var fout: OutFile = .{};
    defer fout.close();

    const Files = struct {
        /// The `close_ofstream` lambda: the metadata goes in front, now
        /// that every tensor's offset is known.
        fn close(f: *OutFile, ctxs: []?*c.gguf_context, cur: c_int) void {
            if (!f.isOpen()) return;
            f.seekStart();
            const ctx = ctxs[@intCast(cur)];
            const data = allocator.alloc(u8, c.gguf_get_meta_size(ctx)) catch oom();
            defer allocator.free(data);
            c.gguf_get_meta_data(ctx, data.ptr);
            f.write(data.ptr, data.len);
            f.close();
        }

        /// The `new_ofstream` lambda.
        fn open(f: *OutFile, ctxs: []?*c.gguf_context, cur: *c_int, index: c_int, out: [:0]const u8, keep_split: bool, n: u16, exc_: *Exception) Thrown!void {
            cur.* = index;
            impl.assert(ctxs[@intCast(index)] != null, "llama-quant.cpp", 1110, "ctx_outs[cur_split] && \"Find uninitialized gguf_context\"");
            var path_buf: ?[]u8 = null;
            defer if (path_buf) |b| allocator.free(b);
            var fname: [*:0]const u8 = out.ptr;
            if (keep_split) {
                const b = allocator.alloc(u8, mmap.pathMax()) catch oom();
                @memset(b, 0);
                path_buf = b;
                _ = c.llama_split_path(b.ptr, b.len, out.ptr, index, n);
                fname = @ptrCast(b.ptr);
            }
            f.* = try OutFile.open(fname, exc_);
            const meta_size = c.gguf_get_meta_size(ctxs[@intCast(index)]);
            // A placeholder for the metadata.
            f.zeros(meta_size);
        }
    };

    // No output file for dry_run.
    if (!params.dry_run) try Files.open(&fout, ctx_outs, &cur_split, 0, fname_out, params.keep_split, n_split, exc);

    // The main pass.
    for (tensors.items, metadata) |weight, *tm| {
        const tensor = weight.tensor;

        if (!params.dry_run and (weight.idx != cur_split and params.keep_split)) {
            Files.close(&fout, ctx_outs, cur_split);
            try Files.open(&fout, ctx_outs, &cur_split, weight.idx, fname_out, params.keep_split, n_split, exc);
        }

        const tensor_size = c.ggml_nbytes(tensor);

        if (!params.dry_run) {
            if (read_data.items.len < tensor_size) read_data.resize(allocator, tensor_size) catch oom();
            tensor.data = read_data.items.ptr;
            try loadDataFor(&ml, &by_name, tensor, exc);
        }

        idx += 1;
        var shape: [256]u8 = undefined;
        impl.logInfo("[%4d/%4d] %-36s - [%s], type = %6s, ", .{
            idx, ml.n_tensors, c.ggml_get_name(tensor), formatTensorShape(tensor, &shape), typeName(tensor.type),
        });

        const cur_type = tensor.type;
        const new_type = tm.target_type;

        // Already the type chosen: nothing to do.
        const do_quantize = cur_type != new_type;

        var new_data: [*]u8 = undefined;
        var new_size: usize = undefined;

        if (params.dry_run) {
            // The size the quantization would have, without quantizing.
            if (do_quantize) {
                new_size = @as(usize, @intCast(c.ggml_nrows(tensor))) * c.ggml_row_size(new_type, tensor.ne[0]);
                impl.logInfo("size = %8.2f MiB -> %8.2f MiB (%s)\n", .{ mib(tensor_size), mib(new_size), typeName(new_type) });
                if (!will_require_imatrix and tm.requires_imatrix) will_require_imatrix = true;
            } else {
                new_size = tensor_size;
                impl.logInfo("size = %8.3f MiB\n", .{mib(new_size)});
            }
            total_size_org += tensor_size;
            total_size_new += new_size;
            continue;
        }

        if (!do_quantize) {
            new_data = @ptrCast(tensor.data);
            new_size = tensor_size;
            impl.logInfo("size = %8.3f MiB\n", .{mib(tensor_size)});
        } else {
            const nelements: usize = @intCast(c.ggml_nelements(tensor));

            var imatrix: ?[*]const f32 = null;
            if (have_imatrix) {
                if (i_data.get(tm.remapped_imatrix_name)) |entry| {
                    const want: usize = @intCast(tensor.ne[0] * tensor.ne[2]);
                    if (entry.*.len == want) {
                        imatrix = entry.*.ptr;
                    } else {
                        impl.logInfo("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", .{
                            "llama_model_quantize_impl", @as(c_int, @truncate(@as(i64, @intCast(entry.*.len)))), @as(c_int, @truncate(tensor.ne[0] * tensor.ne[2])), c.ggml_get_name(tensor),
                        });
                        // An old split Mixtral with a new imatrix lands
                        // here; tok_embd always does, and is let through.
                        if (!nameMatchTokenEmbd(nameOf(tensor))) {
                            return exc.throw(.runtime_error, "imatrix size {d} is different from tensor size {d} for {s}", .{
                                @as(c_int, @truncate(@as(i64, @intCast(entry.*.len)))), @as(c_int, @truncate(tensor.ne[0] * tensor.ne[2])), nameOf(tensor),
                            });
                        }
                    }
                } else {
                    impl.logInfo("\n====== %s: did not find weights for %s\n", .{ "llama_model_quantize_impl", c.ggml_get_name(tensor) });
                }
            }
            if (imatrix == null and tm.requires_imatrix) {
                impl.logError("\n\n============================================================\n", .{});
                impl.logError("Missing importance matrix for tensor %s in a very low-bit quantization\n", .{c.ggml_get_name(tensor)});
                impl.logError("The result will be garbage, so bailing out\n", .{});
                impl.logError("============================================================\n\n", .{});
                return exc.throw(.runtime_error, "Missing importance matrix for tensor {s} in a very low-bit quantization", .{nameOf(tensor)});
            }

            var f32_data: [*]f32 = undefined;
            if (tensor.type == c.GGML_TYPE_F32) {
                f32_data = @ptrCast(@alignCast(tensor.data));
            } else if (c.ggml_is_quantized(tensor.type) and !params.allow_requantize) {
                return exc.throw(.runtime_error, "requantizing from type {s} is disabled", .{typeName(tensor.type)});
            } else {
                try dequantize(tensor, &f32_conv_buf, nelements, nthread, exc);
                f32_data = f32_conv_buf.items.ptr;
            }

            impl.logInfo("converting to %s .. ", .{typeName(new_type)});
            _ = c.fflush(impl.cStdout());

            if (work.items.len < nelements * 4) work.resize(allocator, nelements * 4) catch oom();
            new_data = work.items.ptr;

            const n_per_row = tensor.ne[0];
            const nrows = tensor.ne[1];

            const min_chunk_size: i64 = 32 * 512;
            const chunk_size: i64 = if (n_per_row >= min_chunk_size) n_per_row else n_per_row * @divTrunc(min_chunk_size + n_per_row - 1, n_per_row);

            const nelements_matrix = tensor.ne[0] * tensor.ne[1];
            const nchunk = @divTrunc(nelements_matrix + chunk_size - 1, chunk_size);
            const nthread_use: i64 = if (nthread > 1) @max(1, @min(@as(i64, nthread), nchunk)) else 1;

            // Each expert separately: each has its own importance matrix.
            new_size = 0;
            var j03: i64 = 0;
            while (j03 < tensor.ne[2]) : (j03 += 1) {
                const f32_data_03 = f32_data + @as(usize, @intCast(j03 * nelements_matrix));
                const new_data_03 = new_data + c.ggml_row_size(new_type, n_per_row) * @as(usize, @intCast(j03)) * @as(usize, @intCast(nrows));
                const imatrix_03: ?[*]const f32 = if (imatrix) |im| im + @as(usize, @intCast(j03 * n_per_row)) else null;

                new_size += try quantizeRows(new_type, f32_data_03, new_data_03, chunk_size, nrows, n_per_row, imatrix_03, nthread_use, exc);
            }
            impl.logInfo("size = %8.2f MiB -> %8.2f MiB\n", .{ mib(tensor_size), mib(new_size) });
        }
        total_size_org += tensor_size;
        total_size_new += new_size;

        // The metadata, as we go.
        const ctx = ctx_outs[@intCast(cur_split)];
        c.gguf_set_tensor_type(ctx, tm.name.ptr, new_type);
        impl.assert(c.gguf_get_tensor_size(ctx, c.gguf_find_tensor(ctx, tm.name.ptr)) == new_size, "llama-quant.cpp", 1275, "gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), metadata[i].name.c_str())) == new_size");
        c.gguf_set_tensor_data(ctx, tm.name.ptr, new_data);

        // The data and its padding.
        fout.write(new_data, new_size);
        fout.zeros(std.mem.alignForward(usize, new_size, alignment) - new_size);
    }

    if (!params.dry_run) Files.close(&fout, ctx_outs, cur_split);

    const n_elements: f64 = @floatFromInt(ml.n_elements);
    impl.logInfo("%s: model size  = %8.2f MiB (%.2f BPW)\n", .{ "llama_model_quantize_impl", mib(total_size_org), @as(f64, @floatFromInt(total_size_org)) * 8.0 / n_elements });
    impl.logInfo("%s: quant size  = %8.2f MiB (%.2f BPW)\n", .{ "llama_model_quantize_impl", mib(total_size_new), @as(f64, @floatFromInt(total_size_new)) * 8.0 / n_elements });

    if (params.imatrix == null and params.dry_run and will_require_imatrix) {
        impl.logWarn("%s: WARNING: dry run completed successfully, but actually completing this quantization will require an imatrix!\n", .{"llama_model_quantize_impl"});
    }

    if (qs.n_fallback > 0) {
        impl.logWarn("%s: WARNING: %d of %d tensor(s) required fallback quantization\n", .{ "llama_model_quantize_impl", qs.n_fallback, ml.n_tensors });
    }
}

/// Ports `llama_model_quantize_default_params` (llama-quant.cpp:1311 @c1d0e7a00).
///
/// Return: the default parameters.
pub fn defaultParams() c.llama_model_quantize_params {
    return .{
        .nthread = 0,
        .ftype = c.LLAMA_FTYPE_MOSTLY_Q8_0,
        .output_tensor_type = c.GGML_TYPE_COUNT,
        .token_embedding_type = c.GGML_TYPE_COUNT,
        .allow_requantize = false,
        .quantize_output_tensor = true,
        .only_copy = false,
        .pure = false,
        .keep_split = false,
        .dry_run = false,
        .imatrix = null,
        .kv_overrides = null,
        .tt_overrides = null,
        .prune_layers = null,
    };
}

// -----------------------------------------------------------------------------
// The type-selection helpers of llama-ext.h

/// Mirrors `llama_quant_model_desc` (llama-ext.h:31 @c1d0e7a00).
pub const ModelDesc = extern struct {
    architecture: [*:0]const u8,
    n_embd: u32,
    n_ff: u32,
    n_layer: u32,
    n_head: u32,
    n_head_kv: u32,
    n_expert: u32,
    n_embd_head_k: u32,
    n_embd_head_v: u32,
};

/// Ports `llama_quant_init` (llama-quant.cpp:1350 @c1d0e7a00).
///
/// Parameters:
/// - `model`: the model; borrowed for the state's life.
/// - `params`: the parameters; borrowed for the state's life.
/// - `exc`: receives a pattern's `regex_error`.
///
/// Return: the state, owned by the caller; free it with `quantFree`.
pub fn quantInit(model: *const Model, params: *const c.llama_model_quantize_params, exc: *Exception) Thrown!*State {
    const qs = allocator.create(State) catch oom();
    errdefer allocator.destroy(qs);
    qs.* = try State.init(model, params, exc);
    return qs;
}

/// Ports `llama_quant_free` (llama-quant.cpp:1356 @c1d0e7a00).
///
/// Return: nothing.
pub fn quantFree(qs: *State) void {
    qs.deinit();
    allocator.destroy(qs);
}

/// Ports `llama_quant_model_from_metadata` (llama-quant.cpp:1360 @c1d0e7a00):
/// a model with only what type selection reads.
///
/// Parameters:
/// - `desc`: the descriptor.
/// - `exc`: receives what `llama_model_create` throws.
///
/// Return: the model, owned by the caller; free it with `deinit` and
/// `allocator.destroy`.
pub fn modelFromMetadata(desc: *const ModelDesc, exc: *Exception) Thrown!*Model {
    const a = arch.archFromString(std.mem.span(desc.architecture));
    const m = allocator.create(Model) catch oom();
    errdefer allocator.destroy(m);
    m.* = try Model.createArch(a, model_mod.defaultParams(), exc);
    m.arch = a;

    // Only LLM_TYPE_70B matters to the rules.
    if (m.arch == .llama and desc.n_layer == 80 and desc.n_head != desc.n_head_kv) m.type = .@"70b";

    const hp = &m.hparams;
    hp.n_embd = desc.n_embd;
    hp.n_embd_head_k_full = desc.n_embd_head_k;
    hp.n_embd_head_v_full = desc.n_embd_head_v;
    hp.n_layer_all = desc.n_layer;
    impl.assert(desc.n_layer > 0 and desc.n_layer <= 512, "llama-quant.cpp", 1375, "desc->n_layer > 0 && desc->n_layer <= LLAMA_MAX_LAYERS");
    hp.n_expert = desc.n_expert;

    for (0..desc.n_layer) |i| {
        hp.n_head_arr[i] = desc.n_head;
        hp.n_head_kv_arr[i] = desc.n_head_kv;
        hp.n_ff_arr[i] = desc.n_ff;
    }
    return m;
}

/// Ports `llama_quant_tensor_allows_quantization` (llama-quant.cpp:1387
/// @c1d0e7a00).
///
/// Return: whether `tensor` may be quantized under `qs`'s parameters.
pub fn quantTensorAllowsQuantization(qs: *const State, tensor: *const Tensor) bool {
    return tensorAllowsQuantization(qs.params, qs.model.arch, tensor);
}

/// Ports `llama_quant_compute_types` (llama-quant.cpp:1393 @c1d0e7a00): the
/// type each tensor would get, with the state reset first.
///
/// Parameters:
/// - `qs`: the state; its counters are reset and advanced.
/// - `ftype`: the file type, in place of the parameters'.
/// - `tensors`: the tensors, in file order.
/// - `result_types`: receives one type per tensor.
/// - `exc`: receives what type selection throws.
///
/// Return: nothing, or `error.Thrown`.
pub fn quantComputeTypes(qs: *State, ftype: Ftype, tensors: []const *Tensor, result_types: []GgmlType, exc: *Exception) Thrown!void {
    qs.n_attention_wv = 0;
    qs.n_ffn_down = 0;
    qs.n_ffn_gate = 0;
    qs.n_ffn_up = 0;
    qs.i_attention_wv = 0;
    qs.i_ffn_down = 0;
    qs.i_ffn_gate = 0;
    qs.i_ffn_up = 0;
    qs.n_fallback = 0;
    qs.has_imatrix = false;
    qs.has_tied_embeddings = true;

    const names = allocator.alloc([]const u8, tensors.len) catch oom();
    defer allocator.free(names);
    const categories = allocator.alloc(Category, tensors.len) catch oom();
    defer allocator.free(categories);
    for (tensors, names) |t, *n| n.* = nameOf(t);

    initCounters(qs, names, categories);

    // The caller's parameters with the requested file type.
    var local_params = qs.params.*;
    local_params.ftype = ftype;

    const default_type = ftypeGetDefaultType(ftype);

    for (tensors, categories, result_types) |t, cat, *r| {
        r.* = try getType(qs, &local_params, t, default_type, cat, exc);
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(State);
}

test "categories follow the C++'s precedence" {
    try std.testing.expectEqual(Category.output, getCategory("output.weight"));
    try std.testing.expectEqual(Category.token_embd, getCategory("per_layer_token_embd.weight"));
    try std.testing.expectEqual(Category.attention_qkv, getCategory("blk.0.attn_qkv.weight"));
    try std.testing.expectEqual(Category.attention_v, getCategory("blk.0.attn_v.weight"));
    try std.testing.expectEqual(Category.ffn_up, getCategory("blk.0.ffn_up_exps.weight"));
    try std.testing.expectEqual(Category.ffn_gate, getCategory("blk.0.ffn_gate_inp.weight"));
    try std.testing.expectEqual(Category.other, getCategory("blk.0.attn_norm.weight"));
}

test "use_more_bits truncates as C does" {
    // (0 - 32/8) % 3 is -1 in C, never 2.
    try std.testing.expect(useMoreBits(0, 32));
    try std.testing.expect(!useMoreBits(5, 32));
    try std.testing.expect(useMoreBits(6, 32));
    try std.testing.expect(useMoreBits(28, 32));
}

test "default types" {
    try std.testing.expectEqual(@as(GgmlType, c.GGML_TYPE_IQ2_XS), ftypeGetDefaultType(c.LLAMA_FTYPE_MOSTLY_IQ2_S));
    try std.testing.expectEqual(@as(GgmlType, c.GGML_TYPE_COUNT), ftypeGetDefaultType(c.LLAMA_FTYPE_MOSTLY_NVFP4));
}
