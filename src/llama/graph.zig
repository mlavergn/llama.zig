//! Building a model's compute graph for one ubatch, and feeding it.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-graph.h` -- the graph enums, `llama_cross`,
//!   `llm_graph_input_i` and the inputs below, `llm_graph_params`,
//!   `llm_graph_result` and `llm_graph_context`
//! - `llama.cpp/src/llama-graph.cpp` -- their bodies, for the inputs and
//!   builders the ported architectures reach, and
//!   `llama_relative_position_bucket`
//! - `llama.cpp/src/llama-impl.h` -- `llama_mul_mat_hadamard`
//!
//! All at v0.4.0 (`5266f24da`).
//!
//! # Scope
//!
//! The C++ has an input class and a `build_attn` overload for every kind of
//! memory; here are the ones a KV cache, a recurrent memory and their
//! hybrid need -- `embd`, `embd_h`, `pos`, `out_ids`, `mean`, `cls`, `rs`,
//! `attn_kv`, `mem_hybrid` and `sampling` -- with the shared builders over
//! them. The SWA, DSA, MSA and DeepSeek-V4 variants come with the
//! architectures that use them.
//!
//! # Inputs are an interface
//!
//! `llm_graph_input_i` is an abstract base the result owns a list of. Here
//! an input is a pointer and a table of three functions; `Input.of` builds
//! the table from a type's own `setInput`, `canReuse` and `destroy`.

const std = @import("std");
const impl = @import("impl.zig");
const arch_mod = @import("arch.zig");
const hparams_mod = @import("hparams.zig");
const cparams_mod = @import("cparams.zig");
const batch = @import("batch.zig");
const memory = @import("memory.zig");
const adapter = @import("adapter.zig");
const kv_cache = @import("kv_cache.zig");
const kv_cache_iswa = @import("kv_cache_iswa.zig");
const kv_cache_dsv4 = @import("kv_cache_dsv4.zig");
const kv_cache_msa = @import("kv_cache_msa.zig");
const kv_cache_dsa = @import("kv_cache_dsa.zig");
const kv_cache_dsa_iswa = @import("kv_cache_dsa_iswa.zig");
const memory_recurrent = @import("memory_recurrent.zig");
const memory_hybrid = @import("memory_hybrid.zig");
const memory_hybrid_iswa = @import("memory_hybrid_iswa.zig");
const model_mod = @import("model.zig");

const c = impl.c;
const allocator = impl.allocator;
const Hparams = hparams_mod.Hparams;
const Cparams = cparams_mod.Cparams;
const Ubatch = batch.Ubatch;
const Tensor = c.ggml_tensor;
const Arch = arch_mod.Arch;

fn oom() noreturn {
    impl.abort("llama-graph.cpp", 0, "out of memory");
}

/// `LLAMA_MAX_LAYERS` (llama-hparams.h:11 @5266f24da).
const max_layers = hparams_mod.max_layers;

/// Mirrors `llm_graph_type` (llama-graph.h:36 @5266f24da).
pub const GraphType = enum(c_int) {
    default,
    encoder,
    decoder,
    decoder_mtp,
};

/// Mirrors `llm_fused_op` (llama-graph.h:43 @5266f24da).
pub const FusedOp = enum(c_int) {
    flash_attn,
    gdn_ar,
    gdn_ch,
    lightning_indexer,
    dsv4_hc_pre,
    dsv4_hc_comb,
    dsv4_hc_post,
};

/// Mirrors `llm_ffn_op_type` (llama-graph.h:53 @5266f24da).
pub const FfnOp = enum(c_int) {
    none = 0,
    silu,
    gelu,
    relu,
    relu_sqr,
    swiglu,
    geglu,
    reglu,
    swiglu_oai_moe,
    situ,
};

/// Mirrors `llm_ffn_gate_type` (llama-graph.h:66 @5266f24da).
pub const FfnGate = enum(c_int) {
    seq,
    par,
};

/// Mirrors `llm_norm_type` (llama-graph.h:71 @5266f24da).
pub const NormType = enum(c_int) {
    norm,
    rms,
    group,
};

/// Mirrors `llama_cross` (llama-graph.h:78 @5266f24da): the encoder's
/// output, for a decoder's cross-attention.
pub const Cross = struct {
    n_embd: i64 = 0,
    n_enc: i64 = 0,
    /// The encoder's output embeddings, copied to host memory -- the C++'s
    /// TODO: this needs more work to be correct
    /// (https://github.com/ggml-org/llama.cpp/pull/11213#discussion_r1969892524).
    v_embd: std.ArrayList(f32) = .empty,
    /// Per encoder token, the sequences it belongs to, which the decoder's
    /// cross-attention mask is built from. The C++'s `std::set`, kept
    /// sorted; only membership is ever asked.
    seq_ids_enc: std.ArrayList(std.ArrayList(i32)) = .empty,

    pub fn deinit(self: *Cross) void {
        self.v_embd.deinit(allocator);
        for (self.seq_ids_enc.items) |*s| s.deinit(allocator);
        self.seq_ids_enc.deinit(allocator);
    }

    /// Whether encoder token `j` belongs to `seq_id`.
    pub fn hasSeq(self: *const Cross, j: usize, seq_id: i32) bool {
        return std.sort.binarySearch(i32, self.seq_ids_enc.items[j].items, seq_id, struct {
            fn order(ctx: i32, x: i32) std.math.Order {
                return std.math.order(ctx, x);
            }
        }.order) != null;
    }

    /// `seq_ids_enc[j].insert(seq_id)`.
    pub fn insertSeq(self: *Cross, j: usize, seq_id: i32) void {
        const set = &self.seq_ids_enc.items[j];
        var at: usize = 0;
        while (at < set.items.len and set.items[at] < seq_id) at += 1;
        if (at < set.items.len and set.items[at] == seq_id) return;
        set.insert(allocator, at, seq_id) catch oom();
    }
};

// -----------------------------------------------------------------------------
// Inputs

/// Mirrors `llm_graph_input_i` (llama-graph.h:100 @5266f24da).
pub const Input = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        setInput: *const fn (ptr: *anyopaque, ubatch: ?*const Ubatch) void,
        canReuse: *const fn (ptr: *anyopaque, params: *const Params) bool,
        destroy: *const fn (ptr: *anyopaque) void,
    };

    /// The interface over a heap-allocated `T`. `T` provides
    /// `setInput(*T, ?*const Ubatch)` and may provide
    /// `canReuse(*T, *const Params) bool`; without it, the base's answer
    /// is false.
    pub fn of(comptime T: type, p: *T) Input {
        const Gen = struct {
            fn setInput(ptr: *anyopaque, ubatch: ?*const Ubatch) void {
                T.setInput(@ptrCast(@alignCast(ptr)), ubatch);
            }
            fn canReuse(ptr: *anyopaque, params: *const Params) bool {
                if (@hasDecl(T, "canReuse")) return T.canReuse(@ptrCast(@alignCast(ptr)), params);
                return false;
            }
            fn destroy(ptr: *anyopaque) void {
                const self: *T = @ptrCast(@alignCast(ptr));
                if (@hasDecl(T, "deinit")) self.deinit();
                allocator.destroy(self);
            }
            const vtable: VTable = .{ .setInput = setInput, .canReuse = canReuse, .destroy = destroy };
        };
        return .{ .ptr = p, .vtable = &Gen.vtable };
    }
};

/// `LLAMA_GRAPH_INPUT_DEBUG`, read in `llm_graph_input_i`'s constructor
/// (llama-graph.h:100 @5266f24da).
fn inputDebug() c_int {
    const v = std.c.getenv("LLAMA_GRAPH_INPUT_DEBUG") orelse return 0;
    return impl.atoi(v);
}

/// Ports `llm_graph_input_embd` (llama-graph.h:126 @5266f24da).
pub const InpEmbd = struct {
    /// I32 `[n_batch]`.
    tokens: ?*Tensor = null,
    /// F32 `[n_embd, n_batch]`.
    embd: ?*Tensor = null,
    n_embd: i64,

    /// Ports `llm_graph_input_embd`'s `set_input` (llama-graph.cpp:69 @5266f24da).
    pub fn setInput(self: *InpEmbd, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        if (ubatch.token != null) {
            const n_tokens: usize = ubatch.n_tokens;
            c.ggml_backend_tensor_set(self.tokens, ubatch.token, 0, n_tokens * c.ggml_element_size(self.tokens));
        }
        if (ubatch.embd != null) {
            impl.assert(self.n_embd == self.embd.?.ne[0], "llama-graph.cpp", 77, "n_embd == embd->ne[0]");
            const n_tokens: usize = ubatch.n_tokens;
            c.ggml_backend_tensor_set(self.embd, ubatch.embd, 0, n_tokens * @as(usize, @intCast(self.n_embd)) * c.ggml_element_size(self.embd));
        }
    }

    /// Ports `llm_graph_input_embd`'s `can_reuse` (llama-graph.cpp:85 @5266f24da).
    pub fn canReuse(self: *InpEmbd, params: *const Params) bool {
        var res = true;
        res = res and ((params.ubatch.token == null) or (self.tokens != null and self.tokens.?.ne[0] == params.ubatch.n_tokens));
        res = res and ((params.ubatch.embd == null) or (self.embd != null and self.embd.?.ne[1] == params.ubatch.n_tokens));
        return res;
    }
};

/// Ports `llm_graph_input_embd_h` (llama-graph.h:142 @5266f24da): the MTP
/// head's token embeddings and hidden states.
pub const InpEmbdH = struct {
    tokens: ?*Tensor = null,
    embd: ?*Tensor = null,
    h: ?*Tensor = null,
    n_embd: i64,

    /// Ports `llm_graph_input_embd_h`'s `set_input` (llama-graph.cpp:94 @5266f24da).
    pub fn setInput(self: *InpEmbdH, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        const n_tokens: usize = ubatch.n_tokens;
        const n_embd: usize = @intCast(self.n_embd);
        if (ubatch.token != null) {
            c.ggml_backend_tensor_set(self.tokens, ubatch.token, 0, n_tokens * c.ggml_element_size(self.tokens));
        } else {
            // Multimodal embeddings come through here.
            impl.assert(ubatch.embd != null, "llama-graph.cpp", 101, "ubatch->embd");
            impl.assert(self.n_embd == self.embd.?.ne[0], "llama-graph.cpp", 102, "n_embd == embd->ne[0]");
            c.ggml_backend_tensor_set(self.embd, ubatch.embd, 0, n_tokens * n_embd * c.ggml_element_size(self.h));
        }
        // The hidden state is always provided as an embedding.
        if (ubatch.embd != null) {
            impl.assert(self.n_embd == self.h.?.ne[0], "llama-graph.cpp", 111, "n_embd == h->ne[0]");
            c.ggml_backend_tensor_set(self.h, ubatch.embd, 0, n_tokens * n_embd * c.ggml_element_size(self.h));
        }
    }

    /// Ports `llm_graph_input_embd_h`'s `can_reuse` (llama-graph.cpp:117 @5266f24da).
    pub fn canReuse(self: *InpEmbdH, params: *const Params) bool {
        var res = true;
        res = res and ((params.ubatch.token == null) or (self.tokens != null and self.tokens.?.ne[0] == params.ubatch.n_tokens));
        res = res and ((params.ubatch.embd == null) or (self.embd != null and self.embd.?.ne[1] == params.ubatch.n_tokens));
        res = res and ((params.ubatch.embd == null) or (self.h != null and self.h.?.ne[1] == params.ubatch.n_tokens));
        return res;
    }
};

/// Ports `llm_graph_input_pos` (llama-graph.h:158 @5266f24da).
pub const InpPos = struct {
    /// I32 `[n_batch]`.
    pos: ?*Tensor = null,
    n_pos_per_embd: u32 = 1,

    /// Ports `llm_graph_input_pos`'s `set_input` (llama-graph.cpp:127 @5266f24da).
    pub fn setInput(self: *InpPos, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        if (ubatch.pos == null or self.pos == null) return;
        const n_tokens: usize = ubatch.n_tokens;
        if (ubatch.token != null and self.n_pos_per_embd == 4) {
            // M-RoPE with text tokens: the first three dimensions are the
            // 1D positions, the fourth is zero.
            const pos_data = allocator.alloc(i32, n_tokens * self.n_pos_per_embd) catch oom();
            defer allocator.free(pos_data);
            for (0..n_tokens) |i| {
                pos_data[i] = ubatch.pos[i];
                pos_data[n_tokens + i] = ubatch.pos[i];
                pos_data[2 * n_tokens + i] = ubatch.pos[i];
                pos_data[3 * n_tokens + i] = 0;
            }
            c.ggml_backend_tensor_set(self.pos, pos_data.ptr, 0, pos_data.len * c.ggml_element_size(self.pos));
        } else {
            c.ggml_backend_tensor_set(self.pos, ubatch.pos, 0, n_tokens * self.n_pos_per_embd * c.ggml_element_size(self.pos));
        }
    }

    /// Ports `llm_graph_input_pos`'s `can_reuse` (llama-graph.cpp:149 @5266f24da).
    pub fn canReuse(self: *InpPos, params: *const Params) bool {
        return self.pos.?.ne[0] == @as(i64, params.ubatch.n_tokens) * self.n_pos_per_embd;
    }
};

/// Ports `llm_graph_input_attn_temp` (llama-graph.h:173 @5266f24da):
/// Llama 4's attention temperature tuning, a per-token scale on Q.
pub const InpAttnTemp = struct {
    /// F32 `[1, 1, n_batch]`.
    attn_scale: ?*Tensor = null,
    n_attn_temp_floor_scale: u32,
    f_attn_temp_scale: f32,
    f_attn_temp_offset: f32,

    /// Ports `llm_graph_input_attn_temp`'s `set_input` (llama-graph.cpp:157
    /// @5266f24da). The C++ takes the logarithm and what follows it in
    /// double, and its compiler may contract the multiply-add there; no
    /// fixture sets a temperature scale, so nothing yet says which.
    pub fn setInput(self: *InpAttnTemp, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        if (ubatch.pos == null or self.attn_scale == null) return;
        const n_tokens: usize = ubatch.n_tokens;

        impl.assert(self.f_attn_temp_scale != 0.0, "llama-graph.cpp", 161, "f_attn_temp_scale != 0.0f");
        impl.assert(self.n_attn_temp_floor_scale != 0, "llama-graph.cpp", 162, "n_attn_temp_floor_scale != 0");

        const attn_scale_data = allocator.alloc(f32, n_tokens) catch oom();
        defer allocator.free(attn_scale_data);
        for (attn_scale_data, 0..) |*d, i| {
            const pos: f32 = @floatFromInt(ubatch.pos[i]);
            const floored: f32 = @floor((pos + self.f_attn_temp_offset) / @as(f32, @floatFromInt(self.n_attn_temp_floor_scale)));
            d.* = @floatCast(cLog(@as(f64, floored) + 1.0) * @as(f64, self.f_attn_temp_scale) + 1.0);
        }

        c.ggml_backend_tensor_set(self.attn_scale, attn_scale_data.ptr, 0, n_tokens * c.ggml_element_size(self.attn_scale));
    }
};

/// `std::log` on a double: libm's `log`, as the C++ calls it.
const cLog = @extern(*const fn (f64) callconv(.c) f64, .{ .name = "log" });

/// Ports `llm_graph_input_out_ids` (llama-graph.h:216 @5266f24da).
pub const InpOutIds = struct {
    /// I32 `[n_outputs]`.
    out_ids: ?*Tensor = null,
    n_outputs: u32,

    /// Ports `llm_graph_input_out_ids`'s `set_input` (llama-graph.cpp:199 @5266f24da).
    pub fn setInput(self: *InpOutIds, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        impl.assert(self.out_ids != null, "llama-graph.cpp", 200, "out_ids");
        const n_tokens: u32 = ubatch.n_tokens;
        impl.assert(c.ggml_backend_buffer_is_host(self.out_ids.?.buffer), "llama-graph.cpp", 204, "ggml_backend_buffer_is_host(out_ids->buffer)");
        const data: [*]i32 = @ptrCast(@alignCast(self.out_ids.?.data));
        if (self.n_outputs == n_tokens) {
            for (0..n_tokens) |i| data[i] = @intCast(i);
            return;
        }
        impl.assert(ubatch.output != null, "llama-graph.cpp", 215, "ubatch->output");
        var n_outputs: usize = 0;
        for (0..n_tokens) |i| {
            if (ubatch.output[i] != 0) {
                data[n_outputs] = @intCast(i);
                n_outputs += 1;
            }
        }
    }

    /// Ports `llm_graph_input_out_ids`'s `can_reuse` (llama-graph.cpp:226 @5266f24da).
    pub fn canReuse(self: *InpOutIds, params: *const Params) bool {
        return self.n_outputs == params.n_outputs;
    }
};

/// Ports `llm_graph_input_mean` (llama-graph.h:236 @5266f24da).
pub const InpMean = struct {
    /// F32 `[n_batch, n_batch]`.
    mean: ?*Tensor = null,
    cparams: Cparams,

    /// Ports `llm_graph_input_mean`'s `set_input` (llama-graph.cpp:234 @5266f24da).
    pub fn setInput(self: *InpMean, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        if (!(self.cparams.embeddings and (self.cparams.pooling_type == c.LLAMA_POOLING_TYPE_MEAN or
            self.cparams.pooling_type == c.LLAMA_POOLING_TYPE_RANK))) return;

        const n_tokens: usize = ubatch.n_tokens;
        const n_seq_tokens: usize = ubatch.n_seq_tokens;
        const n_seqs_unq: usize = ubatch.n_seqs_unq;

        impl.assert(self.mean != null, "llama-graph.cpp", 243, "mean");
        impl.assert(c.ggml_backend_buffer_is_host(self.mean.?.buffer), "llama-graph.cpp", 244, "ggml_backend_buffer_is_host(mean->buffer)");

        const data: [*]f32 = @ptrCast(@alignCast(self.mean.?.data));
        @memset(@as([*]u8, @ptrCast(self.mean.?.data))[0 .. n_tokens * n_seqs_unq * c.ggml_element_size(self.mean)], 0);

        const sums = allocator.alloc(u64, n_seqs_unq) catch oom();
        defer allocator.free(sums);
        @memset(sums, 0);
        var i: usize = 0;
        while (i < n_tokens) : (i += n_seq_tokens) {
            for (0..@intCast(ubatch.n_seq_id[i])) |s| {
                const seq_id = ubatch.seq_id[i][s];
                const seq_idx: usize = @intCast(ubatch.seq_idx[@intCast(seq_id)]);
                sums[seq_idx] += ubatch.n_seq_tokens;
            }
        }

        const div = allocator.alloc(f32, n_seqs_unq) catch oom();
        defer allocator.free(div);
        for (div, sums) |*d, sum| d.* = if (sum > 0) 1.0 / @as(f32, @floatFromInt(sum)) else 0.0;

        i = 0;
        while (i < n_tokens) : (i += n_seq_tokens) {
            for (0..@intCast(ubatch.n_seq_id[i])) |s| {
                const seq_id = ubatch.seq_id[i][s];
                const seq_idx: usize = @intCast(ubatch.seq_idx[@intCast(seq_id)]);
                for (0..n_seq_tokens) |j| data[seq_idx * n_tokens + i + j] = div[seq_idx];
            }
        }
    }
};

/// Ports `llm_graph_input_cls` (llama-graph.h:248 @5266f24da).
pub const InpCls = struct {
    /// I32 `[n_batch]`.
    cls: ?*Tensor = null,
    cparams: Cparams,
    arch: Arch,

    /// Ports `llm_graph_input_cls`'s `set_input` (llama-graph.cpp:280 @5266f24da).
    pub fn setInput(self: *InpCls, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        const n_tokens: usize = ubatch.n_tokens;
        const n_seqs_unq: usize = ubatch.n_seqs_unq;
        const pt = self.cparams.pooling_type;
        if (!(self.cparams.embeddings and (pt == c.LLAMA_POOLING_TYPE_CLS or pt == c.LLAMA_POOLING_TYPE_RANK or pt == c.LLAMA_POOLING_TYPE_LAST))) return;

        impl.assert(self.cls != null, "llama-graph.cpp", 289, "cls");
        impl.assert(c.ggml_backend_buffer_is_host(self.cls.?.buffer), "llama-graph.cpp", 290, "ggml_backend_buffer_is_host(cls->buffer)");

        const data: [*]u32 = @ptrCast(@alignCast(self.cls.?.data));
        @memset(@as([*]u8, @ptrCast(self.cls.?.data))[0 .. n_seqs_unq * c.ggml_element_size(self.cls)], 0);

        const target_pos = allocator.alloc(c_int, n_seqs_unq) catch oom();
        defer allocator.free(target_pos);
        @memset(target_pos, -1);
        const target_row = allocator.alloc(c_int, n_seqs_unq) catch oom();
        defer allocator.free(target_row);
        @memset(target_row, -1);

        const last = pt == c.LLAMA_POOLING_TYPE_LAST or
            (pt == c.LLAMA_POOLING_TYPE_RANK and (self.arch == .qwen3 or self.arch == .qwen3vl));

        for (0..n_tokens) |i| {
            const pos = ubatch.pos[i];
            for (0..@intCast(ubatch.n_seq_id[i])) |s| {
                const seq_id = ubatch.seq_id[i][s];
                const seq_idx: usize = @intCast(ubatch.seq_idx[@intCast(seq_id)]);
                if (target_pos[seq_idx] == -1 or (last and pos >= target_pos[seq_idx]) or (!last and pos < target_pos[seq_idx])) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = @intCast(i);
                }
            }
        }

        for (0..n_seqs_unq) |s| {
            if (target_row[s] >= 0) data[s] = @intCast(target_row[s]);
        }
    }
};

/// Ports `llm_graph_input_rs` (llama-graph.h:261 @5266f24da): which
/// recurrent states each sequence copies from.
pub const InpRs = struct {
    /// I32 `[n_rs]`.
    s_copy: ?*Tensor = null,
    /// I32 `[n_seqs]`.
    s_copy_main: ?*Tensor = null,
    /// I32 `[n_rs - n_seqs]`.
    s_copy_extra: ?*Tensor = null,
    mctx: *const memory_recurrent.RecurrentContext,
    head: u32 = 0,
    rs_z: i32 = 0,

    /// Ports `llm_graph_input_rs`'s `set_input` (llama-graph.cpp:329 @5266f24da).
    pub fn setInput(self: *InpRs, _: ?*const Ubatch) void {
        const n_rs = self.mctx.getNRs();
        if (self.s_copy) |s_copy| {
            impl.assert(c.ggml_backend_buffer_is_host(s_copy.buffer), "llama-graph.cpp", 335, "ggml_backend_buffer_is_host(s_copy->buffer)");
            const data: [*]i32 = @ptrCast(@alignCast(s_copy.data));
            // Copy destinations are always between head and head + n.
            for (0..n_rs) |i| data[i] = self.mctx.sCopy(@intCast(i));
        }
    }

    /// Ports `llm_graph_input_rs`'s `can_reuse` (llama-graph.cpp:345 @5266f24da).
    pub fn canReuse(self: *InpRs, params: *const Params) bool {
        const mctx = params.mctx.?.cast(memory_recurrent.RecurrentContext);
        self.mctx = mctx;
        var res = true;
        res = res and self.s_copy.?.ne[0] == mctx.getNRs();
        res = res and self.s_copy_main.?.ne[0] == params.ubatch.n_seqs;
        res = res and self.s_copy_extra.?.ne[0] == @as(i64, mctx.getNRs()) - params.ubatch.n_seqs;
        res = res and self.head == mctx.getHead();
        res = res and self.rs_z == mctx.getRsZ();
        return res;
    }
};

/// Ports `llm_graph_input_pos_bucket` (llama-graph.h:188 @5266f24da): the
/// relative-position buckets of an encoder's tokens, among themselves.
pub const InpPosBucket = struct {
    /// I32 `[n_batch, n_batch]`.
    pos_bucket: ?*Tensor = null,
    hparams: *const Hparams,

    /// Ports `llm_graph_input_pos_bucket`'s `set_input` (llama-graph.cpp:176 @5266f24da).
    pub fn setInput(self: *InpPosBucket, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        const pb = self.pos_bucket orelse return;
        const n_tokens: usize = ubatch.n_tokens;

        impl.assert(c.ggml_backend_buffer_is_host(pb.buffer), "llama-graph.cpp", 180, "ggml_backend_buffer_is_host(pos_bucket->buffer)");
        // The C++'s TODO: use ubatch->n_seqs instead of failing.
        impl.assert(!ubatch.equalSeqs(), "llama-graph.cpp", 181, "!ubatch->equal_seqs()");

        const data: [*]i32 = @ptrCast(@alignCast(pb.data));

        for (0..n_tokens) |j| {
            for (0..n_tokens) |i| {
                data[j * n_tokens + i] = relativePositionBucket(ubatch.pos[i], ubatch.pos[j], self.hparams.n_rel_attn_bkts, true);
            }
        }
    }
};

/// Ports `llm_graph_input_pos_bucket_kv` (llama-graph.h:200 @5266f24da): the
/// relative-position buckets of a decoder's tokens against its KV cache.
pub const InpPosBucketKv = struct {
    /// I32 `[n_kv, n_batch]`.
    pos_bucket: ?*Tensor = null,
    hparams: *const Hparams,
    mctx: *const kv_cache.KvCacheContext,

    /// Ports `llm_graph_input_pos_bucket_kv`'s `set_input` (llama-graph.cpp:193 @5266f24da).
    pub fn setInput(self: *InpPosBucketKv, ubatch: ?*const Ubatch) void {
        if (self.pos_bucket) |pb| self.mctx.setInputPosBucket(pb, ubatch.?);
    }
};

/// Ports `llm_graph_input_cross_embd` (llama-graph.h:284 @5266f24da): the
/// encoder's output, as the decoder's cross-attention reads it.
pub const InpCrossEmbd = struct {
    /// F32 `[n_embd, n_outputs_enc]`.
    cross_embd: ?*Tensor = null,
    cross: *const Cross,

    /// Ports `llm_graph_input_cross_embd`'s `set_input` (llama-graph.cpp:363 @5266f24da).
    pub fn setInput(self: *InpCrossEmbd, _: ?*const Ubatch) void {
        const ce = self.cross_embd orelse return;
        if (self.cross.v_embd.items.len == 0) return;
        std.debug.assert(ce.type == c.GGML_TYPE_F32);
        c.ggml_backend_tensor_set(ce, self.cross.v_embd.items.ptr, 0, c.ggml_nbytes(ce));
    }
};

/// Ports `llm_graph_input_attn_cross` (llama-graph.h:645 @5266f24da): which
/// encoder tokens each decoder token may attend, by sequence.
pub const InpAttnCross = struct {
    /// F32/F16 `[n_outputs_enc, n_batch, 1, 1]`.
    cross_kq_mask: ?*Tensor = null,
    cross_kq_mask_cnv: ?*Tensor = null,
    cross: *const Cross,

    pub fn getKqMaskCross(self: *const InpAttnCross) ?*Tensor {
        return self.cross_kq_mask_cnv;
    }

    fn fillMask(self: *const InpAttnCross, comptime T: type, data: [*]T, n_enc: usize, ubatch: *const Ubatch) void {
        const n_tokens: usize = ubatch.n_tokens;
        for (0..n_tokens) |i| {
            impl.assert(self.cross.seq_ids_enc.items.len != 0, "llama-graph.cpp", 1062, "!cross->seq_ids_enc.empty() && \"llama_encode must be called first\"");
            for (0..n_enc) |j| {
                var f: f32 = -std.math.inf(f32);

                var s: usize = 0;
                while (s < ubatch.n_seq_id[i]) : (s += 1) {
                    const seq_id = ubatch.seq_id[i][s];
                    if (self.cross.hasSeq(j, seq_id)) f = 0.0;
                }

                data[i * n_enc + j] = if (T == u16) c.ggml_fp32_to_fp16(f) else f;
            }
        }
    }

    /// Ports `llm_graph_input_attn_cross`'s `set_input` (llama-graph.cpp:1050 @5266f24da).
    pub fn setInput(self: *InpAttnCross, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        impl.assert(self.cross_kq_mask != null, "llama-graph.cpp", 1051, "cross_kq_mask");
        const m = self.cross_kq_mask.?;

        const n_enc: usize = @intCast(m.ne[0]);

        impl.assert(c.ggml_backend_buffer_is_host(m.buffer), "llama-graph.cpp", 1056, "ggml_backend_buffer_is_host(cross_kq_mask->buffer)");
        // The C++'s TODO: use ubatch->n_seqs instead of failing.
        impl.assert(!ubatch.equalSeqs(), "llama-graph.cpp", 1057, "!ubatch->equal_seqs()");

        if (m.type == c.GGML_TYPE_F16) {
            self.fillMask(u16, @ptrCast(@alignCast(m.data)), n_enc, ubatch);
        } else {
            self.fillMask(f32, @ptrCast(@alignCast(m.data)), n_enc, ubatch);
        }
    }
};

/// Ports `print_mask` (llama-graph.cpp:374 @5266f24da): the top-left 20 by
/// 20 corner of an attention mask, at debug level.
fn printMask(comptime T: type, data: [*]const T, n_tokens: i64, n_kv: i64, n_swa: i64, swa_type: hparams_mod.SwaType) void {
    const func = "print_mask";
    impl.logDebug("%s: === Attention mask ===\n", .{func});
    const swa_type_str: [*:0]const u8 = switch (swa_type) {
        .none => "LLAMA_SWA_TYPE_NONE",
        .standard => "LLAMA_SWA_TYPE_STANDARD",
        .chunked => "LLAMA_SWA_TYPE_CHUNKED",
        .symmetric => "LLAMA_SWA_TYPE_SYMMETRIC",
        else => "unknown",
    };

    impl.logDebug("%s: n_swa : %d, n_kv: %d, swa_type: %s\n", .{ func, @as(c_int, @truncate(n_swa)), @as(c_int, @truncate(n_kv)), swa_type_str });
    impl.logDebug("%s: '0' = can attend, '\u{221e}' = masked\n", .{func});
    impl.logDebug("%s: Rows = query tokens, Columns = key/value tokens\n\n", .{func});

    impl.logDebug("    ", .{});
    var j: c_int = 0;
    while (j < @min(20, n_kv)) : (j += 1) impl.logDebug("%2d", .{j});
    impl.logDebug("\n", .{});

    var i: c_int = 0;
    while (i < @min(20, n_tokens)) : (i += 1) {
        impl.logDebug(" %2d ", .{i});
        j = 0;
        while (j < @min(20, n_kv)) : (j += 1) {
            const raw = data[@intCast(@as(i64, i) * n_kv + j)];
            const val: f32 = if (T == u16) c.ggml_fp16_to_fp32(raw) else raw;
            if (val == -std.math.inf(f32)) {
                impl.logDebug(" \u{221e}", .{});
            } else {
                impl.logDebug(" 0", .{});
            }
        }
        impl.logDebug("\n", .{});
    }
}

/// Ports `llm_graph_input_attn_no_cache` (llama-graph.h:297 @5266f24da): the
/// masks of attention within the batch, for a model without a KV cache.
/// There, the number of KV values is the number of tokens in the batch.
pub const InpAttnNoCache = struct {
    /// F32/F16 `[n_tokens, n_batch/n_stream, 1, n_stream]`.
    self_kq_mask: ?*Tensor = null,
    self_kq_mask_cnv: ?*Tensor = null,
    /// F32/F16 `[n_tokens, n_batch/n_stream, 1, n_stream]`.
    self_kq_mask_swa: ?*Tensor = null,
    self_kq_mask_swa_cnv: ?*Tensor = null,
    hparams: *const Hparams,
    cparams: Cparams,
    /// `llm_graph_input_i`'s `debug`.
    debug: c_int,

    pub fn getKqMask(self: *const InpAttnNoCache) ?*Tensor {
        return self.self_kq_mask_cnv;
    }
    pub fn getKqMaskSwa(self: *const InpAttnNoCache) ?*Tensor {
        return self.self_kq_mask_swa_cnv;
    }

    /// The `fill_mask` lambda of `set_input`.
    fn fillMask(self: *const InpAttnNoCache, comptime T: type, data: [*]T, ne: usize, ubatch: *const Ubatch, n_swa: u32, swa_type: hparams_mod.SwaType) void {
        const n_tokens: i64 = ubatch.n_tokens;
        const n_kv: i64 = ubatch.n_tokens;
        const cast = struct {
            fn f(v: f32) T {
                return if (T == u16) c.ggml_fp32_to_fp16(v) else v;
            }
        }.f;

        @memset(data[0..ne], cast(-std.math.inf(f32)));

        var j1: usize = 0;
        while (j1 < n_tokens) : (j1 += 1) {
            const s1 = ubatch.seq_id[j1][0];
            const p1 = ubatch.pos[j1];

            const idst: u64 = j1 * @as(u64, @intCast(n_kv));

            var j0: usize = 0;
            while (j0 < n_tokens) : (j0 += 1) {
                const s0 = ubatch.seq_id[j0][0];
                const p0 = ubatch.pos[j0];

                // Mask different sequences.
                if (s0 != s1) continue;

                // Mask future tokens.
                if (self.cparams.causal_attn and p0 > p1) continue;

                // Apply SWA, if any.
                if (Hparams.isMaskedSwa(n_swa, swa_type, p0, p1)) continue;

                data[idst + j0] = cast(if (self.hparams.use_alibi) -@as(f32, @floatFromInt(@abs(p0 - p1))) else 0.0);
            }
        }

        if (self.debug != 0) printMask(T, data, n_tokens, n_kv, n_swa, swa_type);
    }

    fn fill(self: *const InpAttnNoCache, m: *Tensor, ubatch: *const Ubatch, n_swa: u32, swa_type: hparams_mod.SwaType) void {
        const ne: usize = @intCast(c.ggml_nelements(m));
        if (m.type == c.GGML_TYPE_F16) {
            self.fillMask(u16, @ptrCast(@alignCast(m.data)), ne, ubatch, n_swa, swa_type);
        } else {
            self.fillMask(f32, @ptrCast(@alignCast(m.data)), ne, ubatch, n_swa, swa_type);
        }
    }

    /// Ports `llm_graph_input_attn_no_cache`'s `set_input` (llama-graph.cpp:409 @5266f24da).
    pub fn setInput(self: *InpAttnNoCache, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        impl.assert(self.self_kq_mask != null, "llama-graph.cpp", 451, "self_kq_mask");
        const m = self.self_kq_mask.?;
        impl.assert(c.ggml_backend_buffer_is_host(m.buffer), "llama-graph.cpp", 452, "ggml_backend_buffer_is_host(self_kq_mask->buffer)");
        self.fill(m, ubatch, 0, .none);

        const swa_type: hparams_mod.SwaType = @fromBackingInt(@intCast(self.hparams.swa_type));
        if (swa_type != .none) {
            impl.assert(self.self_kq_mask_swa != null, "llama-graph.cpp", 460, "self_kq_mask_swa");
            const ms = self.self_kq_mask_swa.?;
            impl.assert(c.ggml_backend_buffer_is_host(ms.buffer), "llama-graph.cpp", 461, "ggml_backend_buffer_is_host(self_kq_mask_swa->buffer)");
            self.fill(ms, ubatch, self.hparams.n_swa, swa_type);
        }
    }
};

/// Ports `llm_graph_input_attn_kv` (llama-graph.h:320 @5266f24da).
pub const InpAttnKv = struct {
    /// I64 `[n_batch]`.
    self_k_idxs: ?*Tensor = null,
    /// I64 `[n_batch]` or `[n_batch*n_embd_v_gqa]`.
    self_v_idxs: ?*Tensor = null,
    /// F32/F16 `[n_kv, n_batch/n_stream, 1, n_stream]`.
    self_kq_mask: ?*Tensor = null,
    self_kq_mask_cnv: ?*Tensor = null,
    self_k_rot: ?*Tensor = null,
    self_v_rot: ?*Tensor = null,
    hparams: *const Hparams,
    cparams: Cparams,
    mctx: *const kv_cache.KvCacheContext,

    pub fn getKIdxs(self: *const InpAttnKv) *Tensor {
        return self.self_k_idxs.?;
    }
    pub fn getVIdxs(self: *const InpAttnKv) *Tensor {
        return self.self_v_idxs.?;
    }
    pub fn getKqMask(self: *const InpAttnKv) ?*Tensor {
        return self.self_kq_mask_cnv;
    }

    /// Ports `llm_graph_input_attn_kv`'s `set_input` (llama-graph.cpp:470 @5266f24da).
    pub fn setInput(self: *InpAttnKv, ubatch: ?*const Ubatch) void {
        self.mctx.setInputKIdxs(self.self_k_idxs.?, ubatch.?);
        self.mctx.setInputVIdxs(self.self_v_idxs.?, ubatch.?);
        // The mask is left unallocated when the graph only stores K/V.
        if (self.self_kq_mask) |m| {
            if (m.buffer != null) self.mctx.setInputKqMask(m, ubatch.?, self.cparams.causal_attn);
        }
        if (self.self_k_rot) |r| {
            if (r.buffer != null) self.mctx.setInputKRot(r);
        }
        if (self.self_v_rot) |r| {
            if (r.buffer != null) self.mctx.setInputVRot(r);
        }
    }

    /// Ports `llm_graph_input_attn_kv`'s `can_reuse` (llama-graph.cpp:489 @5266f24da).
    pub fn canReuse(self: *InpAttnKv, params: *const Params) bool {
        const mctx = params.mctx.?.cast(kv_cache.KvCacheContext);
        self.mctx = mctx;
        var res = true;
        res = res and self.self_k_idxs.?.ne[0] == params.ubatch.n_tokens;
        res = res and canReuseKqMask(self.self_kq_mask.?, mctx, &params.ubatch, &params.cparams);
        return res;
    }
};

/// Ports `llm_graph_input_attn_kv_msa` (llama-graph.h:464 @5266f24da): the
/// standard K/V attention input against an MSA cache's base cache, plus the
/// destination indices for its indexer key cache. The C++ derives from
/// `llm_graph_input_attn_kv`; here that is `base`, which `build_attn`
/// attends through.
pub const InpAttnKvMsa = struct {
    base: InpAttnKv,
    /// I64 `[n_batch]`, or null when MSA is off.
    self_k_idxs_idx: ?*Tensor = null,
    mctx_msa: *const kv_cache_msa.MsaContext,

    pub fn getKIdxsIdx(self: *const InpAttnKvMsa) *Tensor {
        return self.self_k_idxs_idx.?;
    }
    pub fn getKqMask(self: *const InpAttnKvMsa) ?*Tensor {
        return self.base.getKqMask();
    }

    /// Ports `llm_graph_input_attn_kv_msa`'s `set_input` (llama-graph.cpp:534 @5266f24da).
    pub fn setInput(self: *InpAttnKvMsa, ubatch: ?*const Ubatch) void {
        self.base.setInput(ubatch);

        if (self.self_k_idxs_idx) |t| self.mctx_msa.getIdx().setInputKIdxs(t, ubatch.?);
    }

    /// Ports `llm_graph_input_attn_kv_msa`'s `can_reuse` (llama-graph.cpp:542 @5266f24da).
    pub fn canReuse(self: *InpAttnKvMsa, params: *const Params) bool {
        self.mctx_msa = params.mctx.?.cast(kv_cache_msa.MsaContext);

        // The parent operates on the base cache's context.
        self.base.mctx = self.mctx_msa.getBase();

        var res = true;

        res = res and self.base.self_k_idxs.?.ne[0] == params.ubatch.n_tokens;
        if (self.self_k_idxs_idx) |t| res = res and t.ne[0] == params.ubatch.n_tokens;

        res = canReuseKqMask(self.base.self_kq_mask.?, self.base.mctx, &params.ubatch, &params.cparams) and res;

        return res;
    }
};

/// Ports `llm_graph_input_attn_k` (llama-graph.h:362 @5266f24da): a cache
/// that stores K alone, MLA's compressed latent, from which V is a view.
pub const InpAttnK = struct {
    /// I64 `[n_batch]`.
    self_k_idxs: ?*Tensor = null,
    /// F32/F16 `[n_kv, n_batch/n_stream, 1, n_stream]`.
    self_kq_mask: ?*Tensor = null,
    self_kq_mask_cnv: ?*Tensor = null,
    hparams: *const Hparams,
    cparams: Cparams,
    mctx: *const kv_cache.KvCacheContext,

    pub fn getKIdxs(self: *const InpAttnK) *Tensor {
        return self.self_k_idxs.?;
    }
    pub fn getKqMask(self: *const InpAttnK) ?*Tensor {
        return self.self_kq_mask_cnv;
    }

    /// Ports `llm_graph_input_attn_k`'s `set_input` (llama-graph.cpp:504 @5266f24da).
    pub fn setInput(self: *InpAttnK, ubatch: ?*const Ubatch) void {
        self.mctx.setInputKIdxs(self.self_k_idxs.?, ubatch.?);
        self.mctx.setInputKqMask(self.self_kq_mask.?, ubatch.?, self.cparams.causal_attn);
    }

    /// Ports `llm_graph_input_attn_k`'s `can_reuse` (llama-graph.cpp:510 @5266f24da).
    pub fn canReuse(self: *InpAttnK, params: *const Params) bool {
        self.mctx = params.mctx.?.cast(kv_cache.KvCacheContext);
        return self.canReuseImpl(params);
    }

    /// Ports `llm_graph_input_attn_k`'s `can_reuse_impl` (llama-graph.cpp:516
    /// @5266f24da): `can_reuse` without re-binding the memory context.
    pub fn canReuseImpl(self: *const InpAttnK, params: *const Params) bool {
        var res = true;
        res = res and self.self_k_idxs.?.ne[0] == params.ubatch.n_tokens;
        res = canReuseKqMask(self.self_kq_mask.?, self.mctx, &params.ubatch, &params.cparams) and res;
        return res;
    }
};

/// Ports `llm_graph_input_attn_k_dsa` (llama-graph.h:396 @5266f24da): a
/// K-only MLA cache, as `llm_graph_input_attn_k`, beside the lightning
/// indexer's key cache, with the indexer's Hadamard rotation.
pub const InpAttnKDsa = struct {
    /// I64 `[n_batch]`.
    self_k_idxs_mla: ?*Tensor = null,
    /// I64 `[n_batch]`.
    self_k_idxs_lid: ?*Tensor = null,
    /// F32/F16 `[n_kv, n_batch/n_stream, 1, n_stream]`.
    self_kq_mask_mla: ?*Tensor = null,
    self_kq_mask_mla_cnv: ?*Tensor = null,
    /// F32 `[n_kv, n_batch/n_stream, 1, n_stream]`.
    self_kq_mask_lid: ?*Tensor = null,
    self_kq_mask_lid_cnv: ?*Tensor = null,
    self_k_rot_lid: ?*Tensor = null,
    hparams: *const Hparams,
    cparams: Cparams,
    mctx: *const kv_cache_dsa.DsaContext,

    pub fn getKIdxsMla(self: *const InpAttnKDsa) *Tensor {
        return self.self_k_idxs_mla.?;
    }
    pub fn getKIdxsLid(self: *const InpAttnKDsa) *Tensor {
        return self.self_k_idxs_lid.?;
    }
    pub fn getKqMaskMla(self: *const InpAttnKDsa) ?*Tensor {
        return self.self_kq_mask_mla_cnv;
    }
    pub fn getKqMaskLid(self: *const InpAttnKDsa) ?*Tensor {
        return self.self_kq_mask_lid;
    }

    /// Ports `llm_graph_input_attn_k_dsa`'s `set_input` (llama-graph.cpp:560 @5266f24da).
    pub fn setInput(self: *InpAttnKDsa, ubatch: ?*const Ubatch) void {
        self.mctx.getMla().setInputKIdxs(self.self_k_idxs_mla.?, ubatch.?);

        self.mctx.getMla().setInputKqMask(self.self_kq_mask_mla.?, ubatch.?, self.cparams.causal_attn);

        self.mctx.getLid().setInputKIdxs(self.self_k_idxs_lid.?, ubatch.?);

        self.mctx.getLid().setInputKqMask(self.self_kq_mask_lid.?, ubatch.?, self.cparams.causal_attn);

        self.mctx.getLid().setInputKRot(self.self_k_rot_lid.?);
    }

    /// Ports `llm_graph_input_attn_k_dsa`'s `can_reuse` (llama-graph.cpp:572 @5266f24da).
    pub fn canReuse(self: *InpAttnKDsa, params: *const Params) bool {
        self.mctx = params.mctx.?.cast(kv_cache_dsa.DsaContext);
        return self.canReuseImpl(params);
    }

    /// Ports `llm_graph_input_attn_k_dsa`'s `can_reuse_impl` (llama-graph.cpp:578
    /// @5266f24da): `can_reuse` without re-binding the memory context.
    pub fn canReuseImpl(self: *const InpAttnKDsa, params: *const Params) bool {
        var res = true;
        res = res and self.self_k_idxs_mla.?.ne[0] == params.ubatch.n_tokens;
        res = res and self.self_k_idxs_lid.?.ne[0] == params.ubatch.n_tokens;
        res = canReuseKqMask(self.self_kq_mask_mla.?, self.mctx.getMla(), &params.ubatch, &params.cparams) and res;
        res = canReuseKqMask(self.self_kq_mask_lid.?, self.mctx.getLid(), &params.ubatch, &params.cparams) and res;
        return res;
    }
};

/// Ports `llm_graph_input_attn_k_dsa_iswa` (llama-graph.h:438 @5266f24da):
/// the DSA input for the full-attention layers and a K-only input for the
/// SWA layers. It owns both.
pub const InpAttnKDsaIswa = struct {
    inp_dsa: *InpAttnKDsa,
    inp_swa: *InpAttnK,
    mctx: *const kv_cache_dsa_iswa.DsaIswaContext,

    pub fn getDsa(self: *const InpAttnKDsaIswa) *InpAttnKDsa {
        return self.inp_dsa;
    }
    pub fn getSwa(self: *const InpAttnKDsaIswa) *InpAttnK {
        return self.inp_swa;
    }

    pub fn deinit(self: *InpAttnKDsaIswa) void {
        allocator.destroy(self.inp_dsa);
        allocator.destroy(self.inp_swa);
    }

    /// Ports `llm_graph_input_attn_k_dsa_iswa`'s `set_input` (llama-graph.cpp:590 @5266f24da).
    pub fn setInput(self: *InpAttnKDsaIswa, ubatch: ?*const Ubatch) void {
        self.inp_dsa.setInput(ubatch);
        self.inp_swa.setInput(ubatch);
    }

    /// Ports `llm_graph_input_attn_k_dsa_iswa`'s `can_reuse` (llama-graph.cpp:595
    /// @5266f24da): re-binds both halves to the new context.
    pub fn canReuse(self: *InpAttnKDsaIswa, params: *const Params) bool {
        const mctx = params.mctx.?.cast(kv_cache_dsa_iswa.DsaIswaContext);
        self.mctx = mctx;

        self.inp_dsa.mctx = mctx.getDsa();
        self.inp_swa.mctx = mctx.getSwa();

        var res = true;
        res = self.inp_dsa.canReuseImpl(params) and res;
        res = self.inp_swa.canReuseImpl(params) and res;
        return res;
    }
};

/// Ports `llm_graph_input_attn_kv_iswa` (llama-graph.h:483 @5266f24da): the
/// indices, masks and rotations of both halves of an iSWA cache.
pub const InpAttnKvIswa = struct {
    /// I64 `[n_batch]`.
    self_k_idxs: ?*Tensor = null,
    /// I64 `[n_batch]` or `[n_batch*n_embd_v_gqa]`.
    self_v_idxs: ?*Tensor = null,
    /// I64 `[n_batch]`.
    self_k_idxs_swa: ?*Tensor = null,
    /// I64 `[n_batch]` or `[n_batch*n_embd_v_gqa]`.
    self_v_idxs_swa: ?*Tensor = null,
    /// F32/F16 `[n_kv, n_batch/n_stream, 1, n_stream]`.
    self_kq_mask: ?*Tensor = null,
    self_kq_mask_cnv: ?*Tensor = null,
    /// F32/F16 `[n_kv, n_batch/n_stream, 1, n_stream]`.
    self_kq_mask_swa: ?*Tensor = null,
    self_kq_mask_swa_cnv: ?*Tensor = null,
    self_k_rot: ?*Tensor = null,
    self_v_rot: ?*Tensor = null,
    self_k_rot_swa: ?*Tensor = null,
    self_v_rot_swa: ?*Tensor = null,
    hparams: *const Hparams,
    cparams: Cparams,
    mctx: *const kv_cache_iswa.IswaContext,

    pub fn getKIdxs(self: *const InpAttnKvIswa) *Tensor {
        return self.self_k_idxs.?;
    }
    pub fn getVIdxs(self: *const InpAttnKvIswa) *Tensor {
        return self.self_v_idxs.?;
    }
    pub fn getKIdxsSwa(self: *const InpAttnKvIswa) *Tensor {
        return self.self_k_idxs_swa.?;
    }
    pub fn getVIdxsSwa(self: *const InpAttnKvIswa) *Tensor {
        return self.self_v_idxs_swa.?;
    }
    pub fn getKqMask(self: *const InpAttnKvIswa) ?*Tensor {
        return self.self_kq_mask_cnv;
    }
    pub fn getKqMaskSwa(self: *const InpAttnKvIswa) ?*Tensor {
        return self.self_kq_mask_swa_cnv;
    }

    fn allocated(t: ?*Tensor) ?*Tensor {
        const x = t orelse return null;
        return if (x.buffer != null) x else null;
    }

    /// Ports `llm_graph_input_attn_kv_iswa`'s `set_input` (llama-graph.cpp:609
    /// @5266f24da): each half's tensors are set only when allocated, since a
    /// model may have no layers on one side of the window.
    pub fn setInput(self: *InpAttnKvIswa, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        const base = self.mctx.getBase();
        const swa = self.mctx.getSwa();

        if (allocated(self.self_k_idxs)) |t| {
            base.setInputKIdxs(t, ubatch);
            if (self.self_v_idxs) |v| base.setInputVIdxs(v, ubatch);
        }

        // The mask guards on its own buffer: shared cells leave the indices
        // unbacked while the mask stays live.
        if (allocated(self.self_kq_mask)) |m| base.setInputKqMask(m, ubatch, self.cparams.causal_attn);

        if (allocated(self.self_k_idxs_swa)) |t| {
            swa.setInputKIdxs(t, ubatch);
            if (self.self_v_idxs_swa) |v| swa.setInputVIdxs(v, ubatch);
        }

        if (allocated(self.self_kq_mask_swa)) |m| swa.setInputKqMask(m, ubatch, self.cparams.causal_attn);

        if (allocated(self.self_k_rot)) |r| base.setInputKRot(r);
        if (allocated(self.self_v_rot)) |r| base.setInputVRot(r);
        if (allocated(self.self_k_rot_swa)) |r| swa.setInputKRot(r);
        if (allocated(self.self_v_rot_swa)) |r| swa.setInputVRot(r);
    }

    /// Ports `llm_graph_input_attn_kv_iswa`'s `can_reuse` (llama-graph.cpp:652 @5266f24da).
    pub fn canReuse(self: *InpAttnKvIswa, params: *const Params) bool {
        const mctx = params.mctx.?.cast(kv_cache_iswa.IswaContext);
        self.mctx = mctx;

        var res = true;

        // The C++'s TODO leaves the V indices unchecked, here as there.
        if (allocated(self.self_k_idxs)) |t| res = res and t.ne[0] == params.ubatch.n_tokens;
        if (allocated(self.self_kq_mask)) |m| res = canReuseKqMask(m, mctx.getBase(), &params.ubatch, &params.cparams) and res;
        if (allocated(self.self_k_idxs_swa)) |t| res = res and t.ne[0] == params.ubatch.n_tokens;
        if (allocated(self.self_kq_mask_swa)) |m| res = canReuseKqMask(m, mctx.getSwa(), &params.ubatch, &params.cparams) and res;

        return res;
    }
};

/// Ports `llm_graph_input_attn_k_iswa` (llama-graph.h:529 @5266f24da): the
/// indices, masks and rotations of both halves of a K-only iSWA cache,
/// whose K serves as V too (MLA-style attention).
pub const InpAttnKIswa = struct {
    /// I64 `[n_batch]`.
    self_k_idxs: ?*Tensor = null,
    /// I64 `[n_batch]`.
    self_k_idxs_swa: ?*Tensor = null,
    /// F32/F16 `[n_kv, n_batch/n_stream, 1, n_stream]`.
    self_kq_mask: ?*Tensor = null,
    self_kq_mask_cnv: ?*Tensor = null,
    /// F32/F16 `[n_kv, n_batch/n_stream, 1, n_stream]`.
    self_kq_mask_swa: ?*Tensor = null,
    self_kq_mask_swa_cnv: ?*Tensor = null,
    self_k_rot: ?*Tensor = null,
    self_k_rot_swa: ?*Tensor = null,
    hparams: *const Hparams,
    cparams: Cparams,
    mctx: *const kv_cache_iswa.IswaContext,

    pub fn getKIdxs(self: *const InpAttnKIswa) *Tensor {
        return self.self_k_idxs.?;
    }
    pub fn getKIdxsSwa(self: *const InpAttnKIswa) *Tensor {
        return self.self_k_idxs_swa.?;
    }
    pub fn getKqMask(self: *const InpAttnKIswa) ?*Tensor {
        return self.self_kq_mask_cnv;
    }
    pub fn getKqMaskSwa(self: *const InpAttnKIswa) ?*Tensor {
        return self.self_kq_mask_swa_cnv;
    }

    fn allocated(t: ?*Tensor) ?*Tensor {
        const x = t orelse return null;
        return if (x.buffer != null) x else null;
    }

    /// Ports `llm_graph_input_attn_k_iswa`'s `set_input` (llama-graph.cpp:682
    /// @5266f24da): either half's tensors may be unallocated when the model
    /// has no layers on that side of the window.
    pub fn setInput(self: *InpAttnKIswa, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        const base = self.mctx.getBase();
        const swa = self.mctx.getSwa();

        if (allocated(self.self_k_idxs)) |t| base.setInputKIdxs(t, ubatch);

        // The mask guards on its own buffer: shared cells leave the indices
        // unbacked while the mask stays live.
        if (allocated(self.self_kq_mask)) |m| base.setInputKqMask(m, ubatch, self.cparams.causal_attn);

        if (allocated(self.self_k_idxs_swa)) |t| swa.setInputKIdxs(t, ubatch);

        if (allocated(self.self_kq_mask_swa)) |m| swa.setInputKqMask(m, ubatch, self.cparams.causal_attn);

        if (allocated(self.self_k_rot)) |r| base.setInputKRot(r);
        if (allocated(self.self_k_rot_swa)) |r| swa.setInputKRot(r);
    }

    /// Ports `llm_graph_input_attn_k_iswa`'s `can_reuse` (llama-graph.cpp:711 @5266f24da).
    pub fn canReuse(self: *InpAttnKIswa, params: *const Params) bool {
        const mctx = params.mctx.?.cast(kv_cache_iswa.IswaContext);
        self.mctx = mctx;

        var res = true;

        if (allocated(self.self_k_idxs)) |t| res = res and t.ne[0] == params.ubatch.n_tokens;
        if (allocated(self.self_kq_mask)) |m| res = canReuseKqMask(m, mctx.getBase(), &params.ubatch, &params.cparams) and res;
        if (allocated(self.self_k_idxs_swa)) |t| res = res and t.ne[0] == params.ubatch.n_tokens;
        if (allocated(self.self_kq_mask_swa)) |m| res = canReuseKqMask(m, mctx.getSwa(), &params.ubatch, &params.cparams) and res;

        return res;
    }
};

/// Ports `llm_graph_input_dsv4_raw` (llama-graph.h:570 @5266f24da): DSV4's
/// raw inputs are SWA-only, but their mask may be stream-shaped, so raw K
/// can be concatenated with the compressed K in one attention op. It is
/// not an input of its own; `InpDsv4` owns and sets it.
pub const InpDsv4Raw = struct {
    /// I64 `[n_batch]`.
    self_k_idxs: ?*Tensor = null,
    /// F32/F16 `[n_kv, n_batch/n_stream, 1, n_stream]`.
    self_kq_mask: ?*Tensor = null,
    self_kq_mask_cnv: ?*Tensor = null,
    self_k_rot: ?*Tensor = null,
    cparams: Cparams,
    mctx: *const kv_cache_dsv4.RawContext,

    pub fn getKIdxs(self: *const InpDsv4Raw) *Tensor {
        return self.self_k_idxs.?;
    }
    pub fn getKqMask(self: *const InpDsv4Raw) ?*Tensor {
        return self.self_kq_mask_cnv;
    }

    /// Ports `llm_graph_input_dsv4_raw`'s `set_input` (llama-graph.cpp:980 @5266f24da).
    pub fn setInput(self: *InpDsv4Raw, ubatch: ?*const Ubatch) void {
        if (self.self_k_idxs) |t| {
            if (t.buffer != null) self.mctx.setInputKIdxs(t);
        }
        if (self.self_kq_mask) |m| {
            if (m.buffer != null) self.mctx.setInputKqMask(m, ubatch.?, self.cparams.causal_attn);
        }
        if (self.self_k_rot) |r| self.mctx.setInputKRot(r);
    }
};

/// Mirrors `llm_graph_input_dsv4`'s `comp_input` (llama-graph.h:598
/// @5266f24da): one compressor's plan, as graph inputs. A plan's empty
/// index list is no tensor at all.
pub const InpDsv4Comp = struct {
    /// I32 `[n_state]`.
    state_pos: ?*Tensor = null,
    /// I32 `[n_state_persist]`.
    state_persist_src_idxs: ?*Tensor = null,
    /// I32 `[n_state_persist]`.
    state_persist_dst_idxs: ?*Tensor = null,
    /// I32 `[n_state_restore]`.
    state_restore_src_idxs: ?*Tensor = null,
    /// I32 `[n_state_restore]`.
    state_restore_dst_idxs: ?*Tensor = null,
    /// I32 `[n_state_snapshot]`.
    state_snapshot_src_idxs: ?*Tensor = null,
    /// I32 `[n_state_snapshot]`.
    state_snapshot_dst_idxs: ?*Tensor = null,
    /// I32 `[ratio*n_state_write]`.
    state_read_idxs: ?*Tensor = null,
    /// I64 `[n_state_write]`.
    state_write_idxs: ?*Tensor = null,
    /// I32 `[n_state_write]`.
    state_write_pos: ?*Tensor = null,
    /// F32 `[n_kv, n_batch/n_stream, 1, n_stream]`.
    kq_mask: ?*Tensor = null,
    k_rot: ?*Tensor = null,
};

/// Ports `dsv4_set_i64` and `dsv4_set_i32` (llama-graph.cpp:739, 748
/// @5266f24da): an allocated index input from its plan list.
fn dsv4SetIdx(comptime T: type, dst_: ?*Tensor, src: []const T, comptime line: c_int) void {
    const dst = dst_ orelse return;
    if (dst.buffer == null) return;

    impl.assert(dst.ne[0] == @as(i64, @intCast(src.len)), "llama-graph.cpp", line, "dst->ne[0] == (int64_t) src.size()");
    c.ggml_backend_tensor_set(dst, src.ptr, 0, src.len * c.ggml_element_size(dst));
}

/// Ports `dsv4_set_kq_mask` (llama-graph.cpp:757 @5266f24da): each query
/// token sees its completed compressed rows and none past them.
fn dsv4SetKqMask(dst_: ?*Tensor, plan: *const kv_cache_dsv4.CompPlan, n_tokens: u32, n_stream: i64) void {
    const dst = dst_ orelse return;
    if (dst.buffer == null) return;

    impl.assert(dst.type == c.GGML_TYPE_F32 or dst.type == c.GGML_TYPE_F16, "llama-graph.cpp", 766, "dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16");
    impl.assert(n_stream > 0, "llama-graph.cpp", 767, "n_stream > 0");
    impl.assert(@rem(@as(i64, n_tokens), n_stream) == 0, "llama-graph.cpp", 768, "n_tokens%n_stream == 0");
    impl.assert(dst.ne[0] == plan.n_kv, "llama-graph.cpp", 769, "dst->ne[0] == plan.n_kv");
    impl.assert(dst.ne[1] == @divTrunc(@as(i64, n_tokens), n_stream), "llama-graph.cpp", 770, "dst->ne[1] == (int64_t) n_tokens/n_stream");
    impl.assert(dst.ne[2] == 1, "llama-graph.cpp", 771, "dst->ne[2] == 1");
    impl.assert(dst.ne[3] == n_stream, "llama-graph.cpp", 772, "dst->ne[3] == n_stream");
    impl.assert(@as(i64, @intCast(plan.n_visible.items.len)) == @as(i64, n_tokens), "llama-graph.cpp", 773, "(int64_t) plan.n_visible.size() == (int64_t) n_tokens");
    impl.assert(c.ggml_backend_buffer_is_host(dst.buffer), "llama-graph.cpp", 774, "ggml_backend_buffer_is_host(dst->buffer)");

    const ne0: usize = @intCast(dst.ne[0]);
    if (dst.type == c.GGML_TYPE_F32) {
        const data: [*]f32 = @ptrCast(@alignCast(dst.data));
        for (0..n_tokens) |i| {
            const n_visible = plan.n_visible.items[i];
            for (0..ne0) |j| data[i * ne0 + j] = if (@as(i64, @intCast(j)) < n_visible) 0.0 else -std.math.inf(f32);
        }
    } else if (dst.type == c.GGML_TYPE_F16) {
        const data: [*]u16 = @ptrCast(@alignCast(dst.data));
        const fp16_ninf: u16 = c.ggml_fp32_to_fp16(-std.math.inf(f32));
        const fp16_zero: u16 = c.ggml_fp32_to_fp16(0.0);
        for (0..n_tokens) |i| {
            const n_visible = plan.n_visible.items[i];
            for (0..ne0) |j| data[i * ne0 + j] = if (@as(i64, @intCast(j)) < n_visible) fp16_zero else fp16_ninf;
        }
    }
}

/// Ports `dsv4_build_raw_kq_mask` (llama-graph.cpp:801 @5266f24da).
fn dsv4BuildRawKqMask(ctx: *c.ggml_context, mctx: *const kv_cache_dsv4.RawContext, ubatch: *const Ubatch, cparams: *const Cparams, n_stream: i64) *Tensor {
    const n_kv: i64 = mctx.getNKv();
    const n_tokens: i64 = ubatch.n_tokens;

    impl.assert(n_stream > 0, "llama-graph.cpp", 810, "n_stream > 0");
    impl.assert(@rem(n_tokens, n_stream) == 0, "llama-graph.cpp", 811, "n_tokens%n_stream == 0");

    const ty: c.enum_ggml_type = if (cparams.flash_attn) c.GGML_TYPE_F16 else c.GGML_TYPE_F32;

    const res: *Tensor = c.ggml_new_tensor_4d(ctx, ty, n_kv, @divTrunc(n_tokens, n_stream), 1, n_stream);
    c.ggml_set_input(res);
    _ = c.ggml_set_name(res, "attn_inp_kq_mask");
    return res;
}

/// Ports `dsv4_can_reuse_raw_kq_mask` (llama-graph.cpp:822 @5266f24da).
fn dsv4CanReuseRawKqMask(kq_mask: *const Tensor, mctx: *const kv_cache_dsv4.RawContext, ubatch: *const Ubatch, n_stream: i64) bool {
    const n_kv: i64 = mctx.getNKv();
    const n_tokens: i64 = ubatch.n_tokens;

    impl.assert(n_stream > 0, "llama-graph.cpp", 830, "n_stream > 0");

    var res = true;
    res = res and (kq_mask.ne[0] == n_kv);
    res = res and (kq_mask.ne[1] == @divTrunc(n_tokens, n_stream));
    res = res and (kq_mask.ne[2] == 1);
    res = res and (kq_mask.ne[3] == n_stream);
    return res;
}

/// Ports `dsv4_set_comp_inputs` (llama-graph.cpp:864 @5266f24da).
fn dsv4SetCompInputs(inp: *const InpDsv4Comp, plan: *const kv_cache_dsv4.CompPlan, name: [*:0]const u8, debug: bool, n_tokens: u32, n_stream: i64) void {
    dsv4SetIdx(i32, inp.state_pos, plan.state_pos.items, 753);
    dsv4SetIdx(i32, inp.state_persist_src_idxs, plan.state_persist_src_idxs.items, 753);
    dsv4SetIdx(i32, inp.state_persist_dst_idxs, plan.state_persist_dst_idxs.items, 753);
    dsv4SetIdx(i32, inp.state_restore_src_idxs, plan.state_restore_src_idxs.items, 753);
    dsv4SetIdx(i32, inp.state_restore_dst_idxs, plan.state_restore_dst_idxs.items, 753);
    dsv4SetIdx(i32, inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs.items, 753);
    dsv4SetIdx(i32, inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs.items, 753);
    dsv4SetIdx(i32, inp.state_read_idxs, plan.state_read_idxs.items, 753);
    dsv4SetIdx(i64, inp.state_write_idxs, plan.state_write_idxs.items, 744);
    dsv4SetIdx(i32, inp.state_write_pos, plan.state_write_pos.items, 753);
    dsv4SetKqMask(inp.kq_mask, plan, n_tokens, n_stream);

    if (debug or kv_cache_dsv4.compressDebug()) {
        const persist = kv_cache_dsv4.planPositions(plan.state_persist_dst_idxs.items);
        defer allocator.free(persist);
        const write_pos = kv_cache_dsv4.planPositions(plan.state_write_pos.items);
        defer allocator.free(write_pos);
        impl.logInfo("%s: %s n_tokens=%u, n_stream=%d, state_persist_dst=%s, state_write_pos=%s\n", .{ "dsv4_set_comp_inputs", name, n_tokens, @as(c_int, @truncate(n_stream)), persist.ptr, write_pos.ptr });
    }
}

/// Ports `dsv4_can_reuse_tensor_1d` (llama-graph.cpp:891 @5266f24da).
fn dsv4CanReuseTensor1d(t: ?*Tensor, ne0: usize) bool {
    if (t) |x| return x.ne[0] == @as(i64, @intCast(ne0));
    return ne0 == 0;
}

/// Ports `dsv4_can_reuse_kq_mask` (llama-graph.cpp:895 @5266f24da).
fn dsv4CanReuseKqMask(t: ?*Tensor, plan: *const kv_cache_dsv4.CompPlan, n_tokens: u32, n_stream: i64) bool {
    if (plan.n_kv == 0) return t == null;

    impl.assert(n_stream > 0, "llama-graph.cpp", 904, "n_stream > 0");

    const x = t orelse return false;
    return x.ne[0] == plan.n_kv and
        x.ne[1] == @divTrunc(@as(i64, n_tokens), n_stream) and
        x.ne[2] == 1 and
        x.ne[3] == n_stream;
}

/// Ports `dsv4_can_reuse_comp_input` (llama-graph.cpp:913 @5266f24da).
fn dsv4CanReuseCompInput(inp: *const InpDsv4Comp, plan: *const kv_cache_dsv4.CompPlan, n_tokens: u32, n_stream: i64) bool {
    var res = true;
    res = res and dsv4CanReuseTensor1d(inp.state_pos, plan.state_pos.items.len);
    res = res and dsv4CanReuseTensor1d(inp.state_persist_src_idxs, plan.state_persist_src_idxs.items.len);
    res = res and dsv4CanReuseTensor1d(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs.items.len);
    res = res and dsv4CanReuseTensor1d(inp.state_restore_src_idxs, plan.state_restore_src_idxs.items.len);
    res = res and dsv4CanReuseTensor1d(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs.items.len);
    res = res and dsv4CanReuseTensor1d(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs.items.len);
    res = res and dsv4CanReuseTensor1d(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs.items.len);
    res = res and dsv4CanReuseTensor1d(inp.state_read_idxs, plan.state_read_idxs.items.len);
    res = res and dsv4CanReuseTensor1d(inp.state_write_idxs, plan.state_write_idxs.items.len);
    res = res and dsv4CanReuseTensor1d(inp.state_write_pos, plan.state_write_pos.items.len);
    res = res and dsv4CanReuseKqMask(inp.kq_mask, plan, n_tokens, n_stream);
    return res;
}

/// Ports `dsv4_build_input_1d` (llama-graph.cpp:934 @5266f24da): no
/// tensor for an empty list.
fn dsv4BuildInput1d(ctx: *c.ggml_context, ty: c.enum_ggml_type, ne0: usize, comptime prefix: []const u8, name: []const u8, comptime suffix: []const u8) ?*Tensor {
    if (ne0 == 0) return null;

    const res: *Tensor = c.ggml_new_tensor_1d(ctx, ty, @intCast(ne0));
    c.ggml_set_input(res);
    var buf: [64]u8 = undefined;
    const full = std.mem.printSentinel(&buf, prefix ++ "{s}" ++ suffix, .{name}, 0) catch unreachable;
    _ = c.ggml_set_name(res, full.ptr);
    return res;
}

/// Ports `dsv4_build_comp_inputs` (llama-graph.cpp:950 @5266f24da): the
/// mask is f16 where its attention is fused -- flash attention for CSA and
/// HCA, the fused lightning indexer for its own.
fn dsv4BuildCompInputs(ctx: *c.ggml_context, inp: *InpDsv4Comp, plan: *const kv_cache_dsv4.CompPlan, name: []const u8, cparams: *const Cparams, n_stream: i64) void {
    inp.state_pos = dsv4BuildInput1d(ctx, c.GGML_TYPE_I32, plan.state_pos.items.len, "dsv4_", name, "_state_pos");
    inp.state_persist_src_idxs = dsv4BuildInput1d(ctx, c.GGML_TYPE_I32, plan.state_persist_src_idxs.items.len, "dsv4_", name, "_state_persist_src_idxs");
    inp.state_persist_dst_idxs = dsv4BuildInput1d(ctx, c.GGML_TYPE_I32, plan.state_persist_dst_idxs.items.len, "dsv4_", name, "_state_persist_dst_idxs");
    inp.state_restore_src_idxs = dsv4BuildInput1d(ctx, c.GGML_TYPE_I32, plan.state_restore_src_idxs.items.len, "dsv4_", name, "_state_restore_src_idxs");
    inp.state_restore_dst_idxs = dsv4BuildInput1d(ctx, c.GGML_TYPE_I32, plan.state_restore_dst_idxs.items.len, "dsv4_", name, "_state_restore_dst_idxs");
    inp.state_snapshot_src_idxs = dsv4BuildInput1d(ctx, c.GGML_TYPE_I32, plan.state_snapshot_src_idxs.items.len, "dsv4_", name, "_state_snapshot_src_idxs");
    inp.state_snapshot_dst_idxs = dsv4BuildInput1d(ctx, c.GGML_TYPE_I32, plan.state_snapshot_dst_idxs.items.len, "dsv4_", name, "_state_snapshot_dst_idxs");
    inp.state_read_idxs = dsv4BuildInput1d(ctx, c.GGML_TYPE_I32, plan.state_read_idxs.items.len, "dsv4_", name, "_state_read_idxs");
    inp.state_write_idxs = dsv4BuildInput1d(ctx, c.GGML_TYPE_I64, plan.state_write_idxs.items.len, "dsv4_", name, "_state_write_idxs");
    inp.state_write_pos = dsv4BuildInput1d(ctx, c.GGML_TYPE_I32, plan.state_write_pos.items.len, "dsv4_", name, "_state_write_pos");

    if (plan.n_kv > 0) {
        const n_tokens: i64 = @intCast(plan.n_visible.items.len);

        impl.assert(n_stream > 0, "llama-graph.cpp", 971, "n_stream > 0");
        impl.assert(@rem(n_tokens, n_stream) == 0, "llama-graph.cpp", 972, "n_tokens%n_stream == 0");

        const is_lid = std.mem.eql(u8, name, "lid");
        const use_f16 = (!is_lid and cparams.flash_attn) or (is_lid and cparams.fused_lid);
        const m: *Tensor = c.ggml_new_tensor_4d(ctx, if (use_f16) c.GGML_TYPE_F16 else c.GGML_TYPE_F32, plan.n_kv, @divTrunc(n_tokens, n_stream), 1, n_stream);
        inp.kq_mask = m;
        c.ggml_set_input(m);
        var buf: [64]u8 = undefined;
        const full = std.mem.printSentinel(&buf, "dsv4_{s}_kq_mask", .{name}, 0) catch unreachable;
        _ = c.ggml_set_name(m, full.ptr);
    }
}

/// Ports `llm_graph_input_dsv4` (llama-graph.h:596 @5266f24da): the raw
/// cache's inputs and each compressor's plan.
pub const InpDsv4 = struct {
    inp_raw: *InpDsv4Raw,
    inp_csa: InpDsv4Comp = .{},
    inp_hca: InpDsv4Comp = .{},
    inp_lid: InpDsv4Comp = .{},
    cparams: Cparams,
    mctx: *const kv_cache_dsv4.Dsv4Context,
    /// `LLAMA_GRAPH_INPUT_DEBUG`, as the base input reads it.
    debug: c_int = 0,

    pub fn getRaw(self: *const InpDsv4) *InpDsv4Raw {
        return self.inp_raw;
    }
    pub fn getCsa(self: *const InpDsv4) *const InpDsv4Comp {
        return &self.inp_csa;
    }
    pub fn getHca(self: *const InpDsv4) *const InpDsv4Comp {
        return &self.inp_hca;
    }
    pub fn getLid(self: *const InpDsv4) *const InpDsv4Comp {
        return &self.inp_lid;
    }

    pub fn deinit(self: *InpDsv4) void {
        allocator.destroy(self.inp_raw);
    }

    /// Ports `llm_graph_input_dsv4`'s `set_input` (llama-graph.cpp:994 @5266f24da).
    pub fn setInput(self: *InpDsv4, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        const plan_csa = self.mctx.getCsaPlanFor(ubatch);
        const plan_hca = self.mctx.getHcaPlanFor(ubatch);
        const plan_lid = self.mctx.getLidPlanFor(ubatch);
        const n_stream = plan_csa.n_stream;

        self.inp_raw.mctx = self.mctx.getRaw();
        self.inp_raw.setInput(ubatch);

        dsv4SetCompInputs(&self.inp_csa, plan_csa, "csa", self.debug > 0, ubatch.n_tokens, n_stream);
        dsv4SetCompInputs(&self.inp_hca, plan_hca, "hca", self.debug > 0, ubatch.n_tokens, n_stream);
        dsv4SetCompInputs(&self.inp_lid, plan_lid, "lid", self.debug > 0, ubatch.n_tokens, n_stream);

        if (self.inp_csa.k_rot) |r| {
            if (r.buffer != null) self.mctx.getCsa().setInputKRot(r);
        }
        if (self.inp_hca.k_rot) |r| {
            if (r.buffer != null) self.mctx.getHca().setInputKRot(r);
        }
        if (self.inp_lid.k_rot) |r| {
            if (r.buffer != null) self.mctx.getLid().setInputKRot(r);
        }
    }

    /// Ports `llm_graph_input_dsv4`'s `can_reuse` (llama-graph.cpp:1020 @5266f24da).
    pub fn canReuse(self: *InpDsv4, params: *const Params) bool {
        const mctx = params.mctx.?.cast(kv_cache_dsv4.Dsv4Context);

        self.mctx = mctx;
        self.inp_raw.mctx = mctx.getRaw();

        var res = true;

        const plan_csa = mctx.getCsaPlanFor(&params.ubatch);
        const plan_hca = mctx.getHcaPlanFor(&params.ubatch);
        const plan_lid = mctx.getLidPlanFor(&params.ubatch);
        const n_stream = plan_csa.n_stream;

        const raw_ctx = mctx.getRaw();
        self.inp_raw.mctx = raw_ctx;

        if (self.inp_raw.self_k_idxs) |t| {
            if (t.buffer != null) res = res and t.ne[0] == raw_ctx.getNWrite();
        }
        if (self.inp_raw.self_kq_mask) |m| {
            if (m.buffer != null) res = res and dsv4CanReuseRawKqMask(m, raw_ctx, &params.ubatch, n_stream);
        }

        res = res and dsv4CanReuseCompInput(&self.inp_csa, plan_csa, params.ubatch.n_tokens, n_stream);
        res = res and dsv4CanReuseCompInput(&self.inp_hca, plan_hca, params.ubatch.n_tokens, n_stream);
        res = res and dsv4CanReuseCompInput(&self.inp_lid, plan_lid, params.ubatch.n_tokens, n_stream);

        return res;
    }
};

/// Ports `llm_graph_input_mem_hybrid` (llama-graph.h:660 @5266f24da).
pub const InpMemHybrid = struct {
    inp_attn: *InpAttnKv,
    inp_rs: *InpRs,
    cparams: Cparams,
    mctx: *const memory_hybrid.HybridContext,

    pub fn getAttn(self: *const InpMemHybrid) *InpAttnKv {
        return self.inp_attn;
    }
    pub fn getRecr(self: *const InpMemHybrid) *InpRs {
        return self.inp_rs;
    }

    pub fn deinit(self: *InpMemHybrid) void {
        allocator.destroy(self.inp_attn);
        allocator.destroy(self.inp_rs);
    }

    /// Ports `llm_graph_input_mem_hybrid`'s `set_input` (llama-graph.cpp:1086 @5266f24da).
    pub fn setInput(self: *InpMemHybrid, ubatch: ?*const Ubatch) void {
        const attn = self.mctx.getAttn();
        attn.setInputKIdxs(self.inp_attn.self_k_idxs.?, ubatch.?);
        attn.setInputVIdxs(self.inp_attn.self_v_idxs.?, ubatch.?);
        attn.setInputKqMask(self.inp_attn.self_kq_mask.?, ubatch.?, self.cparams.causal_attn);
        if (self.inp_attn.self_k_rot) |r| attn.setInputKRot(r);
        if (self.inp_attn.self_v_rot) |r| attn.setInputVRot(r);

        const recr = self.mctx.getRecr();
        const n_rs = recr.getNRs();
        if (self.inp_rs.s_copy) |s_copy| {
            impl.assert(c.ggml_backend_buffer_is_host(s_copy.buffer), "llama-graph.cpp", 1103, "ggml_backend_buffer_is_host(inp_rs->s_copy->buffer)");
            const data: [*]i32 = @ptrCast(@alignCast(s_copy.data));
            for (0..n_rs) |i| data[i] = recr.sCopy(@intCast(i));
        }
    }

    /// Ports `llm_graph_input_mem_hybrid`'s `can_reuse` (llama-graph.cpp:1113 @5266f24da).
    pub fn canReuse(self: *InpMemHybrid, params: *const Params) bool {
        const mctx = params.mctx.?.cast(memory_hybrid.HybridContext);
        self.mctx = mctx;
        const recr = mctx.getRecr();
        var res = true;
        res = res and self.inp_attn.self_k_idxs.?.ne[0] == params.ubatch.n_tokens;
        res = res and canReuseKqMask(self.inp_attn.self_kq_mask.?, mctx.getAttn(), &params.ubatch, &params.cparams);
        res = res and self.inp_rs.s_copy.?.ne[0] == recr.getNRs();
        res = res and self.inp_rs.s_copy_main.?.ne[0] == params.ubatch.n_seqs;
        res = res and self.inp_rs.s_copy_extra.?.ne[0] == @as(i64, recr.getNRs()) - params.ubatch.n_seqs;
        res = res and self.inp_rs.head == recr.getHead();
        res = res and self.inp_rs.rs_z == recr.getRsZ();
        return res;
    }
};

/// Ports `llm_graph_input_mem_hybrid_iswa` (llama-graph.h:716 @5266f24da):
/// a hybrid memory's inputs with an iSWA attention half.
pub const InpMemHybridIswa = struct {
    inp_attn: *InpAttnKvIswa,
    inp_rs: *InpRs,
    cparams: Cparams,
    mctx: *const memory_hybrid_iswa.HybridIswaContext,

    pub fn getAttn(self: *const InpMemHybridIswa) *InpAttnKvIswa {
        return self.inp_attn;
    }
    pub fn getRecr(self: *const InpMemHybridIswa) *InpRs {
        return self.inp_rs;
    }

    pub fn deinit(self: *InpMemHybridIswa) void {
        allocator.destroy(self.inp_attn);
        allocator.destroy(self.inp_rs);
    }

    /// Ports `llm_graph_input_mem_hybrid_iswa`'s `set_input`
    /// (llama-graph.cpp:1179 @5266f24da): each half's tensors are set only
    /// when allocated, since a model may have no layers on one side of the
    /// window.
    pub fn setInput(self: *InpMemHybridIswa, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        const attn_ctx = self.mctx.getAttn();
        const inp_attn = self.inp_attn;

        if (InpAttnKvIswa.allocated(inp_attn.self_k_idxs)) |t| {
            attn_ctx.getBase().setInputKIdxs(t, ubatch);
            attn_ctx.getBase().setInputVIdxs(inp_attn.self_v_idxs.?, ubatch);
        }

        if (InpAttnKvIswa.allocated(inp_attn.self_kq_mask)) |m| attn_ctx.getBase().setInputKqMask(m, ubatch, self.cparams.causal_attn);

        if (InpAttnKvIswa.allocated(inp_attn.self_k_idxs_swa)) |t| {
            attn_ctx.getSwa().setInputKIdxs(t, ubatch);
            attn_ctx.getSwa().setInputVIdxs(inp_attn.self_v_idxs_swa.?, ubatch);
        }

        if (InpAttnKvIswa.allocated(inp_attn.self_kq_mask_swa)) |m| attn_ctx.getSwa().setInputKqMask(m, ubatch, self.cparams.causal_attn);

        if (inp_attn.self_k_rot) |r| attn_ctx.getBase().setInputKRot(r);
        if (inp_attn.self_v_rot) |r| attn_ctx.getBase().setInputVRot(r);
        if (inp_attn.self_k_rot_swa) |r| attn_ctx.getSwa().setInputKRot(r);
        if (inp_attn.self_v_rot_swa) |r| attn_ctx.getSwa().setInputVRot(r);

        const recr = self.mctx.getRecr();
        const n_rs = recr.getNRs();
        if (self.inp_rs.s_copy) |s_copy| {
            impl.assert(c.ggml_backend_buffer_is_host(s_copy.buffer), "llama-graph.cpp", 1221, "ggml_backend_buffer_is_host(inp_rs->s_copy->buffer)");
            const data: [*]i32 = @ptrCast(@alignCast(s_copy.data));
            // Copy destinations are always between head and head + n.
            for (0..n_rs) |i| data[i] = recr.sCopy(@intCast(i));
        }
    }

    /// Ports `llm_graph_input_mem_hybrid_iswa`'s `can_reuse`
    /// (llama-graph.cpp:1231 @5266f24da). Unlike the plain iSWA input's,
    /// the masks are checked whether or not they are allocated.
    pub fn canReuse(self: *InpMemHybridIswa, params: *const Params) bool {
        const mctx = params.mctx.?.cast(memory_hybrid_iswa.HybridIswaContext);
        self.mctx = mctx;
        const attn_ctx = mctx.getAttn();
        const recr = mctx.getRecr();
        const inp_attn = self.inp_attn;

        var res = true;

        // The C++'s TODO leaves the V indices unchecked, here as there.
        if (InpAttnKvIswa.allocated(inp_attn.self_k_idxs)) |t| res = res and t.ne[0] == params.ubatch.n_tokens;
        res = canReuseKqMask(inp_attn.self_kq_mask.?, attn_ctx.getBase(), &params.ubatch, &params.cparams) and res;
        if (InpAttnKvIswa.allocated(inp_attn.self_k_idxs_swa)) |t| res = res and t.ne[0] == params.ubatch.n_tokens;
        res = canReuseKqMask(inp_attn.self_kq_mask_swa.?, attn_ctx.getSwa(), &params.ubatch, &params.cparams) and res;

        res = res and self.inp_rs.s_copy.?.ne[0] == recr.getNRs();
        res = res and self.inp_rs.s_copy_main.?.ne[0] == params.ubatch.n_seqs;
        res = res and self.inp_rs.s_copy_extra.?.ne[0] == @as(i64, recr.getNRs()) - params.ubatch.n_seqs;
        res = res and self.inp_rs.head == recr.getHead();
        res = res and self.inp_rs.rs_z == recr.getRsZ();
        return res;
    }
};

/// Ports `llm_graph_input_mem_hybrid_k` (llama-graph.h:688 @5266f24da): a
/// hybrid memory whose attention half is a K-only (MLA) cache.
pub const InpMemHybridK = struct {
    inp_attn: *InpAttnK,
    inp_rs: *InpRs,
    cparams: Cparams,
    mctx: *const memory_hybrid.HybridContext,

    pub fn getAttn(self: *const InpMemHybridK) *InpAttnK {
        return self.inp_attn;
    }
    pub fn getRecr(self: *const InpMemHybridK) *InpRs {
        return self.inp_rs;
    }

    pub fn deinit(self: *InpMemHybridK) void {
        allocator.destroy(self.inp_attn);
        allocator.destroy(self.inp_rs);
    }

    /// Ports `llm_graph_input_mem_hybrid_k`'s `set_input` (llama-graph.cpp:1139 @5266f24da).
    pub fn setInput(self: *InpMemHybridK, ubatch: ?*const Ubatch) void {
        const attn = self.mctx.getAttn();
        attn.setInputKIdxs(self.inp_attn.self_k_idxs.?, ubatch.?);
        attn.setInputKqMask(self.inp_attn.self_kq_mask.?, ubatch.?, self.cparams.causal_attn);

        const recr = self.mctx.getRecr();
        const n_rs = recr.getNRs();
        if (self.inp_rs.s_copy) |s_copy| {
            impl.assert(c.ggml_backend_buffer_is_host(s_copy.buffer), "llama-graph.cpp", 1147, "ggml_backend_buffer_is_host(inp_rs->s_copy->buffer)");
            const data: [*]i32 = @ptrCast(@alignCast(s_copy.data));
            for (0..n_rs) |i| data[i] = recr.sCopy(@intCast(i));
        }
    }

    /// Ports `llm_graph_input_mem_hybrid_k`'s `can_reuse` (llama-graph.cpp:1157 @5266f24da).
    pub fn canReuse(self: *InpMemHybridK, params: *const Params) bool {
        const mctx = params.mctx.?.cast(memory_hybrid.HybridContext);
        self.mctx = mctx;
        const recr = mctx.getRecr();
        var res = true;
        res = res and self.inp_attn.self_k_idxs.?.ne[0] == params.ubatch.n_tokens;
        res = res and canReuseKqMask(self.inp_attn.self_kq_mask.?, mctx.getAttn(), &params.ubatch, &params.cparams);
        res = res and self.inp_rs.s_copy.?.ne[0] == recr.getNRs();
        res = res and self.inp_rs.s_copy_main.?.ne[0] == params.ubatch.n_seqs;
        res = res and self.inp_rs.s_copy_extra.?.ne[0] == @as(i64, recr.getNRs()) - params.ubatch.n_seqs;
        res = res and self.inp_rs.head == recr.getHead();
        res = res and self.inp_rs.rs_z == recr.getRsZ();
        return res;
    }
};

/// Ports `llm_graph_input_sampling` (llama-graph.h:744 @5266f24da).
pub const InpSampling = struct {
    samplers: SamplerMap,

    pub fn deinit(self: *InpSampling) void {
        self.samplers.deinit();
    }

    /// Ports `llm_graph_input_sampling`'s `set_input` (llama-graph.cpp:1267
    /// @5266f24da): only the samplers of the ubatch's outputs.
    ///
    /// The C++ walks a `std`'s `unordered_set` of the active sequences; each
    /// sampler's `backend_set_input` touches only its own state, so the
    /// order is not observable, and this walks them in sequence order.
    pub fn setInput(self: *InpSampling, ubatch_: ?*const Ubatch) void {
        const ubatch = ubatch_.?;
        var active = std.bit_set.StaticBitSet(cparams_mod.max_seq).empty;
        for (0..ubatch.n_tokens) |i| {
            if (ubatch.output[i] != 0) active.set(@intCast(ubatch.seq_id[i][0]));
        }
        var it = active.iterator(.{});
        while (it.next()) |seq_id| {
            const sampler = self.samplers.get(@intCast(seq_id)) orelse continue;
            if (sampler.*.iface.*.backend_set_input) |f| f(sampler);
        }
    }

    /// Ports `llm_graph_input_sampling`'s `can_reuse` (llama-graph.cpp:1290 @5266f24da).
    pub fn canReuse(self: *InpSampling, params: *const Params) bool {
        if (self.samplers.items.items.len != params.samplers.items.items.len) return false;
        for (params.samplers.items.items) |e| {
            if (self.samplers.get(e.seq_id) != e.sampler) return false;
        }
        return true;
    }
};

/// `std::map<llama_seq_id, llama_sampler *>`: kept sorted by sequence.
pub const SamplerMap = struct {
    items: std.ArrayList(Entry) = .empty,

    pub const Entry = struct { seq_id: i32, sampler: *c.llama_sampler };

    /// Return: a copy, the caller's to `deinit`.
    pub fn clone(self: *const SamplerMap) SamplerMap {
        var out: SamplerMap = .{};
        out.items.appendSlice(allocator, self.items.items) catch oom();
        return out;
    }

    /// Return: nothing.
    pub fn deinit(self: *SamplerMap) void {
        self.items.deinit(allocator);
    }

    /// Return: the sampler for `seq_id`, or null.
    pub fn get(self: *const SamplerMap, seq_id: i32) ?*c.llama_sampler {
        for (self.items.items) |e| if (e.seq_id == seq_id) return e.sampler;
        return null;
    }

    /// Sets or replaces `seq_id`'s sampler.
    ///
    /// Return: nothing.
    pub fn put(self: *SamplerMap, seq_id: i32, sampler: *c.llama_sampler) void {
        var pos: usize = 0;
        while (pos < self.items.items.len) : (pos += 1) {
            if (self.items.items[pos].seq_id == seq_id) {
                self.items.items[pos].sampler = sampler;
                return;
            }
            if (self.items.items[pos].seq_id > seq_id) break;
        }
        self.items.insert(allocator, pos, .{ .seq_id = seq_id, .sampler = sampler }) catch oom();
    }

    /// Removes `seq_id`'s sampler, if any.
    ///
    /// Return: whether there was one.
    pub fn remove(self: *SamplerMap, seq_id: i32) bool {
        for (self.items.items, 0..) |e, i| {
            if (e.seq_id == seq_id) {
                _ = self.items.orderedRemove(i);
                return true;
            }
        }
        return false;
    }

    /// Ports `llm_graph_params`'s `samplers_equal` (llama-graph.h:791 @5266f24da).
    pub fn eql(lhs: *const SamplerMap, rhs: *const SamplerMap) bool {
        if (lhs.items.items.len != rhs.items.items.len) return false;
        for (lhs.items.items) |e| {
            if (rhs.get(e.seq_id) != e.sampler) return false;
        }
        return true;
    }
};

/// Ports `build_attn_inp_kq_mask` (llama-graph.cpp:29 @5266f24da).
fn buildAttnInpKqMask(ctx: *c.ggml_context, mctx: *const kv_cache.KvCacheContext, ubatch: *const Ubatch, cparams: *const Cparams) *Tensor {
    const n_kv: i64 = mctx.getNKv();
    const n_tokens: i64 = ubatch.n_tokens;
    const n_stream: i64 = if (cparams.kv_unified) 1 else ubatch.n_seqs_unq;
    // Flash attention requires an f16 mask.
    const ty: c.enum_ggml_type = if (cparams.flash_attn) c.GGML_TYPE_F16 else c.GGML_TYPE_F32;
    const res = c.ggml_new_tensor_4d(ctx, ty, n_kv, @divTrunc(n_tokens, n_stream), 1, n_stream);
    c.ggml_set_input(res);
    _ = c.ggml_set_name(res, "attn_inp_kq_mask");
    return res;
}

/// Ports `can_reuse_kq_mask` (llama-graph.cpp:48 @5266f24da).
fn canReuseKqMask(kq_mask: *const Tensor, mctx: *const kv_cache.KvCacheContext, ubatch: *const Ubatch, cparams: *const Cparams) bool {
    const n_kv: i64 = mctx.getNKv();
    const n_tokens: i64 = ubatch.n_tokens;
    const n_stream: i64 = if (cparams.kv_unified) 1 else ubatch.n_seqs_unq;
    var res = true;
    res = res and (kq_mask.ne[0] == n_kv);
    res = res and (kq_mask.ne[1] == @divTrunc(n_tokens, n_stream));
    res = res and (kq_mask.ne[2] == 1);
    res = res and (kq_mask.ne[3] == n_stream);
    return res;
}

// -----------------------------------------------------------------------------
// Parameters and results

/// The `llm_graph_cb` callback (llama-graph.h:767 @5266f24da): called on
/// every named tensor as the graph is built.
pub const Callback = struct {
    ctx: ?*anyopaque = null,
    func: ?*const fn (ctx: ?*anyopaque, ubatch: *const Ubatch, cur: *Tensor, name: [*:0]const u8, il: c_int) void = null,
};

/// Mirrors `llm_graph_params` (llama-graph.h:771 @5266f24da).
pub const Params = struct {
    arch: Arch = .unknown,
    hparams: *const Hparams,
    cparams: Cparams = .{},
    /// A copy, holding a reference to the ubatch's data.
    ubatch: Ubatch = .{},
    gtype: GraphType = .default,
    sched: c.ggml_backend_sched_t = null,
    backend_cpu: c.ggml_backend_t = null,
    cvec: ?*const adapter.Cvec = null,
    loras: ?*const adapter.Loras = null,
    mctx: ?memory.Context = null,
    cross: ?*const Cross = null,
    samplers: SamplerMap = .{},
    n_outputs: u32 = 0,
    cb: Callback = .{},
    res: ?*Result = null,

    /// Return: a copy that owns its own references, for `set_params`.
    pub fn clone(self: *const Params) Params {
        var out = self.*;
        out.ubatch.retain();
        out.samplers = self.samplers.clone();
        return out;
    }

    /// Releases what `clone` took.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Params) void {
        self.ubatch.release();
        self.samplers.deinit();
    }

    /// Ports `llm_graph_params`'s `allow_reuse` (llama-graph.h:814 @5266f24da):
    /// whether `other` would build a graph of the same topology.
    pub fn allowReuse(self: *const Params, other: *const Params) bool {
        const u = &self.ubatch;
        const o = &other.ubatch;
        var can_reuse_ubatch =
            u.equalSeqs() == o.equalSeqs() and
            u.n_tokens == o.n_tokens and
            u.n_seq_tokens == o.n_seq_tokens and
            u.n_seqs == o.n_seqs and
            u.n_seqs_unq == o.n_seqs_unq and
            ((u.token == null and o.token == null) or
                (u.embd == null and o.embd == null) or
                (u.token != null and o.token != null and u.embd != null and o.embd != null));

        // With equal_seqs splits, the participating sequences must match.
        if (can_reuse_ubatch and u.equalSeqs()) {
            if (u.data == null) {
                can_reuse_ubatch = false;
            } else {
                for (0..u.n_seqs_unq) |s| {
                    can_reuse_ubatch = can_reuse_ubatch and u.seq_id_unq[s] == o.seq_id_unq[s];
                }
            }
        }
        if (!can_reuse_ubatch) return false;
        if (self.n_outputs != other.n_outputs) return false;
        if (!SamplerMap.eql(&self.samplers, &other.samplers)) return false;
        if (self.samplers.items.items.len > 0) {
            if (u.data == null or o.data == null) return false;
            // The outputs must be the same for all samplers.
            for (0..u.n_tokens) |i| {
                if (u.output[i] != o.output[i] or u.seq_id[i][0] != o.seq_id[i][0]) return false;
            }
        }
        if (self.cparams.nextn_layer_offset != other.cparams.nextn_layer_offset) return false;
        return self.cparams.embeddings == other.cparams.embeddings and
            self.cparams.embeddings_nextn == other.cparams.embeddings_nextn and
            self.cparams.embeddings_nextn_masked == other.cparams.embeddings_nextn_masked and
            self.cparams.causal_attn == other.cparams.causal_attn and
            self.arch == other.arch and
            self.gtype == other.gtype and
            self.cvec == other.cvec and
            self.loras == other.loras and
            self.cross == other.cross;
    }
};

/// Mirrors `llm_graph_fused_node` (llama-graph.h:886 @5266f24da).
pub const FusedNode = struct {
    op: FusedOp,
    tensor: *Tensor,
    il: c_int,
};

/// Ports `llm_graph_result` (llama-graph.h:892 @5266f24da): a built graph,
/// its inputs, and the tensors the context reads back.
pub const Result = struct {
    t_inp_tokens: ?*Tensor = null,
    /// `[n_embd_inp, n_tokens]`.
    t_inp_embd: ?*Tensor = null,
    t_logits: ?*Tensor = null,
    t_embd: ?*Tensor = null,
    t_embd_pooled: ?*Tensor = null,
    /// `[n_embd, n_outputs]`: the hidden state before the output norm.
    t_h_nextn: ?*Tensor = null,
    t_layer_inp: [max_layers + 1]?*Tensor = @splat(null),
    t_sampled: std.ArrayList(?*Tensor) = .empty,
    t_sampled_probs: std.ArrayList(?*Tensor) = .empty,
    t_sampled_logits: std.ArrayList(?*Tensor) = .empty,
    t_candidates: std.ArrayList(?*Tensor) = .empty,

    inputs: std.ArrayList(Input) = .empty,
    fused_nodes: std.ArrayList(FusedNode) = .empty,

    ctx_compute: ?*c.ggml_context = null,
    /// The memory the compute context lives in.
    buf_compute_meta: std.ArrayList(u8) = .empty,
    gf: ?*c.ggml_cgraph = null,
    max_nodes: i64,

    /// The previous graph's parameters, for deciding on reuse.
    params: ?Params = null,
    debug: c_int = 0,

    /// Ports the constructor of `llm_graph_result` (llama-graph.cpp:1308 @5266f24da).
    ///
    /// Return: the result, the caller's to `destroy`.
    pub fn create(max_nodes: i64) *Result {
        const self = allocator.create(Result) catch oom();
        self.* = .{ .max_nodes = max_nodes };
        self.reset();
        if (std.c.getenv("LLAMA_GRAPH_RESULT_DEBUG")) |v| self.debug = impl.atoi(v);
        return self;
    }

    /// Return: nothing.
    pub fn destroy(self: *Result) void {
        self.clearInputs();
        self.inputs.deinit(allocator);
        self.fused_nodes.deinit(allocator);
        self.t_sampled.deinit(allocator);
        self.t_sampled_probs.deinit(allocator);
        self.t_sampled_logits.deinit(allocator);
        self.t_candidates.deinit(allocator);
        if (self.ctx_compute) |ctx| c.ggml_free(ctx);
        self.buf_compute_meta.deinit(allocator);
        if (self.params) |*p| p.deinit();
        allocator.destroy(self);
    }

    fn clearInputs(self: *Result) void {
        for (self.inputs.items) |inp| inp.vtable.destroy(inp.ptr);
        self.inputs.clearRetainingCapacity();
    }

    /// Ports `llm_graph_result`'s `get_max_nodes` (llama-graph.cpp:1315 @5266f24da).
    pub fn getMaxNodes(self: *const Result) i64 {
        return self.max_nodes;
    }

    pub fn getCtx(self: *const Result) *c.ggml_context {
        return self.ctx_compute.?;
    }
    pub fn getGf(self: *const Result) *c.ggml_cgraph {
        return self.gf.?;
    }
    pub fn getLogits(self: *const Result) ?*Tensor {
        return self.t_logits;
    }
    pub fn getEmbd(self: *const Result) ?*Tensor {
        return self.t_embd;
    }
    pub fn getEmbdPooled(self: *const Result) ?*Tensor {
        return self.t_embd_pooled;
    }
    pub fn getHNextn(self: *const Result) ?*Tensor {
        return self.t_h_nextn;
    }
    pub fn getInpTokens(self: *const Result) ?*Tensor {
        return self.t_inp_tokens;
    }

    /// Ports `llm_graph_result`'s `reset` (llama-graph.cpp:1319 @5266f24da).
    pub fn reset(self: *Result) void {
        self.t_inp_tokens = null;
        self.t_inp_embd = null;
        self.t_logits = null;
        self.t_embd = null;
        self.t_embd_pooled = null;
        self.t_h_nextn = null;
        self.t_layer_inp = @splat(null);
        self.t_sampled.clearRetainingCapacity();
        self.t_sampled_probs.clearRetainingCapacity();
        self.t_sampled_logits.clearRetainingCapacity();
        self.t_candidates.clearRetainingCapacity();

        if (self.params) |*p| p.deinit();
        self.params = null;

        self.clearInputs();
        self.fused_nodes.clearRetainingCapacity();

        const n_nodes: usize = @intCast(self.max_nodes);
        self.buf_compute_meta.resize(allocator, c.ggml_tensor_overhead() * n_nodes + c.ggml_graph_overhead_custom(n_nodes, false)) catch oom();

        if (self.ctx_compute) |ctx| c.ggml_free(ctx);
        self.ctx_compute = c.ggml_init(.{
            .mem_size = self.buf_compute_meta.items.len,
            .mem_buffer = self.buf_compute_meta.items.ptr,
            .no_alloc = true,
        });
        self.gf = c.ggml_new_graph_custom(self.ctx_compute, n_nodes, false);
    }

    /// Ports `llm_graph_result`'s `set_inputs` (llama-graph.cpp:1353 @5266f24da).
    pub fn setInputs(self: *Result, ubatch: ?*const Ubatch) void {
        for (self.inputs.items) |inp| inp.vtable.setInput(inp.ptr, ubatch);
    }

    /// Ports `llm_graph_result`'s `set_outputs` (llama-graph.cpp:1359 @5266f24da).
    pub fn setOutputs(self: *Result, params: *const Params) void {
        if (self.t_logits) |t| c.ggml_set_output(t);
        if (self.t_embd) |t| c.ggml_set_output(t);
        if (self.t_embd_pooled) |t| c.ggml_set_output(t);
        if (self.t_h_nextn) |t| c.ggml_set_output(t);
        for (params.cparams.embeddings_layer_inp, 0..) |want, il| {
            if (want) {
                impl.assert(self.t_layer_inp[il] != null, "llama-graph.cpp", 1376, "t_layer_inp[il] != nullptr && \"layer input tensor is null\"");
                c.ggml_set_output(self.t_layer_inp[il]);
            }
        }
        for (self.t_sampled.items) |t| if (t) |x| c.ggml_set_output(x);
        for (self.t_sampled_probs.items) |t| if (t) |x| c.ggml_set_output(x);
        for (self.t_sampled_logits.items) |t| if (t) |x| c.ggml_set_output(x);
        for (self.t_candidates.items) |t| if (t) |x| c.ggml_set_output(x);
    }

    /// Ports `llm_graph_result`'s `can_reuse` (llama-graph.cpp:1403 @5266f24da):
    /// whether this graph can serve `params` once its inputs are updated.
    pub fn canReuse(self: *Result, params: *const Params) bool {
        const prev = if (self.params) |*p| p else return false;
        if (!prev.allowReuse(params)) {
            if (self.debug > 1) impl.logDebug("%s: cannot reuse graph due to incompatible graph parameters\n", .{"can_reuse"});
            return false;
        }
        if (self.debug > 1) impl.logDebug("%s: checking compatibility of %d inputs:\n", .{ "can_reuse", @as(c_int, @intCast(self.inputs.items.len)) });
        var res = true;
        for (self.inputs.items) |inp| {
            const cur = inp.vtable.canReuse(inp.ptr, params);
            if (self.debug > 1) impl.logDebug("%s: can_reuse = %d\n", .{ "placeholder", @as(c_int, @intFromBool(cur)) });
            res = res and cur;
        }
        if (self.debug > 0) impl.logDebug("%s: can reuse graph = %d\n", .{ "can_reuse", @as(c_int, @intFromBool(res)) });
        return res;
    }

    /// Ports `llm_graph_result`'s `add_input` (llama-graph.cpp:1435 @5266f24da):
    /// the result takes ownership.
    ///
    /// Return: `p`.
    pub fn addInput(self: *Result, comptime T: type, p: *T) *T {
        self.inputs.append(allocator, Input.of(T, p)) catch oom();
        return p;
    }

    /// Ports `llm_graph_result`'s `add_fused_node` (llama-graph.cpp:1440 @5266f24da).
    pub fn addFusedNode(self: *Result, node: FusedNode) void {
        self.fused_nodes.append(allocator, node) catch oom();
    }

    /// Ports `llm_graph_result`'s `set_params` (llama-graph.cpp:1444 @5266f24da):
    /// keep a copy of the parameters the graph was built with.
    pub fn setParams(self: *Result, params: *const Params) void {
        if (self.params) |*p| p.deinit();
        self.params = params.clone();
    }
};

/// Ports `llama_mul_mat_hadamard` (llama-impl.h:57 @5266f24da).
pub fn mulMatHadamard(ctx: *c.ggml_context, cur: *Tensor, rot: *Tensor) *Tensor {
    const n = rot.ne[0];
    var res: *Tensor = if (!c.ggml_is_contiguous(cur))
        c.ggml_cont_2d(ctx, cur, n, @divTrunc(c.ggml_nelements(cur), n))
    else
        c.ggml_reshape_2d(ctx, cur, n, @divTrunc(c.ggml_nelements(cur), n));
    res = c.ggml_mul_mat(ctx, rot, res);
    c.ggml_mul_mat_set_hint(res, c.GGML_HINT_SRC0_IS_HADAMARD);
    res = c.ggml_reshape_4d(ctx, res, cur.ne[0], cur.ne[1], cur.ne[2], cur.ne[3]);
    return res;
}

extern "c" fn log(x: f64) f64;

/// Ports `llama_relative_position_bucket` (llama-graph.cpp:3822 @5266f24da).
///
/// The C++ computes the large-distance bucket for every position, and for
/// a distance of 0 that is `floorf(-inf)` converted to `int32_t`, which is
/// undefined; the value is then never used. It is computed here only where
/// it is used, which gives the same answers.
pub fn relativePositionBucket(x: i32, y: i32, n_buckets_: u64, bidirectional: bool) i32 {
    // Move to hparams if a T5 variant appears with a different value.
    const max_distance: i64 = 128;
    var n_buckets = n_buckets_;
    if (bidirectional) n_buckets >>= 1;
    const max_exact: i64 = @intCast(n_buckets >> 1);

    var relative_position: i32 = x -% y;
    var relative_bucket: i32 = 0;
    if (bidirectional) {
        relative_bucket +%= @as(i32, @truncate(@as(i64, @bitCast(@as(u64, @intFromBool(relative_position > 0)) *% n_buckets))));
        relative_position = @intCast(@abs(relative_position));
    } else {
        relative_position = -@min(relative_position, 0);
    }

    if (relative_position < max_exact) return relative_bucket +% relative_position;

    // `logf` of a double narrows it; the product is a float; the division
    // by `log` is in double; `floorf` narrows again.
    const lf = impl.logf(@floatCast(1.0 * @as(f64, @floatFromInt(relative_position)) / @as(f64, @floatFromInt(max_exact))));
    const prod: f32 = lf * @as(f32, @floatFromInt(n_buckets -% @as(u64, @intCast(max_exact))));
    const v: f64 = @as(f64, @floatFromInt(max_exact)) + @as(f64, prod) / log(1.0 * @as(f64, @floatFromInt(max_distance)) / @as(f64, @floatFromInt(max_exact)));
    var relative_position_if_large: i32 = @intFromFloat(@floor(@as(f32, @floatCast(v))));
    relative_position_if_large = @min(relative_position_if_large, @as(i32, @truncate(@as(i64, @bitCast(n_buckets -% 1)))));
    return relative_bucket +% relative_position_if_large;
}

// -----------------------------------------------------------------------------
// Building

/// `llm_graph_get_rows_fn` (llama-graph.h:975 @5266f24da): how `build_rs`
/// gathers the states a ubatch reads. The C++'s is a `std::function`, and
/// Mamba's capture what the scan reads; `context` is that capture.
pub const GetRowsFn = struct {
    context: ?*const anyopaque = null,
    func: *const fn (context: ?*const anyopaque, ctx: *c.ggml_context, states: *Tensor, ids: *Tensor) *Tensor,

    /// Return: the gathered states, or whatever the closure builds from them.
    pub fn call(self: GetRowsFn, ctx: *c.ggml_context, states: *Tensor, ids: *Tensor) *Tensor {
        return self.func(self.context, ctx, states, ids);
    }
};

/// `build_rs`'s default, `ggml_get_rows` itself.
const ggml_get_rows_fn: GetRowsFn = .{ .func = ggmlGetRows };

fn ggmlGetRows(_: ?*const anyopaque, ctx: *c.ggml_context, states: *Tensor, ids: *Tensor) *Tensor {
    return c.ggml_get_rows(ctx, states, ids);
}

/// Mirrors `llm_graph_qkv` (llama-graph.h:977 @5266f24da).
pub const Qkv = struct {
    /// `[n_embd_head, n_head, n_tokens]`.
    q: *Tensor,
    /// `[n_embd_head, n_head_kv, n_tokens]`.
    k: *Tensor,
    /// `[n_embd_head, n_head_kv, n_tokens]`.
    v: *Tensor,
};

/// Ports `llm_graph_context` (llama-graph.h:983 @5266f24da): the state an
/// architecture's graph is built with, and the builders it shares.
pub const Builder = struct {
    arch: Arch,
    hparams: *const Hparams,
    cparams: *const Cparams,
    ubatch: *const Ubatch,

    n_embd: i64,
    n_layer: i64,
    n_layer_nextn: i64,
    n_rot: i64,
    /// The user's context size, which can differ from `n_ctx_train`.
    n_ctx: i64,
    n_head: i64,
    n_head_kv: i64,
    n_embd_head_k: i64,
    n_embd_k_gqa: i64,
    n_embd_head_v: i64,
    n_embd_v_gqa: i64,
    n_expert: i64,
    n_expert_used: i64,

    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    norm_eps: f32,
    norm_rms_eps: f32,

    n_tokens: i64,
    n_outputs: i64,
    /// For YaRN.
    n_ctx_orig: i32,

    pooling_type: c.enum_llama_pooling_type,
    rope_type: c.enum_llama_rope_type,

    sched: c.ggml_backend_sched_t,
    backend_cpu: c.ggml_backend_t,

    cvec: ?*const adapter.Cvec,
    loras: ?*const adapter.Loras,
    mctx: ?memory.Context,
    cross: ?*const Cross,
    samplers: *const SamplerMap,

    cb_func: Callback,
    res: *Result,
    ctx0: *c.ggml_context,
    gf: *c.ggml_cgraph,

    /// Ports the constructor of `llm_graph_context` (llama-graph.cpp:1452
    /// @5266f24da).
    ///
    /// Parameters:
    /// - `params`: what the graph is built from; borrowed for the builder's
    ///   lifetime, as the C++ holds references into it.
    ///
    /// Return: the builder.
    pub fn init(params: *const Params) Builder {
        const hp = params.hparams;
        const cp = &params.cparams;
        const res = params.res.?;
        res.setParams(params);
        return .{
            .arch = params.arch,
            .hparams = hp,
            .cparams = cp,
            .ubatch = &params.ubatch,
            .n_embd = hp.n_embd,
            .n_layer = hp.nLayer(),
            .n_layer_nextn = hp.n_layer_nextn,
            .n_rot = hp.nRot(0),
            .n_ctx = cp.n_ctx,
            .n_head = hp.nHead(0),
            .n_head_kv = hp.nHeadKv(0),
            .n_embd_head_k = hp.nEmbdHeadK(0),
            .n_embd_k_gqa = hp.nEmbdKGqa(0),
            .n_embd_head_v = hp.nEmbdHeadV(0),
            .n_embd_v_gqa = hp.nEmbdVGqa(0),
            .n_expert = hp.n_expert,
            .n_expert_used = if (cp.warmup) hp.n_expert else hp.nExpertUsed(0),
            .freq_base = cp.rope_freq_base,
            .freq_scale = cp.rope_freq_scale,
            .ext_factor = cp.yarn_ext_factor,
            .attn_factor = cp.yarn_attn_factor,
            .beta_fast = cp.yarn_beta_fast,
            .beta_slow = cp.yarn_beta_slow,
            .norm_eps = hp.f_norm_eps,
            .norm_rms_eps = hp.f_norm_rms_eps,
            .n_tokens = params.ubatch.n_tokens,
            .n_outputs = params.n_outputs,
            .n_ctx_orig = @bitCast(cp.n_ctx_orig_yarn),
            .pooling_type = cp.pooling_type,
            .rope_type = hp.rope_type,
            .sched = params.sched,
            .backend_cpu = params.backend_cpu,
            .cvec = params.cvec,
            .loras = params.loras,
            .mctx = params.mctx,
            .cross = params.cross,
            .samplers = &params.samplers,
            .cb_func = params.cb,
            .res = res,
            .ctx0 = res.getCtx(),
            .gf = res.getGf(),
        };
    }

    /// Ports `llm_graph_context`'s `cb` (llama-graph.cpp:1497 @5266f24da).
    pub fn cb(self: *const Builder, cur: *Tensor, name: [*:0]const u8, il: c_int) void {
        if (self.cb_func.func) |f| f(self.cb_func.ctx, self.ubatch, cur, name, il);
    }

    /// Ports `llm_graph_context`'s `build_cvec` (llama-graph.cpp:1505 @5266f24da).
    pub fn buildCvec(self: *const Builder, cur: *Tensor, il: c_int) *Tensor {
        return self.cvec.?.applyTo(self.ctx0, cur, il);
    }

    fn lorasOf(self: *const Builder) []const adapter.Loras.Entry {
        return if (self.loras) |l| l.items.items else &.{};
    }

    /// Ports `llm_graph_context`'s `build_lora_mm` (llama-graph.cpp:1511
    /// @5266f24da): a matrix product, with the per-tensor scale and every
    /// active LoRA's delta.
    pub fn buildLoraMm(self: *const Builder, w: *Tensor, cur: *Tensor, w_s: ?*Tensor) *Tensor {
        var res = c.ggml_mul_mat(self.ctx0, w, cur);
        if (w_s) |s| res = c.ggml_mul(self.ctx0, res, s);
        for (self.lorasOf()) |lora| {
            const lw = lora.lora.getWeight(w) orelse continue;
            const scale = lw.getScale(lora.lora.alpha, lora.scale);
            var ab_cur = c.ggml_mul_mat(self.ctx0, lw.b, c.ggml_mul_mat(self.ctx0, lw.a, cur));
            ab_cur = c.ggml_scale(self.ctx0, ab_cur, scale);
            res = c.ggml_add(self.ctx0, res, ab_cur);
        }
        return res;
    }

    /// Ports `llm_graph_context`'s `build_lora_mm_id` (llama-graph.cpp:1542 @5266f24da).
    pub fn buildLoraMmId(self: *const Builder, w: *Tensor, cur: *Tensor, ids: *Tensor, w_s: ?*Tensor) *Tensor {
        var res = c.ggml_mul_mat_id(self.ctx0, w, cur, ids);
        if (w_s) |ws| {
            const n_expert = ws.ne[0];
            const n_tokens = cur.ne[2];
            var s = c.ggml_reshape_3d(self.ctx0, ws, 1, n_expert, 1);
            s = c.ggml_repeat_4d(self.ctx0, s, 1, n_expert, n_tokens, 1);
            s = c.ggml_get_rows(self.ctx0, s, ids);
            res = c.ggml_mul(self.ctx0, res, s);
        }
        for (self.lorasOf()) |lora| {
            const lw = lora.lora.getWeight(w) orelse continue;
            const alpha = lora.lora.alpha;
            const rank: f32 = @floatFromInt(lw.b.?.ne[0]);
            const scale = if (alpha != 0) lora.scale * alpha / rank else lora.scale;
            var ab_cur = c.ggml_mul_mat_id(self.ctx0, lw.b, c.ggml_mul_mat_id(self.ctx0, lw.a, cur, ids), ids);
            ab_cur = c.ggml_scale(self.ctx0, ab_cur, scale);
            res = c.ggml_add(self.ctx0, res, ab_cur);
        }
        return res;
    }

    /// Ports `llm_graph_context`'s `build_norm` (llama-graph.cpp:1580 @5266f24da).
    pub fn buildNorm(self: *const Builder, cur_: *Tensor, mw: ?*Tensor, mb: ?*Tensor, ty: NormType, il: c_int) *Tensor {
        var cur = cur_;
        switch (ty) {
            .norm => cur = c.ggml_norm(self.ctx0, cur, self.hparams.f_norm_eps),
            .rms => cur = c.ggml_rms_norm(self.ctx0, cur, self.hparams.f_norm_rms_eps),
            .group => {
                cur = c.ggml_reshape_3d(self.ctx0, cur, cur.ne[0], 1, cur.ne[1]);
                cur = c.ggml_group_norm(self.ctx0, cur, @intCast(self.hparams.n_norm_groups), self.hparams.f_norm_group_eps);
                cur = c.ggml_reshape_2d(self.ctx0, cur, cur.ne[0], cur.ne[2]);
            },
        }
        if (mw != null or mb != null) self.cb(cur, "norm", il);
        if (mw) |w| {
            cur = c.ggml_mul(self.ctx0, cur, w);
            if (mb != null) self.cb(cur, "norm_w", il);
        }
        if (mb) |b| cur = c.ggml_add(self.ctx0, cur, b);
        return cur;
    }

    /// Ports `llm_graph_context`'s `build_qkv` (llama-graph.cpp:1616 @5266f24da):
    /// the Q, K and V projections, fused or separate, biased and clamped
    /// where the model says.
    pub fn buildQkv(self: *const Builder, layer: *const model_mod.Layer, cur: *Tensor, n_embd_head: i64, n_head: i64, n_head_kv: i64, il: c_int) Qkv {
        const ctx = self.ctx0;
        const n_embd_q = n_embd_head * n_head;
        const n_embd_kv = n_embd_head * n_head_kv;
        const clamp = self.hparams.f_clamp_kqv;
        var q: *Tensor = undefined;
        var k: *Tensor = undefined;
        var v: *Tensor = undefined;

        if (layer.wqkv) |wqkv| {
            var qkv = self.buildLoraMm(wqkv, cur, layer.wqkv_s);
            self.cb(qkv, "wqkv", il);
            if (layer.wqkv_b) |b| {
                qkv = c.ggml_add(ctx, qkv, b);
                self.cb(qkv, "wqkv_b", il);
            }
            if (clamp > 0.0) {
                qkv = c.ggml_clamp(ctx, qkv, -clamp, clamp);
                self.cb(qkv, "wqkv_clamped", il);
            }
            q = c.ggml_view_3d(ctx, qkv, n_embd_head, n_head, self.n_tokens, c.ggml_row_size(qkv.type, n_embd_head), qkv.nb[1], 0);
            k = c.ggml_view_3d(ctx, qkv, n_embd_head, n_head_kv, self.n_tokens, c.ggml_row_size(qkv.type, n_embd_head), qkv.nb[1], c.ggml_row_size(qkv.type, n_embd_q));
            v = c.ggml_view_3d(ctx, qkv, n_embd_head, n_head_kv, self.n_tokens, c.ggml_row_size(qkv.type, n_embd_head), qkv.nb[1], c.ggml_row_size(qkv.type, n_embd_q + n_embd_kv));
        } else {
            q = self.buildLoraMm(layer.wq.?, cur, layer.wq_s);
            self.cb(q, "Qcur", il);
            if (layer.wq_b) |b| {
                q = c.ggml_add(ctx, q, b);
                self.cb(q, "Qcur", il);
            }
            if (clamp > 0.0) {
                q = c.ggml_clamp(ctx, q, -clamp, clamp);
                self.cb(q, "Qcur_clamped", il);
            }
            k = self.buildLoraMm(layer.wk.?, cur, layer.wk_s);
            self.cb(k, "Kcur", il);
            if (layer.wk_b) |b| {
                k = c.ggml_add(ctx, k, b);
                self.cb(k, "Kcur", il);
            }
            if (clamp > 0.0) {
                k = c.ggml_clamp(ctx, k, -clamp, clamp);
                self.cb(k, "Kcur_clamped", il);
            }
            v = self.buildLoraMm(layer.wv.?, cur, layer.wv_s);
            self.cb(v, "Vcur", il);
            if (layer.wv_b) |b| {
                v = c.ggml_add(ctx, v, b);
                self.cb(v, "Vcur", il);
            }
            if (clamp > 0.0) {
                v = c.ggml_clamp(ctx, v, -clamp, clamp);
                self.cb(v, "Vcur_clamped", il);
            }
            q = c.ggml_reshape_3d(ctx, q, n_embd_head, n_head, self.n_tokens);
            k = c.ggml_reshape_3d(ctx, k, n_embd_head, n_head_kv, self.n_tokens);
            v = c.ggml_reshape_3d(ctx, v, n_embd_head, n_head_kv, self.n_tokens);
        }
        self.cb(q, "Qcur", il);
        self.cb(k, "Kcur", il);
        self.cb(v, "Vcur", il);
        return .{ .q = q, .k = k, .v = v };
    }

    fn hasLora(self: *const Builder, w: ?*Tensor) bool {
        const t = w orelse return false;
        for (self.lorasOf()) |lora| if (lora.lora.getWeight(t) != null) return true;
        return false;
    }

    /// Ports `llm_graph_context`'s `build_ffn` (llama-graph.cpp:1693 @5266f24da).
    pub fn buildFfn(
        self: *const Builder,
        cur_: *Tensor,
        up: ?*Tensor,
        up_b: ?*Tensor,
        up_s: ?*Tensor,
        gate: ?*Tensor,
        gate_b: ?*Tensor,
        gate_s: ?*Tensor,
        down: ?*Tensor,
        down_b: ?*Tensor,
        down_s: ?*Tensor,
        act_scales: ?*Tensor,
        type_op: FfnOp,
        type_gate_: FfnGate,
        il: c_int,
    ) *Tensor {
        const ctx = self.ctx0;
        var cur = cur_;
        var type_gate = type_gate_;
        const nvfp4 = c.GGML_TYPE_NVFP4;

        // NVFP4's `*_s` must not follow a LoRA residual or a bias.
        impl.assert(up_s == null or up_b == null or up == null or up.?.type != nvfp4, "llama-graph.cpp", 1724, "!up_s || !up_b || !up || up->type != GGML_TYPE_NVFP4");
        impl.assert(gate_s == null or gate_b == null or gate == null or gate.?.type != nvfp4, "llama-graph.cpp", 1725, "!gate_s || !gate_b || !gate || gate->type != GGML_TYPE_NVFP4");
        impl.assert(down_s == null or down_b == null or down == null or down.?.type != nvfp4, "llama-graph.cpp", 1726, "!down_s || !down_b || !down || down->type != GGML_TYPE_NVFP4");
        impl.assert(up_s == null or up == null or up.?.type != nvfp4 or !self.hasLora(up), "llama-graph.cpp", 1727, "!up_s || !up || up->type != GGML_TYPE_NVFP4 || !has_lora(up)");
        impl.assert(gate_s == null or gate == null or gate.?.type != nvfp4 or !self.hasLora(gate), "llama-graph.cpp", 1728, "!gate_s || !gate || gate->type != GGML_TYPE_NVFP4 || !has_lora(gate)");
        impl.assert(down_s == null or down == null or down.?.type != nvfp4 or !self.hasLora(down), "llama-graph.cpp", 1729, "!down_s || !down || down->type != GGML_TYPE_NVFP4 || !has_lora(down)");

        var tmp = if (up) |u| self.buildLoraMm(u, cur, null) else cur;
        self.cb(tmp, "ffn_up", il);
        if (up_b) |b| {
            tmp = c.ggml_add(ctx, tmp, b);
            self.cb(tmp, "ffn_up_b", il);
        }
        if (up_s) |s| {
            tmp = c.ggml_mul(ctx, tmp, s);
            self.cb(tmp, "ffn_up_s", il);
        }

        if (gate) |g| {
            switch (type_gate) {
                .seq => {
                    cur = self.buildLoraMm(g, tmp, null);
                    self.cb(cur, "ffn_gate", il);
                },
                .par => {
                    cur = self.buildLoraMm(g, cur, null);
                    self.cb(cur, "ffn_gate", il);
                },
            }
            if (gate_b) |b| {
                cur = c.ggml_add(ctx, cur, b);
                self.cb(cur, "ffn_gate_b", il);
            }
            if (gate_s) |s| {
                cur = c.ggml_mul(ctx, cur, s);
                self.cb(cur, "ffn_gate_s", il);
            }
        } else {
            cur = tmp;
        }

        switch (type_op) {
            .silu => silu: {
                if (gate != null and type_gate == .par) {
                    if (il >= 0) {
                        const limit = self.hparams.swiglu_clamp_shexp[@intCast(il)];
                        const eps: f32 = 1e-6;
                        if (limit > eps) {
                            if (self.arch == .deepseek4 or (self.arch == .dflash and self.hparams.dsv4_hc_mult > 0)) {
                                cur = c.ggml_swiglu_clamp(ctx, cur, tmp, limit);
                            } else {
                                tmp = c.ggml_clamp(ctx, tmp, -limit, limit);
                                self.cb(tmp, "ffn_up_clamped", il);
                                var gate_act = c.ggml_silu(ctx, cur);
                                self.cb(gate_act, "ffn_silu", il);
                                gate_act = c.ggml_clamp(ctx, gate_act, -std.math.inf(f32), limit);
                                self.cb(gate_act, "ffn_silu_clamped", il);
                                cur = c.ggml_mul(ctx, gate_act, tmp);
                            }
                            self.cb(cur, "ffn_swiglu_limited", il);
                            type_gate = .seq;
                            break :silu;
                        }
                    }
                    cur = c.ggml_swiglu_split(ctx, cur, tmp);
                    self.cb(cur, "ffn_swiglu", il);
                    type_gate = .seq;
                } else {
                    cur = c.ggml_silu(ctx, cur);
                    self.cb(cur, "ffn_silu", il);
                }
            },
            .gelu => {
                if (gate != null and type_gate == .par) {
                    cur = c.ggml_geglu_split(ctx, cur, tmp);
                    self.cb(cur, "ffn_geglu", il);
                    type_gate = .seq;
                } else {
                    cur = c.ggml_gelu(ctx, cur);
                    self.cb(cur, "ffn_gelu", il);
                    if (act_scales) |s| {
                        cur = c.ggml_div(ctx, cur, s);
                        self.cb(cur, "ffn_act", il);
                    }
                }
            },
            .relu => {
                if (gate != null and type_gate == .par) {
                    cur = c.ggml_reglu_split(ctx, cur, tmp);
                    self.cb(cur, "ffn_reglu", il);
                    type_gate = .seq;
                } else {
                    cur = c.ggml_relu(ctx, cur);
                    self.cb(cur, "ffn_relu", il);
                }
            },
            .relu_sqr => {
                cur = c.ggml_relu(ctx, cur);
                self.cb(cur, "ffn_relu", il);
                cur = c.ggml_sqr(ctx, cur);
                self.cb(cur, "ffn_sqr(relu)", il);
            },
            .swiglu => {
                cur = c.ggml_swiglu(ctx, cur);
                self.cb(cur, "ffn_swiglu", il);
            },
            .swiglu_oai_moe => {
                if (gate != null and type_gate == .par) {
                    // The same alpha and limit as gpt-oss.
                    cur = c.ggml_swiglu_oai(ctx, cur, tmp, 1.702, 7.0);
                    self.cb(cur, "ffn_swiglu_oai", il);
                    type_gate = .seq;
                } else {
                    impl.abort("llama-graph.cpp", 1847, "LLM_FFN_SWIGLU_OAI_MOE requires a parallel gate");
                }
            },
            .geglu => {
                cur = c.ggml_geglu(ctx, cur);
                self.cb(cur, "ffn_geglu", il);
            },
            .reglu => {
                cur = c.ggml_reglu(ctx, cur);
                self.cb(cur, "ffn_reglu", il);
            },
            .situ => impl.abort("llama-graph.cpp", 1860, "not yet supported"),
            .none => impl.abort("llama-graph.cpp", 1862, "fatal error"),
        }

        if (gate != null and type_gate == .par) {
            cur = c.ggml_mul(ctx, cur, tmp);
            self.cb(cur, "ffn_gate_par", il);
        }

        if (down) |d| {
            cur = self.buildLoraMm(d, cur, null);
            if (self.arch == .glm4 or self.arch == .glm4_moe or self.arch == .jais2) {
                // Half-precision accumulators lose these models.
                c.ggml_mul_mat_set_prec(cur, c.GGML_PREC_F32);
            }
        }
        if (down_b != null) self.cb(cur, "ffn_down", il);
        if (down_b) |b| cur = c.ggml_add(ctx, cur, b);
        if (down_s) |s| {
            cur = c.ggml_mul(ctx, cur, s);
            self.cb(cur, "ffn_down_s", il);
        }
        return cur;
    }

    /// The arguments past `cur` of `build_moe_ffn`, which has two
    /// overloads. The shorter `build_moe_ffn` (llama-graph.cpp:1894
    /// @5266f24da) has no biases and forwards `nullptr` for them; here the
    /// defaults are that `nullptr`, and the trailing defaults are the C++'s.
    pub const MoeFfn = struct {
        gate_inp: ?*Tensor,
        gate_inp_b: ?*Tensor = null,
        up_exps: ?*Tensor,
        up_exps_b: ?*Tensor = null,
        gate_exps: ?*Tensor,
        gate_exps_b: ?*Tensor = null,
        down_exps: *Tensor,
        down_exps_b: ?*Tensor = null,
        exp_probs_b: ?*Tensor = null,
        n_expert: i64,
        n_expert_used: i64,
        type_op: FfnOp,
        norm_w: bool,
        w_scale: f32,
        gating_op: hparams_mod.ExpertGatingFunc,
        probs_in: ?*Tensor = null,
        gate_up_exps: ?*Tensor = null,
        gate_up_exps_b: ?*Tensor = null,
        up_exps_s: ?*Tensor = null,
        gate_exps_s: ?*Tensor = null,
        down_exps_s: ?*Tensor = null,
        selected_experts_in: ?*Tensor = null,
    };

    /// Ports `llm_graph_context`'s `build_moe_ffn` (llama-graph.cpp:1938
    /// @5266f24da): route each token to its top experts, run them, and sum
    /// their outputs by the router's weights.
    ///
    /// Return: `[n_embd, n_tokens]`, contiguous.
    pub fn buildMoeFfn(self: *const Builder, cur_: *Tensor, a: MoeFfn, il: c_int) *Tensor {
        const ctx = self.ctx0;
        const hp = self.hparams;
        var cur = cur_;
        const n_embd = cur.ne[0];
        const n_tokens = cur.ne[1];
        // Llama 4 applies the sigmoid-ed weights before the FFN.
        const weight_before_ffn = self.arch == .llama4;

        var logits: *Tensor = undefined;
        if (a.probs_in) |p| {
            logits = p;
        } else {
            logits = self.buildLoraMm(a.gate_inp.?, cur, null); // [n_expert, n_tokens]
            if (a.gating_op == .sqrt_softplus) c.ggml_mul_mat_set_prec(logits, c.GGML_PREC_F32);
            self.cb(logits, "ffn_moe_logits", il);
        }

        if (a.gate_inp_b) |b| {
            logits = c.ggml_add(ctx, logits, b);
            self.cb(logits, "ffn_moe_logits_biased", il);
        }

        var probs: *Tensor = switch (a.gating_op) {
            .softmax => c.ggml_soft_max(ctx, logits), // [n_expert, n_tokens]
            .sigmoid => c.ggml_sigmoid(ctx, logits), // [n_expert, n_tokens]
            .softmax_weight => logits, // [n_expert, n_tokens]
            .sqrt_softplus => c.ggml_sqrt(ctx, c.ggml_softplus(ctx, logits)), // [n_expert, n_tokens]
            else => impl.abort("llama-graph.cpp", 2003, "fatal error"),
        };
        self.cb(probs, "ffn_moe_probs", il);

        // The expert-selection bias of DeepSeek V3: the probabilities stay
        // unbiased, since they become the expert weights.
        var selection_probs = probs;
        if (a.exp_probs_b) |b| {
            selection_probs = c.ggml_add(ctx, probs, b);
            self.cb(selection_probs, "ffn_moe_probs_biased", il);
        }

        // Llama 4 has no exp_probs_b, and its sigmoid comes after the top-k.
        if (self.arch == .llama4) selection_probs = logits;

        if (self.arch == .grovemoe) {
            selection_probs = c.ggml_sigmoid(ctx, logits); // [n_expert, n_tokens]
            self.cb(selection_probs, "ffn_moe_probs_biased", il);
        }

        // Select the top n_group_used expert groups.
        if (hp.n_expert_groups > 1 and n_tokens > 0) {
            const n_exp_per_group = @divTrunc(a.n_expert, @as(i64, hp.n_expert_groups));

            // Organize the experts into n_expert_groups.
            const selection_groups: *Tensor = c.ggml_reshape_3d(ctx, selection_probs, n_exp_per_group, hp.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

            var group_scores: *Tensor = c.ggml_argsort_top_k(ctx, selection_groups, 2); // [2, n_expert_groups, n_tokens]
            group_scores = c.ggml_get_rows(ctx, c.ggml_reshape_4d(ctx, selection_groups, 1, selection_groups.ne[0], selection_groups.ne[1], selection_groups.ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

            // The top n_group_used expert groups.
            group_scores = c.ggml_sum_rows(ctx, c.ggml_reshape_3d(ctx, group_scores, group_scores.ne[1], group_scores.ne[2], group_scores.ne[3])); // [1, n_expert_groups, n_tokens]
            group_scores = c.ggml_reshape_2d(ctx, group_scores, group_scores.ne[1], group_scores.ne[2]); // [n_expert_groups, n_tokens]

            const expert_groups: *Tensor = c.ggml_argsort_top_k(ctx, group_scores, @intCast(hp.n_group_used)); // [n_group_used, n_tokens]
            self.cb(expert_groups, "ffn_moe_group_topk", il);

            // Mask out the other groups.
            selection_probs = c.ggml_get_rows(ctx, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
            selection_probs = c.ggml_set_rows(ctx, c.ggml_fill(ctx, selection_groups, -std.math.inf(f32)), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
            selection_probs = c.ggml_reshape_2d(ctx, selection_probs, a.n_expert, n_tokens); // [n_expert, n_tokens]
            self.cb(selection_probs, "ffn_moe_probs_masked", il);
        }

        // Select the experts.
        var selected_experts: *Tensor = undefined;
        if (a.selected_experts_in) |s| {
            selected_experts = s;
        } else {
            selected_experts = c.ggml_argsort_top_k(ctx, selection_probs, @intCast(a.n_expert_used)); // [n_expert_used, n_tokens]
            const argsort: *Tensor = selected_experts.src[0];
            self.cb(argsort, "ffn_moe_argsort", il);
        }
        self.cb(selected_experts, "ffn_moe_topk", il);

        if (self.arch == .grovemoe and a.n_expert != hp.n_expert) {
            // The C++'s TODO: a scalar division once ggml has one.
            const f_sel: *Tensor = c.ggml_cast(ctx, selected_experts, c.GGML_TYPE_F32);
            selected_experts = c.ggml_cast(ctx, c.ggml_scale(ctx, f_sel, 1.0 / @as(f32, @floatFromInt(hp.n_group_experts))), c.GGML_TYPE_I32);
            probs = c.ggml_reshape_3d(ctx, probs, 1, hp.n_expert, n_tokens);
        } else {
            probs = c.ggml_reshape_3d(ctx, probs, 1, a.n_expert, n_tokens);
        }

        var weights: *Tensor = c.ggml_get_rows(ctx, probs, selected_experts); // [1, n_expert_used, n_tokens]
        self.cb(weights, "ffn_moe_weights", il);

        if (a.gating_op == .softmax_weight) {
            weights = c.ggml_reshape_2d(ctx, weights, a.n_expert_used, n_tokens);
            weights = c.ggml_soft_max(ctx, weights); // [n_expert_used, n_tokens]
            weights = c.ggml_reshape_3d(ctx, weights, 1, a.n_expert_used, n_tokens);
            self.cb(weights, "ffn_moe_weights_softmax", il);
        }

        if (a.norm_w) {
            weights = c.ggml_reshape_2d(ctx, weights, a.n_expert_used, n_tokens);

            var weights_sum: *Tensor = c.ggml_sum_rows(ctx, weights); // [1, n_tokens]
            self.cb(weights_sum, "ffn_moe_weights_sum", il);

            // Avoid a division by zero: clamp to the smallest normal F16.
            weights_sum = c.ggml_clamp(ctx, weights_sum, 6.103515625e-5, std.math.inf(f32));
            self.cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

            weights = c.ggml_div(ctx, weights, weights_sum); // [n_expert_used, n_tokens]
            self.cb(weights, "ffn_moe_weights_norm", il);

            weights = c.ggml_reshape_3d(ctx, weights, 1, a.n_expert_used, n_tokens);
        }
        if (a.w_scale != 0.0 and a.w_scale != 1.0) {
            weights = c.ggml_scale(ctx, weights, a.w_scale);
            self.cb(weights, "ffn_moe_weights_scaled", il);
        }

        // Early, so that the top-k MoE fusion can see it.
        c.ggml_build_forward_expand(self.gf, weights);

        cur = c.ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);

        if (weight_before_ffn) {
            // Repeat cur to [n_embd, n_expert_used, n_tokens].
            const repeated: *Tensor = c.ggml_repeat_4d(ctx, cur, n_embd, a.n_expert_used, n_tokens, 1);
            cur = c.ggml_mul(ctx, repeated, weights);
            self.cb(cur, "ffn_moe_weighted", il);
        }

        var up: *Tensor = undefined;

        if (a.gate_up_exps) |gue| {
            // Merged gate and up: one mul_mat_id, split into two views.
            var gate_up = self.buildLoraMmId(gue, cur, selected_experts, a.up_exps_s); // [n_ff*2, n_expert_used, n_tokens]
            self.cb(gate_up, "ffn_moe_gate_up", il);

            if (a.up_exps_s != null) self.cb(gate_up, "ffn_moe_gate_up_scaled", il);

            if (a.gate_up_exps_b) |b| {
                gate_up = c.ggml_add_id(ctx, gate_up, b, selected_experts);
                self.cb(gate_up, "ffn_moe_gate_up_biased", il);
            }

            const n_ff = @divTrunc(gate_up.ne[0], 2);
            cur = c.ggml_view_3d(ctx, gate_up, n_ff, gate_up.ne[1], gate_up.ne[2], gate_up.nb[1], gate_up.nb[2], 0);
            self.cb(cur, "ffn_moe_gate", il);
            up = c.ggml_view_3d(ctx, gate_up, n_ff, gate_up.ne[1], gate_up.ne[2], gate_up.nb[1], gate_up.nb[2], @as(usize, @intCast(n_ff)) * gate_up.nb[0]);
            self.cb(up, "ffn_moe_up", il);
        } else {
            // Separate gate and up.
            up = self.buildLoraMmId(a.up_exps.?, cur, selected_experts, a.up_exps_s); // [n_ff, n_expert_used, n_tokens]
            self.cb(up, "ffn_moe_up", il);

            if (a.up_exps_s != null) self.cb(up, "ffn_moe_up_scaled", il);

            if (a.up_exps_b) |b| {
                up = c.ggml_add_id(ctx, up, b, selected_experts);
                self.cb(up, "ffn_moe_up_biased", il);
            }

            if (a.gate_exps) |g| {
                cur = self.buildLoraMmId(g, cur, selected_experts, a.gate_exps_s); // [n_ff, n_expert_used, n_tokens]
                self.cb(cur, "ffn_moe_gate", il);
            } else {
                cur = up;
            }

            if (a.gate_exps_s != null) self.cb(cur, "ffn_moe_gate_scaled", il);

            if (a.gate_exps_b) |b| {
                cur = c.ggml_add_id(ctx, cur, b, selected_experts);
                self.cb(cur, "ffn_moe_gate_biased", il);
            }
        }

        const has_gate = a.gate_exps != null or a.gate_up_exps != null;

        switch (a.type_op) {
            .silu => silu: {
                if (a.gate_exps != null and il >= 0) {
                    const limit = hp.swiglu_clamp_exp[@intCast(il)];
                    const eps: f32 = 1e-6;
                    if (limit > eps) {
                        if (self.arch == .deepseek4 or (self.arch == .dflash and hp.dsv4_hc_mult > 0)) {
                            cur = c.ggml_swiglu_clamp(ctx, cur, up, limit);
                        } else {
                            up = c.ggml_clamp(ctx, up, -limit, limit);
                            self.cb(up, "ffn_moe_up_clamped", il);
                            var gate_act: *Tensor = c.ggml_silu(ctx, cur);
                            self.cb(gate_act, "ffn_moe_silu", il);
                            gate_act = c.ggml_clamp(ctx, gate_act, -std.math.inf(f32), limit);
                            self.cb(gate_act, "ffn_moe_silu_clamped", il);
                            cur = c.ggml_mul(ctx, gate_act, up);
                        }
                        self.cb(cur, "ffn_moe_swiglu_limited", il);
                        break :silu;
                    }
                }

                if (has_gate) {
                    cur = c.ggml_swiglu_split(ctx, cur, up);
                    self.cb(cur, "ffn_moe_swiglu", il);
                } else {
                    cur = c.ggml_silu(ctx, cur);
                    self.cb(cur, "ffn_moe_silu", il);
                }
            },
            .situ => {
                // situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
                impl.assert(has_gate, "llama-graph.cpp", 2200, "has_gate");
                const beta = hp.situ_beta;
                const lb = hp.situ_linear_beta;

                var act: *Tensor = c.ggml_scale(ctx, c.ggml_tanh(ctx, c.ggml_scale(ctx, cur, 1.0 / beta)), beta);
                act = c.ggml_mul(ctx, act, c.ggml_sigmoid(ctx, cur));
                if (lb > 0.0) up = c.ggml_scale(ctx, c.ggml_tanh(ctx, c.ggml_scale(ctx, up, 1.0 / lb)), lb);
                cur = c.ggml_mul(ctx, act, up);
                self.cb(cur, "ffn_moe_situ", il);
            },
            .gelu => {
                if (has_gate) {
                    cur = c.ggml_geglu_split(ctx, cur, up);
                    self.cb(cur, "ffn_moe_geglu", il);
                } else {
                    cur = c.ggml_gelu(ctx, cur);
                    self.cb(cur, "ffn_moe_gelu", il);
                }
            },
            .swiglu_oai_moe => {
                // The C++'s TODO: move these to the hparams.
                const alpha: f32 = 1.702;
                const limit: f32 = 7.0;
                cur = c.ggml_swiglu_oai(ctx, cur, up, alpha, limit);
                self.cb(cur, "ffn_moe_swiglu_oai", il);
            },
            .relu => {
                if (has_gate) {
                    cur = c.ggml_reglu_split(ctx, cur, up);
                    self.cb(cur, "ffn_moe_reglu", il);
                } else {
                    cur = c.ggml_relu(ctx, cur);
                    self.cb(cur, "ffn_moe_relu", il);
                }
            },
            .relu_sqr => {
                // The C++'s TODO: gated squared ReLU.
                if (has_gate) impl.abort("llama-graph.cpp", 2239, "fatal error: gated squared relu not implemented");
                cur = c.ggml_relu(ctx, cur);
                cur = c.ggml_sqr(ctx, cur);
                self.cb(cur, "ffn_moe_relu_sqr", il);
            },
            else => impl.abort("llama-graph.cpp", 2246, "fatal error"),
        }

        var experts = self.buildLoraMmId(a.down_exps, cur, selected_experts, a.down_exps_s); // [n_embd, n_expert_used, n_tokens]
        self.cb(experts, "ffn_moe_down", il);

        if (a.down_exps_s != null) self.cb(experts, "ffn_moe_down_scaled", il);

        if (a.down_exps_b) |b| {
            experts = c.ggml_add_id(ctx, experts, b, selected_experts);
            self.cb(experts, "ffn_moe_down_biased", il);
        }

        if (!weight_before_ffn) {
            experts = c.ggml_mul(ctx, experts, weights);
            self.cb(experts, "ffn_moe_weighted", il);
        }

        c.ggml_build_forward_expand(self.gf, experts);

        std.debug.assert(a.n_expert_used > 0);

        // The views go into the graph before the adds. Aggregated over the
        // layer's own `n_expert_used`, not the argument, so a warmup -- which
        // routes to every expert -- does not add a node per expert, and a
        // Puzzle-style model with per-layer counts sums the right number.
        // ref: https://github.com/ggml-org/llama.cpp/pull/14753
        const n_expert_used_il = hp.nExpertUsed(@intCast(il));
        var cur_experts: [hparams_mod.max_experts]*Tensor = undefined;
        for (0..n_expert_used_il) |i| {
            cur_experts[i] = c.ggml_view_2d(ctx, experts, n_embd, n_tokens, experts.nb[2], i * experts.nb[1]);
            c.ggml_build_forward_expand(self.gf, cur_experts[i]);
        }

        var moe_out = cur_experts[0];
        for (1..n_expert_used_il) |i| {
            moe_out = c.ggml_add(ctx, moe_out, cur_experts[i]);
            c.ggml_build_forward_expand(self.gf, moe_out);
        }

        // A single expert's view is not contiguous.
        if (n_expert_used_il == 1) moe_out = c.ggml_cont(ctx, moe_out);

        self.cb(moe_out, "ffn_moe_out", il);

        return moe_out;
    }

    // -------------------------------------------------------------------------
    // Inputs

    /// Ports `llm_graph_context`'s `build_inp_embd` (llama-graph.cpp:2303
    /// @5266f24da): the token embeddings, or the embeddings given, chosen by
    /// the batch at compute time.
    pub fn buildInpEmbd(self: *const Builder, tok_embd: *Tensor) *Tensor {
        const ctx = self.ctx0;
        const n_embd_inp: i64 = self.hparams.nEmbdInp();
        const n_embd: i64 = self.hparams.n_embd;
        std.debug.assert(n_embd_inp >= n_embd);

        const inp = allocator.create(InpEmbd) catch oom();
        inp.* = .{ .n_embd = n_embd_inp };

        inp.tokens = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, self.ubatch.n_tokens);
        self.cb(inp.tokens.?, "inp_tokens", -1);
        c.ggml_set_input(inp.tokens);
        self.res.t_inp_tokens = inp.tokens;

        inp.embd = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, n_embd_inp, self.ubatch.n_tokens);
        self.cb(inp.embd.?, "inp_embd", -1);
        c.ggml_set_input(inp.embd);

        // Select one of the two inputs, by the batch's contents.
        var inps: [2]*Tensor = undefined;

        // The token embeddings path.
        {
            var cur = c.ggml_get_rows(ctx, tok_embd, inp.tokens);
            for (self.lorasOf()) |lora| {
                const lw = lora.lora.getWeight(tok_embd) orelse continue;
                const scale = lw.getScale(lora.lora.alpha, lora.scale);
                const delta = c.ggml_scale(ctx, c.ggml_mul_mat(ctx, lw.b, c.ggml_get_rows(ctx, lw.a, inp.tokens)), scale);
                cur = c.ggml_add(ctx, cur, delta);
            }
            if (n_embd_inp != n_embd) cur = c.ggml_pad(ctx, cur, @intCast(@as(i64, self.hparams.nEmbdInp()) - n_embd), 0, 0, 0);
            inps[0] = cur;
        }
        // The vector embeddings path.
        inps[1] = inp.embd.?;

        std.debug.assert(c.ggml_are_same_shape(inps[0], inps[1]));
        std.debug.assert(c.ggml_are_same_stride(inps[0], inps[1]));

        var cur = c.ggml_build_forward_select(self.gf, @ptrCast(&inps), inps.len, if (self.ubatch.token != null) 0 else 1);
        if (n_embd_inp != n_embd) cur = c.ggml_view_2d(ctx, cur, n_embd, self.n_tokens, cur.*.nb[1], 0);

        self.res.t_inp_embd = cur;

        // Granite scales; for deepstack models, only token inputs are.
        if (self.hparams.f_embedding_scale != 0.0 and (self.ubatch.token != null or self.hparams.n_deepstack_layers == 0)) {
            if (!c.ggml_is_contiguous(cur)) cur = c.ggml_cont(ctx, cur);
            cur = c.ggml_scale(ctx, cur, self.hparams.f_embedding_scale);
        }

        self.cb(cur, "embd", -1);
        _ = self.res.addInput(InpEmbd, inp);

        // The embeddings are materialized in the graph immediately.
        c.ggml_build_forward_expand(self.gf, cur);
        return cur;
    }

    /// Ports `llm_graph_context`'s `build_inp_pos` (llama-graph.cpp:2392 @5266f24da).
    pub fn buildInpPos(self: *const Builder) *Tensor {
        const inp = allocator.create(InpPos) catch oom();
        inp.* = .{ .n_pos_per_embd = self.hparams.nPosPerEmbd() };
        inp.pos = c.ggml_new_tensor_1d(self.ctx0, c.GGML_TYPE_I32, self.n_tokens * self.hparams.nPosPerEmbd());
        c.ggml_set_input(inp.pos);
        _ = self.res.addInput(InpPos, inp);
        return inp.pos.?;
    }

    /// Ports `llm_graph_context`'s `build_inp_attn_scale` (llama-graph.cpp:2405
    /// @5266f24da): Llama 4's temperature-tuning scale, `[1, 1, n_tokens]`
    /// so it broadcasts over Q.
    pub fn buildInpAttnScale(self: *const Builder) *Tensor {
        const hp = self.hparams;
        const inp = allocator.create(InpAttnTemp) catch oom();
        inp.* = .{
            .n_attn_temp_floor_scale = hp.n_attn_temp_floor_scale,
            .f_attn_temp_scale = hp.f_attn_temp_scale,
            .f_attn_temp_offset = hp.f_attn_temp_offset,
        };

        // This needs to be 1x1xN for broadcasting.
        inp.attn_scale = c.ggml_new_tensor_3d(self.ctx0, c.GGML_TYPE_F32, 1, 1, self.n_tokens);
        c.ggml_set_input(inp.attn_scale);
        _ = c.ggml_set_name(inp.attn_scale, "attn_scale");

        _ = self.res.addInput(InpAttnTemp, inp);
        return inp.attn_scale.?;
    }

    /// Ports `llm_graph_context`'s `build_inp_out_ids` (llama-graph.cpp:2420
    /// @5266f24da). Always built, even when every token is an output, so the
    /// graph's topology does not depend on the output count.
    pub fn buildInpOutIds(self: *const Builder) *Tensor {
        const inp = allocator.create(InpOutIds) catch oom();
        inp.* = .{ .n_outputs = @intCast(self.n_outputs) };
        inp.out_ids = c.ggml_new_tensor_1d(self.ctx0, c.GGML_TYPE_I32, self.n_outputs);
        c.ggml_set_input(inp.out_ids);
        _ = self.res.addInput(InpOutIds, inp);
        return inp.out_ids.?;
    }

    /// Ports `llm_graph_context`'s `build_inp_mean` (llama-graph.cpp:2441 @5266f24da).
    pub fn buildInpMean(self: *const Builder) *Tensor {
        const inp = allocator.create(InpMean) catch oom();
        inp.* = .{ .cparams = self.cparams.* };
        inp.mean = c.ggml_new_tensor_2d(self.ctx0, c.GGML_TYPE_F32, self.n_tokens, self.ubatch.n_seqs_unq);
        c.ggml_set_input(inp.mean);
        _ = self.res.addInput(InpMean, inp);
        return inp.mean.?;
    }

    /// Ports `llm_graph_context`'s `build_inp_cls` (llama-graph.cpp:2454 @5266f24da).
    pub fn buildInpCls(self: *const Builder) *Tensor {
        const inp = allocator.create(InpCls) catch oom();
        inp.* = .{ .cparams = self.cparams.*, .arch = self.arch };
        inp.cls = c.ggml_new_tensor_1d(self.ctx0, c.GGML_TYPE_I32, self.ubatch.n_seqs_unq);
        c.ggml_set_input(inp.cls);
        _ = self.res.addInput(InpCls, inp);
        return inp.cls.?;
    }

    // -------------------------------------------------------------------------
    // Attention

    /// Ports `llm_graph_context`'s `build_attn_mha` (llama-graph.cpp:2536 @5266f24da).
    ///
    /// `n_kv_max` bounds the finite entries of every mask row, letting a
    /// backend run flash attention over them alone; 0 for a dense mask.
    pub fn buildAttnMha(self: *const Builder, q_: *Tensor, k_: *Tensor, v_: *Tensor, kq_b: ?*Tensor, kq_mask: ?*Tensor, sinks: ?*Tensor, v_mla: ?*Tensor, n_kv_max: i64, kq_scale: f32, il: c_int) *Tensor {
        const ctx = self.ctx0;
        var q = q_;
        var k = k_;
        var v = v_;
        const v_trans = v.nb[1] > v.nb[2];

        // Split the batch into streams if needed.
        const n_stream = k.ne[3];
        q = c.ggml_view_4d(ctx, q, q.ne[0], q.ne[1], @divTrunc(q.ne[2], n_stream), n_stream, q.nb[1], q.nb[2], q.nb[3] / @as(usize, @intCast(n_stream)), 0);

        q = c.ggml_permute(ctx, q, 0, 2, 1, 3);
        k = c.ggml_permute(ctx, k, 0, 2, 1, 3);
        v = c.ggml_permute(ctx, v, 0, 2, 1, 3);

        var cur: *Tensor = undefined;
        const use_flash_attn = self.cparams.flash_attn and kq_b == null;
        if (use_flash_attn) {
            impl.assert(kq_b == null, "llama-graph.cpp", 2562, "kq_b == nullptr && \"Flash attention does not support KQ bias yet\"");
            if (v_trans) v = c.ggml_transpose(ctx, v);
            // Without a KV cache, as in a non-causal embedding model.
            if (k.type == c.GGML_TYPE_F32) k = c.ggml_cast(ctx, k, c.GGML_TYPE_F16);
            if (v.type == c.GGML_TYPE_F32) v = c.ggml_cast(ctx, v, c.GGML_TYPE_F16);

            cur = c.ggml_flash_attn_ext(ctx, q, k, v, kq_mask, kq_scale, self.hparams.f_max_alibi_bias, if (self.hparams.attn_soft_cap) self.hparams.f_attn_logit_softcapping else 0.0);
            self.res.addFusedNode(.{ .op = .flash_attn, .tensor = cur, .il = il });

            c.ggml_flash_attn_ext_add_sinks(cur, sinks);
            impl.assert(n_kv_max >= 0 and n_kv_max <= std.math.maxInt(i32), "llama-graph.cpp", 2582, "n_kv_max >= 0 && n_kv_max <= INT32_MAX");
            c.ggml_flash_attn_ext_set_n_kv_max(cur, @intCast(n_kv_max));
            c.ggml_flash_attn_ext_set_prec(cur, c.GGML_PREC_F32);

            if (v_mla) |vm| {
                // As a matrix-matrix product with n_tokens in dimension 1;
                // the permutations only reinterpret the data.
                cur = c.ggml_permute(ctx, cur, 0, 2, 1, 3);
                cur = c.ggml_mul_mat(ctx, vm, cur);
                self.cb(cur, "fattn_mla", il);
                cur = c.ggml_permute(ctx, cur, 0, 2, 1, 3);
                cur = c.ggml_cont(ctx, cur);
            }
            cur = c.ggml_reshape_2d(ctx, cur, cur.ne[0] * cur.ne[1], cur.ne[2] * cur.ne[3]);
        } else {
            var kq = c.ggml_mul_mat(ctx, k, q);
            self.cb(kq, "kq", il);
            // This op needs a high floating-point range.
            c.ggml_mul_mat_set_prec(kq, c.GGML_PREC_F32);

            if (self.arch == .grok) {
                // kq = 30 * tanh(kq / 30), after the output multiplier.
                kq = c.ggml_tanh(ctx, c.ggml_scale(ctx, kq, self.hparams.f_attn_out_scale / self.hparams.f_attn_logit_softcapping));
                self.cb(kq, "kq_tanh", il);
                kq = c.ggml_scale(ctx, kq, self.hparams.f_attn_logit_softcapping);
                self.cb(kq, "kq_scaled", il);
            }
            if (self.hparams.attn_soft_cap) {
                kq = c.ggml_scale(ctx, kq, 1.0 / self.hparams.f_attn_logit_softcapping);
                self.cb(kq, "kq_scaled_1", il);
                kq = c.ggml_tanh(ctx, kq);
                self.cb(kq, "kq_tanh", il);
                kq = c.ggml_scale(ctx, kq, self.hparams.f_attn_logit_softcapping);
                self.cb(kq, "kq_scaled_2", il);
            }
            if (kq_b) |b| {
                kq = c.ggml_add(ctx, kq, b);
                self.cb(kq, "kq_plus_kq_b", il);
            }

            kq = c.ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, self.hparams.f_max_alibi_bias);
            c.ggml_soft_max_add_sinks(kq, sinks);
            self.cb(kq, "kq_soft_max", il);

            if (!v_trans) {
                // Avoid this branch.
                v = c.ggml_cont(ctx, c.ggml_transpose(ctx, v));
                self.cb(v, "v_cont", il);
            }

            var kqv = c.ggml_mul_mat(ctx, v, kq);
            self.cb(kqv, "kqv", il);

            // MLA with absorption decompresses from MQA back to MHA.
            if (v_mla) |vm| {
                kqv = c.ggml_mul_mat(ctx, vm, kqv);
                self.cb(kqv, "kqv_mla", il);
            }

            cur = c.ggml_permute(ctx, kqv, 0, 2, 1, 3);
            // Recombine the streams.
            cur = c.ggml_cont_2d(ctx, cur, cur.ne[0] * cur.ne[1], cur.ne[2] * cur.ne[3]);

            if (!self.cparams.offload_kqv) {
                // Every node between the KV store and the output runs on the CPU.
                c.ggml_backend_sched_set_tensor_backend(self.sched, cur, self.backend_cpu);
            }
        }

        c.ggml_build_forward_expand(self.gf, cur);
        return cur;
    }

    /// Ports `build_attn_inp_kv_impl` (llama-graph.cpp:2751 @5266f24da).
    fn buildAttnInpKvImpl(self: *const Builder, mctx_cur: *const kv_cache.KvCacheContext) *InpAttnKv {
        const inp = allocator.create(InpAttnKv) catch oom();
        inp.* = .{ .hparams = self.hparams, .cparams = self.cparams.*, .mctx = mctx_cur };
        impl.assert(@as(hparams_mod.SwaType, @fromBackingInt(@intCast(self.hparams.swa_type))) == .none, "llama-graph.cpp", 2761, "hparams.swa_type == LLAMA_SWA_TYPE_NONE && \"Use llama_kv_cache_iswa for SWA\"");
        inp.self_k_idxs = mctx_cur.buildInputKIdxs(self.ctx0, self.ubatch);
        inp.self_v_idxs = mctx_cur.buildInputVIdxs(self.ctx0, self.ubatch);
        inp.self_kq_mask = buildAttnInpKqMask(self.ctx0, mctx_cur, self.ubatch, self.cparams);
        inp.self_kq_mask_cnv = inp.self_kq_mask;
        inp.self_k_rot = mctx_cur.buildInputKRot(self.ctx0);
        inp.self_v_rot = mctx_cur.buildInputVRot(self.ctx0);
        return inp;
    }

    /// Ports `llm_graph_context`'s `build_attn_inp_kv` (llama-graph.cpp:2776 @5266f24da).
    pub fn buildAttnInpKv(self: *const Builder) *InpAttnKv {
        const mctx_cur = self.mctx.?.cast(kv_cache.KvCacheContext);
        return self.res.addInput(InpAttnKv, self.buildAttnInpKvImpl(mctx_cur));
    }

    /// Ports `llm_graph_context`'s `build_attn_inp_kv_msa` (llama-graph.cpp:3293
    /// @5266f24da): the base cache's input, and the indexer cache's
    /// destination indices when MSA is on.
    pub fn buildAttnInpKvMsa(self: *const Builder, msa_enabled: bool) *InpAttnKvMsa {
        const mctx_cur = self.mctx.?.cast(kv_cache_msa.MsaContext);

        const inp = allocator.create(InpAttnKvMsa) catch oom();
        inp.* = .{ .base = .{ .hparams = self.hparams, .cparams = self.cparams.*, .mctx = mctx_cur.getBase() }, .mctx_msa = mctx_cur };

        const mctx_base = mctx_cur.getBase();
        const mctx_idx = mctx_cur.getIdx();

        {
            impl.assert(@as(hparams_mod.SwaType, @fromBackingInt(@intCast(self.hparams.swa_type))) == .none, "llama-graph.cpp", 3302, "hparams.swa_type == LLAMA_SWA_TYPE_NONE && \"Use llama_kv_cache_iswa for SWA\"");

            inp.base.self_k_idxs = mctx_base.buildInputKIdxs(self.ctx0, self.ubatch);
            inp.base.self_v_idxs = mctx_base.buildInputVIdxs(self.ctx0, self.ubatch);

            inp.base.self_kq_mask = buildAttnInpKqMask(self.ctx0, mctx_base, self.ubatch, self.cparams);
            inp.base.self_kq_mask_cnv = inp.base.self_kq_mask;
        }

        inp.base.self_k_rot = mctx_base.buildInputKRot(self.ctx0);
        inp.base.self_v_rot = mctx_base.buildInputVRot(self.ctx0);

        if (msa_enabled) {
            inp.self_k_idxs_idx = mctx_idx.buildInputKIdxs(self.ctx0, self.ubatch);
        }

        return self.res.addInput(InpAttnKvMsa, inp);
    }

    /// `llm_graph_context`'s `build_attn`: the C++ overloads it on the
    /// input's type, and this dispatches the same way, so an architecture
    /// built for either memory calls it alike.
    ///
    /// Return: the attention output, projected by `wo` when given.
    pub fn buildAttn(
        self: *const Builder,
        inp: anytype,
        wo: ?*Tensor,
        wo_b: ?*Tensor,
        wo_s: ?*Tensor,
        q_cur: *Tensor,
        k_cur: ?*Tensor,
        v_cur: ?*Tensor,
        kq_b: ?*Tensor,
        sinks: ?*Tensor,
        v_mla: ?*Tensor,
        kq_scale: f32,
        il: c_int,
    ) *Tensor {
        return switch (@TypeOf(inp)) {
            *InpAttnKv => self.buildAttnKv(inp, wo, wo_b, wo_s, q_cur, k_cur.?, v_cur.?, kq_b, sinks, v_mla, kq_scale, il),
            *InpAttnNoCache => self.buildAttnNoCache(inp, wo, wo_b, wo_s, q_cur, k_cur.?, v_cur.?, kq_b, sinks, v_mla, kq_scale, il),
            *InpAttnK => self.buildAttnK(inp, wo, wo_b, wo_s, q_cur, k_cur.?, v_cur.?, kq_b, sinks, v_mla, kq_scale, il),
            *InpAttnCross => self.buildAttnCross(inp, wo, wo_b, wo_s, q_cur, k_cur.?, v_cur.?, kq_b, sinks, v_mla, kq_scale, il),
            *InpAttnKvIswa => self.buildAttnKvIswa(inp, wo, wo_b, wo_s, q_cur, k_cur, v_cur, kq_b, sinks, v_mla, kq_scale, il),
            // `llm_graph_input_attn_kv_msa` derives from `llm_graph_input_attn_kv`.
            *InpAttnKvMsa => self.buildAttnKv(&inp.base, wo, wo_b, wo_s, q_cur, k_cur.?, v_cur.?, kq_b, sinks, v_mla, kq_scale, il),
            *InpAttnKIswa => self.buildAttnKIswa(inp, wo, wo_b, wo_s, q_cur, k_cur, v_cur.?, kq_b, sinks, v_mla, kq_scale, il),
            *InpAttnKDsa => @compileError("the DSA overload of build_attn takes the indexer's top_k: call buildAttnDsa"),
            else => @compileError("build_attn has no overload for " ++ @typeName(@TypeOf(inp))),
        };
    }

    /// Ports `llm_graph_context`'s `build_attn` over a KV cache
    /// (llama-graph.cpp:2784 @5266f24da): store K and V, attend, project.
    fn buildAttnKv(
        self: *const Builder,
        inp: *InpAttnKv,
        wo: ?*Tensor,
        wo_b: ?*Tensor,
        wo_s: ?*Tensor,
        q_cur_: *Tensor,
        k_cur_: *Tensor,
        v_cur_: *Tensor,
        kq_b: ?*Tensor,
        sinks: ?*Tensor,
        v_mla: ?*Tensor,
        kq_scale: f32,
        il: c_int,
    ) *Tensor {
        const ctx = self.ctx0;
        impl.assert(v_mla == null, "llama-graph.cpp", 2797, "v_mla == nullptr");
        var q_cur = q_cur_;
        var k_cur = k_cur_;
        var v_cur = v_cur_;

        if (inp.self_k_rot) |r| {
            q_cur = mulMatHadamard(ctx, q_cur, r);
            k_cur = mulMatHadamard(ctx, k_cur, r);
        }
        if (inp.self_v_rot) |r| v_cur = mulMatHadamard(ctx, v_cur, r);

        // Added together so they are not reordered, which reduces the
        // number of splits; k last, so RoPE can fuse into the cache write.
        c.ggml_build_forward_expand(self.gf, q_cur);
        c.ggml_build_forward_expand(self.gf, v_cur);
        c.ggml_build_forward_expand(self.gf, k_cur);

        const mctx_cur = inp.mctx;

        // Store to the KV cache.
        c.ggml_build_forward_expand(self.gf, mctx_cur.cpyK(ctx, k_cur, inp.getKIdxs(), il));
        c.ggml_build_forward_expand(self.gf, mctx_cur.cpyV(ctx, v_cur, inp.getVIdxs(), il));

        const kq_mask = inp.getKqMask();
        const k = mctx_cur.getK(ctx, il);
        const v = mctx_cur.getV(ctx, il);

        var cur = self.buildAttnMha(q_cur, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
        self.cb(cur, "kqv_out", il);

        if (inp.self_v_rot) |r| cur = mulMatHadamard(ctx, cur, r);

        if (wo) |w| {
            if (self.arch == .glm4 or self.arch == .glm4_moe or self.arch == .jais2) {
                cur = self.buildLoraMm(w, cur, null);
                c.ggml_mul_mat_set_prec(cur, c.GGML_PREC_F32);
                if (wo_s) |s| cur = c.ggml_mul(ctx, cur, s);
            } else {
                cur = self.buildLoraMm(w, cur, wo_s);
            }
        }
        if (wo_b) |b| cur = c.ggml_add(ctx, cur, b);
        return cur;
    }

    /// Ports `llm_graph_context`'s `build_attn_inp_no_cache` (llama-graph.cpp:2674
    /// @5266f24da).
    pub fn buildAttnInpNoCache(self: *const Builder) *InpAttnNoCache {
        const inp = allocator.create(InpAttnNoCache) catch oom();
        inp.* = .{ .hparams = self.hparams, .cparams = self.cparams.*, .debug = inputDebug() };

        // Flash attention needs an F16 mask.
        const type_mask: c.enum_ggml_type = if (self.cparams.flash_attn) c.GGML_TYPE_F16 else c.GGML_TYPE_F32;

        // Without a KV cache, the number of KV values is the number of
        // tokens in the batch.
        inp.self_kq_mask = c.ggml_new_tensor_4d(self.ctx0, type_mask, self.n_tokens, self.n_tokens, 1, 1);
        c.ggml_set_input(inp.self_kq_mask);

        inp.self_kq_mask_cnv = inp.self_kq_mask;

        if (@as(hparams_mod.SwaType, @fromBackingInt(@intCast(self.hparams.swa_type))) != .none) {
            inp.self_kq_mask_swa = c.ggml_new_tensor_4d(self.ctx0, type_mask, self.n_tokens, self.n_tokens, 1, 1);
            c.ggml_set_input(inp.self_kq_mask_swa);

            inp.self_kq_mask_swa_cnv = inp.self_kq_mask_swa;
        } else {
            inp.self_kq_mask_swa = null;
            inp.self_kq_mask_swa_cnv = null;
        }

        return self.res.addInput(InpAttnNoCache, inp);
    }

    /// Ports `llm_graph_context`'s `build_attn` without a cache
    /// (llama-graph.cpp:2699 @5266f24da): attention within the batch.
    fn buildAttnNoCache(
        self: *const Builder,
        inp: *InpAttnNoCache,
        wo: ?*Tensor,
        wo_b: ?*Tensor,
        wo_s: ?*Tensor,
        q_cur: *Tensor,
        k_cur: *Tensor,
        v_cur: *Tensor,
        kq_b: ?*Tensor,
        sinks: ?*Tensor,
        v_mla: ?*Tensor,
        kq_scale: f32,
        il: c_int,
    ) *Tensor {
        // Added together so they are not reordered, which reduces the
        // number of splits.
        c.ggml_build_forward_expand(self.gf, q_cur);
        c.ggml_build_forward_expand(self.gf, k_cur);
        c.ggml_build_forward_expand(self.gf, v_cur);

        const is_swa = self.hparams.isSwa(@intCast(il));

        const kq_mask = if (is_swa) inp.getKqMaskSwa() else inp.getKqMask();

        // [TAG_NO_CACHE_PAD] The C++'s TODO: with equal sequences the three
        // tensors could be split into a stream each, which may not be worth it
        // (https://github.com/ggml-org/llama.cpp/pull/15636).

        const q = q_cur;
        const k = k_cur;
        const v = v_cur;

        var cur = self.buildAttnMha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
        self.cb(cur, "kqv_out", il);

        if (wo) |w| cur = self.buildLoraMm(w, cur, wo_s);

        if (wo_b) |b| cur = c.ggml_add(self.ctx0, cur, b);

        return cur;
    }

    /// Ports `build_attn_inp_k_impl` (llama-graph.cpp:2859 @5266f24da).
    fn buildAttnInpKImpl(self: *const Builder, mctx_cur: *const kv_cache.KvCacheContext) *InpAttnK {
        const inp = allocator.create(InpAttnK) catch oom();
        inp.* = .{ .hparams = self.hparams, .cparams = self.cparams.*, .mctx = mctx_cur };

        impl.assert(@as(hparams_mod.SwaType, @fromBackingInt(@intCast(self.hparams.swa_type))) == .none, "llama-graph.cpp", 2869, "hparams.swa_type == LLAMA_SWA_TYPE_NONE && \"Use llama_kv_cache_iswa for SWA\"");

        inp.self_k_idxs = mctx_cur.buildInputKIdxs(self.ctx0, self.ubatch);

        inp.self_kq_mask = buildAttnInpKqMask(self.ctx0, mctx_cur, self.ubatch, self.cparams);
        inp.self_kq_mask_cnv = inp.self_kq_mask;

        return inp;
    }

    /// Ports `llm_graph_context`'s `build_attn_inp_k` (llama-graph.cpp:2880 @5266f24da).
    pub fn buildAttnInpK(self: *const Builder) *InpAttnK {
        const mctx_cur = self.mctx.?.cast(kv_cache.KvCacheContext);
        return self.res.addInput(InpAttnK, self.buildAttnInpKImpl(mctx_cur));
    }

    /// Ports `llm_graph_context`'s `build_attn` over a K-only cache
    /// (llama-graph.cpp:2888 @5266f24da): V is a view of the cached K, as
    /// wide as `v_cur`.
    fn buildAttnK(
        self: *const Builder,
        inp: *InpAttnK,
        wo: ?*Tensor,
        wo_b: ?*Tensor,
        wo_s: ?*Tensor,
        q_cur: *Tensor,
        k_cur: *Tensor,
        v_cur: *Tensor,
        kq_b: ?*Tensor,
        sinks: ?*Tensor,
        v_mla: ?*Tensor,
        kq_scale: f32,
        il: c_int,
    ) *Tensor {
        const ctx = self.ctx0;

        // Added together so they are not reordered, which reduces the
        // number of splits; k last, so RoPE can fuse into the cache write.
        c.ggml_build_forward_expand(self.gf, q_cur);
        c.ggml_build_forward_expand(self.gf, v_cur);
        c.ggml_build_forward_expand(self.gf, k_cur);

        const mctx_cur = inp.mctx;

        // Store to the KV cache.
        c.ggml_build_forward_expand(self.gf, mctx_cur.cpyK(ctx, k_cur, inp.getKIdxs(), il));

        const kq_mask = inp.getKqMask();

        const q = q_cur;
        const k = mctx_cur.getK(ctx, il);
        const v: *Tensor = c.ggml_view_4d(ctx, k, v_cur.ne[0], k.ne[1], k.ne[2], k.ne[3], k.nb[1], k.nb[2], k.nb[3], 0);

        var cur = self.buildAttnMha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
        self.cb(cur, "kqv_out", il);

        if (wo) |w| {
            if (self.arch == .glm4 or self.arch == .glm4_moe) {
                // GLM 4 and its MoE lose precision with half-precision accumulators.
                cur = self.buildLoraMm(w, cur, null);
                c.ggml_mul_mat_set_prec(cur, c.GGML_PREC_F32);
                if (wo_s) |s| cur = c.ggml_mul(ctx, cur, s);
            } else {
                cur = self.buildLoraMm(w, cur, wo_s);
            }
        }

        if (wo_b) |b| cur = c.ggml_add(ctx, cur, b);

        return cur;
    }

    /// Ports `llm_graph_context`'s `build_attn` over a DSA cache
    /// (llama-graph.cpp:2946 @5266f24da): the K-only attention of
    /// `build_attn` over `llm_graph_input_attn_k`, with the mask narrowed to
    /// the keys the lightning indexer chose.
    ///
    /// Parameters:
    /// - `top_k`: `[n_indexer_top_k, n_tokens]`, the chosen key positions.
    ///
    /// Return: the attention output, projected by `wo` when given.
    pub fn buildAttnDsa(
        self: *const Builder,
        inp: *InpAttnKDsa,
        wo: ?*Tensor,
        wo_b: ?*Tensor,
        wo_s: ?*Tensor,
        q_cur: *Tensor,
        k_cur: *Tensor,
        v_cur: *Tensor,
        kq_b: ?*Tensor,
        sinks: ?*Tensor,
        v_mla: ?*Tensor,
        top_k: *Tensor,
        kq_scale: f32,
        il: c_int,
    ) *Tensor {
        const ctx = self.ctx0;

        // Added together so they are not reordered, which reduces the
        // number of splits; k last, so RoPE can fuse into the cache write.
        c.ggml_build_forward_expand(self.gf, q_cur);
        c.ggml_build_forward_expand(self.gf, v_cur);
        c.ggml_build_forward_expand(self.gf, k_cur);

        const mctx_cur = inp.mctx.getMla();

        // Store to the KV cache.
        c.ggml_build_forward_expand(self.gf, mctx_cur.cpyK(ctx, k_cur, inp.getKIdxsMla(), il));

        const kq_mask = inp.getKqMaskMla().?;

        // A new KQ mask, filled with -INFINITY.
        var kq_mask_all: *Tensor = c.ggml_fill(ctx, kq_mask, -std.math.inf(f32));

        // Reshaped into rows of size 1:
        // `[n_kv, n_batch, 1, n_stream]` -> `[1, n_kv, n_batch, n_stream]`.
        kq_mask_all = c.ggml_view_4d(ctx, kq_mask_all, 1, kq_mask_all.ne[0], kq_mask_all.ne[1], kq_mask_all.ne[3], kq_mask_all.nb[0], kq_mask_all.nb[1], kq_mask_all.nb[2], 0);

        // The top-k indices reshaped:
        // `[n_top_k, n_batch, 1, n_stream]` -> `[n_top_k, n_batch, n_stream, 1]`.
        const top_k_3d: *Tensor = c.ggml_view_4d(ctx, top_k, top_k.ne[0], top_k.ne[1], top_k.ne[3], 1, top_k.nb[1], top_k.nb[2], @as(usize, @intCast(top_k.ne[3])) * top_k.nb[3], 0);

        // Zeros in rows of size 1, `[1, n_top_k, n_batch, n_stream]`: the
        // values that unmask the top-k elements.
        var zeros: *Tensor = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, 1, top_k_3d.ne[0], top_k_3d.ne[1], top_k_3d.ne[2]);
        zeros = c.ggml_fill(ctx, zeros, 0.0);

        // Unmask the elements at the top-k indices:
        // `ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])`.
        var kq_mask_top_k: *Tensor = c.ggml_set_rows(ctx, kq_mask_all, zeros, top_k_3d);

        // Restore the KQ mask's shape:
        // `[1, n_kv, n_batch, n_stream]` -> `[n_kv, n_batch, 1, n_stream]`.
        kq_mask_top_k = c.ggml_view_4d(ctx, kq_mask_top_k, kq_mask_top_k.ne[1], kq_mask_top_k.ne[2], 1, kq_mask_top_k.ne[3], kq_mask_top_k.nb[2], kq_mask_top_k.nb[3], kq_mask_top_k.nb[3], 0);

        // Combined with the original KQ mask.
        kq_mask_top_k = c.ggml_add(ctx, kq_mask_top_k, kq_mask);

        const q = q_cur;
        const k = mctx_cur.getK(ctx, il);
        const v: *Tensor = c.ggml_view_4d(ctx, k, v_cur.ne[0], k.ne[1], k.ne[2], k.ne[3], k.nb[1], k.nb[2], k.nb[3], 0);

        var cur = self.buildAttnMha(q, k, v, kq_b, kq_mask_top_k, sinks, v_mla, top_k.ne[0], kq_scale, il);
        self.cb(cur, "kqv_out", il);

        if (wo) |w| cur = self.buildLoraMm(w, cur, wo_s);

        if (wo_b) |b| cur = c.ggml_add(ctx, cur, b);

        return cur;
    }

    /// Ports `build_attn_inp_k_dsa_impl` (llama-graph.cpp:3235 @5266f24da).
    fn buildAttnInpKDsaImpl(self: *const Builder, mctx_cur: *const kv_cache_dsa.DsaContext) *InpAttnKDsa {
        const inp = allocator.create(InpAttnKDsa) catch oom();
        inp.* = .{ .hparams = self.hparams, .cparams = self.cparams.*, .mctx = mctx_cur };

        {
            inp.self_k_idxs_mla = mctx_cur.getMla().buildInputKIdxs(self.ctx0, self.ubatch);

            inp.self_kq_mask_mla = buildAttnInpKqMask(self.ctx0, mctx_cur.getMla(), self.ubatch, self.cparams);
            inp.self_kq_mask_mla_cnv = inp.self_kq_mask_mla;
        }

        {
            inp.self_k_idxs_lid = mctx_cur.getLid().buildInputKIdxs(self.ctx0, self.ubatch);

            // The mask's type follows the fused lightning indexer's use,
            // which requires an F16 mask.
            var cparams_copy = self.cparams.*;
            cparams_copy.flash_attn = self.cparams.fused_lid;

            inp.self_kq_mask_lid = buildAttnInpKqMask(self.ctx0, mctx_cur.getLid(), self.ubatch, &cparams_copy);
            inp.self_kq_mask_lid_cnv = inp.self_kq_mask_lid;

            inp.self_k_rot_lid = mctx_cur.getLid().buildInputKRot(self.ctx0);
        }

        return inp;
    }

    /// Ports `llm_graph_context`'s `build_attn_inp_k_dsa` (llama-graph.cpp:3267 @5266f24da).
    pub fn buildAttnInpKDsa(self: *const Builder) *InpAttnKDsa {
        const mctx_cur = self.mctx.?.cast(kv_cache_dsa.DsaContext);
        return self.res.addInput(InpAttnKDsa, self.buildAttnInpKDsaImpl(mctx_cur));
    }

    /// Ports `llm_graph_context`'s `build_attn_inp_k_dsa_iswa` (llama-graph.cpp:3275
    /// @5266f24da).
    pub fn buildAttnInpKDsaIswa(self: *const Builder) *InpAttnKDsaIswa {
        const mctx_cur = self.mctx.?.cast(kv_cache_dsa_iswa.DsaIswaContext);

        const inp_dsa = self.buildAttnInpKDsaImpl(mctx_cur.getDsa());

        // `build_attn_inp_k_impl` rejects SWA caches, so the input is built
        // here directly.
        const inp_swa = allocator.create(InpAttnK) catch oom();
        inp_swa.* = .{ .hparams = self.hparams, .cparams = self.cparams.*, .mctx = mctx_cur.getSwa() };

        inp_swa.self_k_idxs = mctx_cur.getSwa().buildInputKIdxs(self.ctx0, self.ubatch);

        inp_swa.self_kq_mask = buildAttnInpKqMask(self.ctx0, mctx_cur.getSwa(), self.ubatch, self.cparams);
        inp_swa.self_kq_mask_cnv = inp_swa.self_kq_mask;

        const inp = allocator.create(InpAttnKDsaIswa) catch oom();
        inp.* = .{ .inp_dsa = inp_dsa, .inp_swa = inp_swa, .mctx = mctx_cur };

        return self.res.addInput(InpAttnKDsaIswa, inp);
    }

    /// Ports `llm_graph_context`'s `build_inp_cross_embd` (llama-graph.cpp:2467
    /// @5266f24da): the encoder's output when there is one, else a tensor
    /// of its worst-case shape.
    pub fn buildInpCrossEmbd(self: *const Builder) *Tensor {
        const cross = self.cross.?;
        const inp = allocator.create(InpCrossEmbd) catch oom();
        inp.* = .{ .cross = cross };

        // The C++'s TODO: the encoder's output tensor itself could be used
        // here once that is correct; for now only its shape is.
        const have = cross.v_embd.items.len != 0;
        const n_embd: i64 = if (have) cross.n_embd else self.hparams.nEmbdInp();
        const n_enc: i64 = if (have) cross.n_enc else self.hparams.n_ctx_train;

        const cur: *Tensor = c.ggml_new_tensor_2d(self.ctx0, c.GGML_TYPE_F32, n_embd, n_enc);
        c.ggml_set_input(cur);
        inp.cross_embd = cur;

        _ = self.res.addInput(InpCrossEmbd, inp);

        return cur;
    }

    /// Ports `llm_graph_context`'s `build_inp_pos_bucket_enc` (llama-graph.cpp:2491 @5266f24da).
    pub fn buildInpPosBucketEnc(self: *const Builder) *Tensor {
        const inp = allocator.create(InpPosBucket) catch oom();
        inp.* = .{ .hparams = self.hparams };

        const cur: *Tensor = c.ggml_new_tensor_2d(self.ctx0, c.GGML_TYPE_I32, self.n_tokens, self.n_tokens);
        c.ggml_set_input(cur);
        inp.pos_bucket = cur;

        _ = self.res.addInput(InpPosBucket, inp);

        return cur;
    }

    /// Ports `llm_graph_context`'s `build_inp_pos_bucket_dec` (llama-graph.cpp:2504 @5266f24da).
    pub fn buildInpPosBucketDec(self: *const Builder) *Tensor {
        const mctx_cur = self.mctx.?.cast(kv_cache.KvCacheContext);

        const inp = allocator.create(InpPosBucketKv) catch oom();
        inp.* = .{ .hparams = self.hparams, .mctx = mctx_cur };

        const n_kv: i64 = mctx_cur.getNKv();

        const cur: *Tensor = c.ggml_new_tensor_2d(self.ctx0, c.GGML_TYPE_I32, n_kv, self.n_tokens);
        c.ggml_set_input(cur);
        inp.pos_bucket = cur;

        _ = self.res.addInput(InpPosBucketKv, inp);

        return cur;
    }

    /// Ports `llm_graph_context`'s `build_pos_bias` (llama-graph.cpp:2521
    /// @5266f24da): the learned bias of each relative-position bucket, as
    /// a `[n_kv, n_tokens, n_head]` attention bias.
    pub fn buildPosBias(self: *const Builder, pos_bucket: *Tensor, attn_rel_b: *Tensor) *Tensor {
        const ctx = self.ctx0;
        const pos_bucket_1d: *Tensor = c.ggml_reshape_1d(ctx, pos_bucket, pos_bucket.ne[0] * pos_bucket.ne[1]);
        self.cb(pos_bucket_1d, "pos_bucket_1d", -1);

        var pos_bias: *Tensor = c.ggml_get_rows(ctx, attn_rel_b, pos_bucket_1d);

        pos_bias = c.ggml_reshape_3d(ctx, pos_bias, pos_bias.ne[0], pos_bucket.ne[0], pos_bucket.ne[1]);
        pos_bias = c.ggml_permute(ctx, pos_bias, 2, 0, 1, 3);
        pos_bias = c.ggml_cont(ctx, pos_bias);

        self.cb(pos_bias, "pos_bias", -1);

        return pos_bias;
    }

    /// Ports `llm_graph_context`'s `build_attn_inp_cross` (llama-graph.cpp:3176 @5266f24da).
    pub fn buildAttnInpCross(self: *const Builder) *InpAttnCross {
        const cross = self.cross.?;
        const inp = allocator.create(InpAttnCross) catch oom();
        inp.* = .{ .cross = cross };

        const n_enc: i32 = if (cross.v_embd.items.len != 0) @intCast(cross.n_enc) else @intCast(self.hparams.n_ctx_train);

        // Flash attention needs an F16 mask.
        const type_mask: c.enum_ggml_type = if (self.cparams.flash_attn) c.GGML_TYPE_F16 else c.GGML_TYPE_F32;

        inp.cross_kq_mask = c.ggml_new_tensor_4d(self.ctx0, type_mask, n_enc, self.n_tokens, 1, 1);
        c.ggml_set_input(inp.cross_kq_mask);

        inp.cross_kq_mask_cnv = inp.cross_kq_mask;

        return self.res.addInput(InpAttnCross, inp);
    }

    /// Ports `llm_graph_context`'s `build_attn` over the encoder's output
    /// (llama-graph.cpp:3192 @5266f24da): K and V come from the encoder,
    /// nothing is cached.
    fn buildAttnCross(
        self: *const Builder,
        inp: *InpAttnCross,
        wo: ?*Tensor,
        wo_b: ?*Tensor,
        wo_s: ?*Tensor,
        q_cur: *Tensor,
        k_cur: *Tensor,
        v_cur: *Tensor,
        kq_b: ?*Tensor,
        sinks: ?*Tensor,
        v_mla: ?*Tensor,
        kq_scale: f32,
        il: c_int,
    ) *Tensor {
        // Added together so they are not reordered, which reduces the
        // number of splits.
        c.ggml_build_forward_expand(self.gf, q_cur);
        c.ggml_build_forward_expand(self.gf, k_cur);
        c.ggml_build_forward_expand(self.gf, v_cur);

        const kq_mask = inp.getKqMaskCross();

        const q = q_cur;
        const k = k_cur;
        const v = v_cur;

        var cur = self.buildAttnMha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
        self.cb(cur, "kqv_out", il);

        if (wo) |w| cur = self.buildLoraMm(w, cur, wo_s);

        if (wo_b) |b| cur = c.ggml_add(self.ctx0, cur, b);

        return cur;
    }

    /// Ports `llm_graph_context`'s `build_attn_inp_kv_iswa` (llama-graph.cpp:3324
    /// @5266f24da).
    pub fn buildAttnInpKvIswa(self: *const Builder) *InpAttnKvIswa {
        const mctx_cur = self.mctx.?.cast(kv_cache_iswa.IswaContext);
        return self.res.addInput(InpAttnKvIswa, self.buildAttnInpKvIswaImpl(mctx_cur));
    }

    /// The body of `build_attn_inp_kv_iswa`, which the hybrid iSWA memory's
    /// input builds too.
    fn buildAttnInpKvIswaImpl(self: *const Builder, mctx_cur: *const kv_cache_iswa.IswaContext) *InpAttnKvIswa {
        const inp = allocator.create(InpAttnKvIswa) catch oom();
        inp.* = .{ .hparams = self.hparams, .cparams = self.cparams.*, .mctx = mctx_cur };

        {
            const base = mctx_cur.getBase();
            inp.self_k_idxs = base.buildInputKIdxs(self.ctx0, self.ubatch);
            inp.self_v_idxs = base.buildInputVIdxs(self.ctx0, self.ubatch);

            inp.self_kq_mask = buildAttnInpKqMask(self.ctx0, base, self.ubatch, self.cparams);
            inp.self_kq_mask_cnv = inp.self_kq_mask;
        }

        {
            impl.assert(@as(hparams_mod.SwaType, @fromBackingInt(@intCast(self.hparams.swa_type))) != .none, "llama-graph.cpp", 3338, "hparams.swa_type != LLAMA_SWA_TYPE_NONE && \"Use llama_kv_cache for non-SWA\"");

            const swa = mctx_cur.getSwa();
            inp.self_k_idxs_swa = swa.buildInputKIdxs(self.ctx0, self.ubatch);
            inp.self_v_idxs_swa = swa.buildInputVIdxs(self.ctx0, self.ubatch);

            inp.self_kq_mask_swa = buildAttnInpKqMask(self.ctx0, swa, self.ubatch, self.cparams);
            inp.self_kq_mask_swa_cnv = inp.self_kq_mask_swa;
        }

        inp.self_k_rot = mctx_cur.getBase().buildInputKRot(self.ctx0);
        inp.self_v_rot = mctx_cur.getBase().buildInputVRot(self.ctx0);

        inp.self_k_rot_swa = mctx_cur.getSwa().buildInputKRot(self.ctx0);
        inp.self_v_rot_swa = mctx_cur.getSwa().buildInputVRot(self.ctx0);

        return inp;
    }

    /// Ports `llm_graph_context`'s `build_attn` over an iSWA cache
    /// (llama-graph.cpp:3022 @5266f24da): the layer's side of the window
    /// picks the half. `k_cur` and `v_cur` are null for a layer that reads
    /// another layer's cells and stores nothing.
    fn buildAttnKvIswa(
        self: *const Builder,
        inp: *InpAttnKvIswa,
        wo: ?*Tensor,
        wo_b: ?*Tensor,
        wo_s: ?*Tensor,
        q_cur_: *Tensor,
        k_cur_: ?*Tensor,
        v_cur_: ?*Tensor,
        kq_b: ?*Tensor,
        sinks: ?*Tensor,
        v_mla: ?*Tensor,
        kq_scale: f32,
        il: c_int,
    ) *Tensor {
        const ctx = self.ctx0;
        const is_swa = self.hparams.isSwa(@intCast(il));
        var q_cur = q_cur_;
        var k_cur = k_cur_;
        var v_cur = v_cur_;

        const k_rot = if (is_swa) inp.self_k_rot_swa else inp.self_k_rot;
        const v_rot = if (is_swa) inp.self_v_rot_swa else inp.self_v_rot;

        if (k_rot) |r| {
            q_cur = mulMatHadamard(ctx, q_cur, r);
            if (k_cur) |k| k_cur = mulMatHadamard(ctx, k, r);
        }
        if (v_rot) |r| {
            if (v_cur) |v| v_cur = mulMatHadamard(ctx, v, r);
        }

        // Added together so they are not reordered, which reduces the
        // number of splits.
        c.ggml_build_forward_expand(self.gf, q_cur);
        if (k_cur) |k| c.ggml_build_forward_expand(self.gf, k);
        if (v_cur) |v| c.ggml_build_forward_expand(self.gf, v);

        const mctx_iswa = inp.mctx;
        const mctx_cur = if (is_swa) mctx_iswa.getSwa() else mctx_iswa.getBase();

        // Store to the KV cache, where there is anything to store.
        if (k_cur) |k| {
            const k_idxs = if (is_swa) inp.getKIdxsSwa() else inp.getKIdxs();
            c.ggml_build_forward_expand(self.gf, mctx_cur.cpyK(ctx, k, k_idxs, il));
        }
        if (v_cur) |v| {
            const v_idxs = if (is_swa) inp.getVIdxsSwa() else inp.getVIdxs();
            c.ggml_build_forward_expand(self.gf, mctx_cur.cpyV(ctx, v, v_idxs, il));
        }

        const kq_mask = if (is_swa) inp.getKqMaskSwa() else inp.getKqMask();

        const q = q_cur;
        const k = mctx_cur.getK(ctx, il);
        const v = mctx_cur.getV(ctx, il);

        var cur = self.buildAttnMha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
        self.cb(cur, "kqv_out", il);

        if (v_rot) |r| cur = mulMatHadamard(ctx, cur, r);

        if (wo) |w| cur = self.buildLoraMm(w, cur, wo_s);

        if (wo_b) |b| cur = c.ggml_add(ctx, cur, b);

        return cur;
    }

    /// Ports `llm_graph_context`'s `build_attn_inp_k_iswa` (llama-graph.cpp:3356
    /// @5266f24da).
    pub fn buildAttnInpKIswa(self: *const Builder) *InpAttnKIswa {
        const mctx_cur = self.mctx.?.cast(kv_cache_iswa.IswaContext);

        const inp = allocator.create(InpAttnKIswa) catch oom();
        inp.* = .{ .hparams = self.hparams, .cparams = self.cparams.*, .mctx = mctx_cur };

        {
            const base = mctx_cur.getBase();
            inp.self_k_idxs = base.buildInputKIdxs(self.ctx0, self.ubatch);

            inp.self_kq_mask = buildAttnInpKqMask(self.ctx0, base, self.ubatch, self.cparams);
            inp.self_kq_mask_cnv = inp.self_kq_mask;
        }

        {
            impl.assert(@as(hparams_mod.SwaType, @fromBackingInt(@intCast(self.hparams.swa_type))) != .none, "llama-graph.cpp", 3369, "hparams.swa_type != LLAMA_SWA_TYPE_NONE && \"Use llama_kv_cache for non-SWA\"");

            const swa = mctx_cur.getSwa();
            inp.self_k_idxs_swa = swa.buildInputKIdxs(self.ctx0, self.ubatch);

            inp.self_kq_mask_swa = buildAttnInpKqMask(self.ctx0, swa, self.ubatch, self.cparams);
            inp.self_kq_mask_swa_cnv = inp.self_kq_mask_swa;
        }

        inp.self_k_rot = mctx_cur.getBase().buildInputKRot(self.ctx0);

        inp.self_k_rot_swa = mctx_cur.getSwa().buildInputKRot(self.ctx0);

        return self.res.addInput(InpAttnKIswa, inp);
    }

    /// Ports `llm_graph_context`'s `build_attn` over a K-only iSWA cache
    /// (llama-graph.cpp:3109 @5266f24da): the cached K serves as V, viewed
    /// at `v_cur`'s width. `k_cur` is null for a layer that stores nothing.
    fn buildAttnKIswa(
        self: *const Builder,
        inp: *InpAttnKIswa,
        wo: ?*Tensor,
        wo_b: ?*Tensor,
        wo_s: ?*Tensor,
        q_cur_: *Tensor,
        k_cur_: ?*Tensor,
        v_cur: *Tensor,
        kq_b: ?*Tensor,
        sinks: ?*Tensor,
        v_mla: ?*Tensor,
        kq_scale: f32,
        il: c_int,
    ) *Tensor {
        const ctx = self.ctx0;
        const is_swa = self.hparams.isSwa(@intCast(il));
        var q_cur = q_cur_;
        var k_cur = k_cur_;

        const k_rot = if (is_swa) inp.self_k_rot_swa else inp.self_k_rot;

        if (k_rot) |r| {
            q_cur = mulMatHadamard(ctx, q_cur, r);
            if (k_cur) |k| k_cur = mulMatHadamard(ctx, k, r);
        }

        // Added together so they are not reordered, which reduces the
        // number of splits.
        c.ggml_build_forward_expand(self.gf, q_cur);
        if (k_cur) |k| c.ggml_build_forward_expand(self.gf, k);

        const mctx_iswa = inp.mctx;
        const mctx_cur = if (is_swa) mctx_iswa.getSwa() else mctx_iswa.getBase();

        // Store to the KV cache, where there is anything to store.
        if (k_cur) |k| {
            const k_idxs = if (is_swa) inp.getKIdxsSwa() else inp.getKIdxs();
            c.ggml_build_forward_expand(self.gf, mctx_cur.cpyK(ctx, k, k_idxs, il));
        }

        const kq_mask = if (is_swa) inp.getKqMaskSwa() else inp.getKqMask();

        // MLA-style attention: the cached K is used as V.
        const q = q_cur;
        const k = mctx_cur.getK(ctx, il);
        const v: *Tensor = c.ggml_view_4d(ctx, k, v_cur.ne[0], k.ne[1], k.ne[2], k.ne[3], k.nb[1], k.nb[2], k.nb[3], 0);

        var cur = self.buildAttnMha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
        self.cb(cur, "kqv_out", il);

        if (k_rot) |r| cur = mulMatHadamard(ctx, cur, r);

        if (wo) |w| cur = self.buildLoraMm(w, cur, wo_s);

        if (wo_b) |b| cur = c.ggml_add(ctx, cur, b);

        return cur;
    }

    /// Ports `llm_graph_context`'s `build_inp_dsv4` (llama-graph.cpp:3384
    /// @5266f24da): the raw cache's indices, mask and rotation, then each
    /// compressor's plan as inputs.
    pub fn buildInpDsv4(self: *const Builder) *InpDsv4 {
        const mctx_cur = self.mctx.?.cast(kv_cache_dsv4.Dsv4Context);
        const raw_ctx = mctx_cur.getRaw();

        const inp_raw = allocator.create(InpDsv4Raw) catch oom();
        inp_raw.* = .{ .cparams = self.cparams.*, .mctx = raw_ctx };

        const n_stream = mctx_cur.getCsaPlanFor(self.ubatch).n_stream;

        impl.assert(@as(hparams_mod.SwaType, @fromBackingInt(@intCast(self.hparams.swa_type))) != .none, "llama-graph.cpp", 3392, "hparams.swa_type != LLAMA_SWA_TYPE_NONE && \"DSV4 expects SWA raw cache\"");

        inp_raw.self_k_idxs = raw_ctx.buildInputKIdxs(self.ctx0, self.ubatch);
        inp_raw.self_kq_mask = dsv4BuildRawKqMask(self.ctx0, raw_ctx, self.ubatch, self.cparams, n_stream);
        inp_raw.self_kq_mask_cnv = inp_raw.self_kq_mask;

        inp_raw.self_k_rot = raw_ctx.buildInputKRot(self.ctx0);
        const inp = allocator.create(InpDsv4) catch oom();
        inp.* = .{ .inp_raw = inp_raw, .cparams = self.cparams.*, .mctx = mctx_cur, .debug = inputDebug() };

        dsv4BuildCompInputs(self.ctx0, &inp.inp_csa, mctx_cur.getCsaPlanFor(self.ubatch), "csa", self.cparams, n_stream);
        dsv4BuildCompInputs(self.ctx0, &inp.inp_hca, mctx_cur.getHcaPlanFor(self.ubatch), "hca", self.cparams, n_stream);
        dsv4BuildCompInputs(self.ctx0, &inp.inp_lid, mctx_cur.getLidPlanFor(self.ubatch), "lid", self.cparams, n_stream);
        inp.inp_csa.k_rot = mctx_cur.getCsa().buildInputKRot(self.ctx0);
        inp.inp_hca.k_rot = mctx_cur.getHca().buildInputKRot(self.ctx0);
        inp.inp_lid.k_rot = mctx_cur.getLid().buildInputKRot(self.ctx0);

        return self.res.addInput(InpDsv4, inp);
    }

    // -------------------------------------------------------------------------
    // Recurrent

    /// Ports `llm_graph_context`'s `build_rs` over explicit state indices
    /// (llama-graph.cpp:3411 @5266f24da).
    pub fn buildRsRaw(
        self: *const Builder,
        s: *Tensor,
        state_copy_main: *Tensor,
        state_copy_extra: *Tensor,
        state_size: i32,
        n_seqs: i32,
        n_rs: u32,
        rs_head: u32,
        rs_size: u32,
        rs_zero: i32,
        get_state_rows: GetRowsFn,
    ) *Tensor {
        _ = rs_size;
        const ctx = self.ctx0;
        const states = c.ggml_reshape_2d(ctx, s, state_size, s.ne[1]);

        // Clear one state, which is then copied to the others cleared; a
        // no-op when the view is zero-sized.
        const has_zero: i64 = @intFromBool(rs_zero >= 0);
        const state_zero = c.ggml_view_1d(ctx, states, state_size * has_zero, @as(usize, @intCast(@as(i64, rs_zero) * has_zero)) * states.*.nb[1]);
        c.ggml_build_forward_expand(self.gf, c.ggml_scale_inplace(ctx, state_zero, 0));

        // Copy states: {state_size, rs_size} -> {state_size, n_seqs}; the
        // destinations are all between rs_head and rs_head + n_rs.
        const output_states = get_state_rows.call(ctx, states, state_copy_main);
        c.ggml_build_forward_expand(self.gf, output_states);

        // Copy the extra states, which will not change further.
        const states_extra = c.ggml_get_rows(ctx, states, state_copy_extra);
        c.ggml_build_forward_expand(self.gf, c.ggml_cpy(ctx, states_extra, c.ggml_view_2d(ctx, s, state_size, @as(i64, n_rs) - n_seqs, s.nb[1], (@as(usize, rs_head) + @as(usize, @intCast(n_seqs))) * s.nb[1])));

        return output_states;
    }

    /// Ports `build_rs_inp_impl` (llama-graph.cpp:3447 @5266f24da).
    fn buildRsInpImpl(self: *const Builder, mctx_cur: *const memory_recurrent.RecurrentContext) *InpRs {
        const inp = allocator.create(InpRs) catch oom();
        inp.* = .{ .mctx = mctx_cur };
        const n_rs: i64 = mctx_cur.getNRs();
        const n_seqs: i64 = self.ubatch.n_seqs;
        inp.s_copy = c.ggml_new_tensor_1d(self.ctx0, c.GGML_TYPE_I32, n_rs);
        c.ggml_set_input(inp.s_copy);
        inp.s_copy_main = c.ggml_view_1d(self.ctx0, inp.s_copy, n_seqs, 0);
        inp.s_copy_extra = c.ggml_view_1d(self.ctx0, inp.s_copy, n_rs - n_seqs, @as(usize, @intCast(n_seqs)) * inp.s_copy.?.nb[0]);
        inp.head = mctx_cur.getHead();
        inp.rs_z = mctx_cur.getRsZ();
        return inp;
    }

    /// Ports `llm_graph_context`'s `build_rs_inp` (llama-graph.cpp:3469 @5266f24da).
    pub fn buildRsInp(self: *const Builder) *InpRs {
        const mctx_cur = self.mctx.?.cast(memory_recurrent.RecurrentContext);
        return self.res.addInput(InpRs, self.buildRsInpImpl(mctx_cur));
    }

    /// Ports `llm_graph_context`'s `build_rs` over an input (llama-graph.cpp:3477 @5266f24da).
    pub fn buildRs(self: *const Builder, inp: *const InpRs, s: *Tensor, state_size: i32, n_seqs: i32, get_state_rows: ?GetRowsFn) *Tensor {
        const kv_state = inp.mctx;
        return self.buildRsRaw(s, inp.s_copy_main.?, inp.s_copy_extra.?, state_size, n_seqs, kv_state.getNRs(), kv_state.getHead(), kv_state.getSize(), kv_state.getRsZ(), get_state_rows orelse ggml_get_rows_fn);
    }

    /// Ports `llm_graph_context`'s `build_rwkv_token_shift_load`
    /// (llama-graph.cpp:3490 @5266f24da): each sequence's token-shift
    /// state, `{n_embd, token_shift_count, n_seqs}`.
    pub fn buildRwkvTokenShiftLoad(self: *const Builder, inp: *const InpRs, ubatch: *const Ubatch, il: c_int) *Tensor {
        const mctx_cur = self.mctx.?.cast(memory_recurrent.RecurrentContext);
        const token_shift_count: i64 = self.hparams.token_shift_count;
        const n_seqs: i64 = ubatch.n_seqs;

        const token_shift_all = mctx_cur.getRL(il);

        var token_shift = self.buildRs(inp, token_shift_all, @intCast(self.hparams.nEmbdR()), @intCast(n_seqs), null);

        token_shift = c.ggml_reshape_3d(self.ctx0, token_shift, self.hparams.n_embd, token_shift_count, n_seqs);

        return token_shift;
    }

    /// Ports `llm_graph_context`'s `build_rwkv_token_shift_store`
    /// (llama-graph.cpp:3511 @5266f24da): the copy of the new token-shift
    /// state into the sequences' cells, for the caller to expand.
    pub fn buildRwkvTokenShiftStore(self: *const Builder, token_shift: *Tensor, ubatch: *const Ubatch, il: c_int) *Tensor {
        const mctx_cur = self.mctx.?.cast(memory_recurrent.RecurrentContext);
        const token_shift_count: i64 = self.hparams.token_shift_count;
        const n_embd: i64 = self.hparams.n_embd;
        const n_seqs: i64 = ubatch.n_seqs;
        const kv_head: usize = mctx_cur.getHead();
        const r_l = mctx_cur.getRL(il);
        const n_embd_r: usize = self.hparams.nEmbdR();

        return c.ggml_cpy(
            self.ctx0,
            c.ggml_view_1d(self.ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
            c.ggml_view_1d(self.ctx0, r_l, @as(i64, @intCast(n_embd_r)) * n_seqs, n_embd_r * kv_head * c.ggml_element_size(r_l)),
        );
    }

    /// Ports `llm_graph_context`'s `build_inp_mem_hybrid` (llama-graph.cpp:3531 @5266f24da).
    pub fn buildInpMemHybrid(self: *const Builder) *InpMemHybrid {
        const mctx_cur = self.mctx.?.cast(memory_hybrid.HybridContext);
        const inp_rs = self.buildRsInpImpl(mctx_cur.getRecr());
        const inp_attn = self.buildAttnInpKvImpl(mctx_cur.getAttn());
        const inp = allocator.create(InpMemHybrid) catch oom();
        inp.* = .{ .inp_attn = inp_attn, .inp_rs = inp_rs, .cparams = self.cparams.*, .mctx = mctx_cur };
        return self.res.addInput(InpMemHybrid, inp);
    }

    /// Ports `llm_graph_context`'s `build_inp_mem_hybrid_iswa`
    /// (llama-graph.cpp:3553 @5266f24da): the recurrent input, and both
    /// halves' indices and masks for the attention -- without the
    /// rotations `build_attn_inp_kv_iswa` adds.
    pub fn buildInpMemHybridIswa(self: *const Builder) *InpMemHybridIswa {
        const mctx_cur = self.mctx.?.cast(memory_hybrid_iswa.HybridIswaContext);

        const inp_rs = self.buildRsInpImpl(mctx_cur.getRecr());

        // The iSWA attention input.
        const attn_ctx = mctx_cur.getAttn();

        const inp_attn = allocator.create(InpAttnKvIswa) catch oom();
        inp_attn.* = .{ .hparams = self.hparams, .cparams = self.cparams.*, .mctx = attn_ctx };

        {
            const base = attn_ctx.getBase();
            inp_attn.self_k_idxs = base.buildInputKIdxs(self.ctx0, self.ubatch);
            inp_attn.self_v_idxs = base.buildInputVIdxs(self.ctx0, self.ubatch);

            inp_attn.self_kq_mask = buildAttnInpKqMask(self.ctx0, base, self.ubatch, self.cparams);
            inp_attn.self_kq_mask_cnv = inp_attn.self_kq_mask;
        }

        {
            const swa = attn_ctx.getSwa();
            inp_attn.self_k_idxs_swa = swa.buildInputKIdxs(self.ctx0, self.ubatch);
            inp_attn.self_v_idxs_swa = swa.buildInputVIdxs(self.ctx0, self.ubatch);

            inp_attn.self_kq_mask_swa = buildAttnInpKqMask(self.ctx0, swa, self.ubatch, self.cparams);
            inp_attn.self_kq_mask_swa_cnv = inp_attn.self_kq_mask_swa;
        }

        const inp = allocator.create(InpMemHybridIswa) catch oom();
        inp.* = .{ .inp_attn = inp_attn, .inp_rs = inp_rs, .cparams = self.cparams.*, .mctx = mctx_cur };
        return self.res.addInput(InpMemHybridIswa, inp);
    }

    /// Ports `llm_graph_context`'s `build_inp_mem_hybrid_k` (llama-graph.cpp:3542 @5266f24da).
    pub fn buildInpMemHybridK(self: *const Builder) *InpMemHybridK {
        const mctx_cur = self.mctx.?.cast(memory_hybrid.HybridContext);
        const inp_rs = self.buildRsInpImpl(mctx_cur.getRecr());
        const inp_attn = self.buildAttnInpKImpl(mctx_cur.getAttn());
        const inp = allocator.create(InpMemHybridK) catch oom();
        inp.* = .{ .inp_attn = inp_attn, .inp_rs = inp_rs, .cparams = self.cparams.*, .mctx = mctx_cur };
        return self.res.addInput(InpMemHybridK, inp);
    }

    // -------------------------------------------------------------------------
    // Pooling, sampling, dense output

    /// Ports `llm_graph_context`'s `build_dense_out` (llama-graph.cpp:3584 @5266f24da).
    pub fn buildDenseOut(self: *const Builder, dense_2: ?*Tensor, dense_2_b: ?*Tensor, dense_3: ?*Tensor) void {
        if (!self.cparams.embeddings or !(dense_2 != null or dense_2_b != null or dense_3 != null)) return;
        var cur = if (self.res.t_embd_pooled) |t| t else self.res.t_embd;
        impl.assert(cur != null, "llama-graph.cpp", 3592, "cur != nullptr && \"missing t_embd_pooled/t_embd\"");
        if (dense_2) |d| cur = c.ggml_mul_mat(self.ctx0, d, cur);
        if (dense_2_b) |b| cur = c.ggml_add(self.ctx0, cur, b);
        if (dense_3) |d| cur = c.ggml_mul_mat(self.ctx0, d, cur);
        self.cb(cur.?, "result_embd_pooled", -1);
        self.res.t_embd_pooled = cur;
        c.ggml_build_forward_expand(self.gf, cur);
    }

    /// Ports `llm_graph_context`'s `build_pooling` (llama-graph.cpp:3609 @5266f24da).
    pub fn buildPooling(self: *const Builder, cls: ?*Tensor, cls_b: ?*Tensor, cls_out: ?*Tensor, cls_out_b: ?*Tensor, cls_norm: ?*Tensor) void {
        if (!self.cparams.embeddings) return;
        const ctx = self.ctx0;
        const inp = self.res.t_embd;
        impl.assert(inp != null, "llama-graph.cpp", 3631, "inp != nullptr && \"missing result_norm/result_embd tensor\"");

        var cur: *Tensor = undefined;
        switch (self.pooling_type) {
            c.LLAMA_POOLING_TYPE_NONE => cur = inp.?,
            c.LLAMA_POOLING_TYPE_MEAN => {
                const inp_mean = self.buildInpMean();
                cur = c.ggml_mul_mat(ctx, c.ggml_cont(ctx, c.ggml_transpose(ctx, inp)), inp_mean);
            },
            c.LLAMA_POOLING_TYPE_CLS, c.LLAMA_POOLING_TYPE_LAST => {
                const inp_cls = self.buildInpCls();
                cur = c.ggml_get_rows(ctx, inp, inp_cls);
            },
            c.LLAMA_POOLING_TYPE_RANK => {
                if (self.arch == .modern_bert) {
                    const inp_mean = self.buildInpMean();
                    cur = c.ggml_mul_mat(ctx, c.ggml_cont(ctx, c.ggml_transpose(ctx, inp)), inp_mean);
                } else {
                    const inp_cls = self.buildInpCls();
                    cur = c.ggml_get_rows(ctx, inp, inp_cls);
                }
                // The classification head.
                if (cls) |w| {
                    cur = c.ggml_mul_mat(ctx, w, cur);
                    if (cls_b) |b| cur = c.ggml_add(ctx, cur, b);
                    cur = if (self.arch == .modern_bert) c.ggml_gelu(ctx, cur) else c.ggml_tanh(ctx, cur);
                    if (cls_norm) |n| cur = self.buildNorm(cur, n, null, .norm, -1);
                }
                // A single-layer classification head, where there is one.
                if (cls_out) |w| {
                    cur = c.ggml_mul_mat(ctx, w, cur);
                    if (cls_out_b) |b| cur = c.ggml_add(ctx, cur, b);
                }
                // The Qwen3 reranker's softmax.
                if (self.arch == .qwen3 or self.arch == .qwen3vl) cur = c.ggml_soft_max(ctx, cur);
            },
            else => impl.abort("llama-graph.cpp", 3699, "unknown pooling type"),
        }

        self.cb(cur, "result_embd_pooled", -1);
        self.res.t_embd_pooled = cur;
        c.ggml_build_forward_expand(self.gf, cur);
    }

    /// Ports `llm_graph_context`'s `build_sampling` (llama-graph.cpp:3709
    /// @5266f24da): the backend samplers' graph, one row per output.
    pub fn buildSampling(self: *const Builder) void {
        if (self.samplers.items.items.len == 0 or self.res.t_logits == null) return;
        const ctx = self.ctx0;

        var outs: [2]*Tensor = undefined;
        outs[0] = self.res.t_logits.?;

        const inp_sampling = allocator.create(InpSampling) catch oom();
        inp_sampling.* = .{ .samplers = self.samplers.clone() };
        _ = self.res.addInput(InpSampling, inp_sampling);

        // `std::map<llama_seq_id, std::vector<uint32_t>>`: each sequence's
        // output rows, by sequence.
        var rows_of: std.ArrayList(struct { seq_id: i32, rows: std.ArrayList(u32) }) = .empty;
        defer {
            for (rows_of.items) |*r| r.rows.deinit(allocator);
            rows_of.deinit(allocator);
        }
        var n_rows: u32 = 0;
        for (0..self.ubatch.n_tokens) |i| {
            if (self.ubatch.output[i] == 0) continue;
            const seq_id = self.ubatch.seq_id[i][0];
            var pos: usize = 0;
            while (pos < rows_of.items.len and rows_of.items[pos].seq_id < seq_id) pos += 1;
            if (pos == rows_of.items.len or rows_of.items[pos].seq_id != seq_id) {
                rows_of.insert(allocator, pos, .{ .seq_id = seq_id, .rows = .empty }) catch oom();
            }
            rows_of.items[pos].rows.append(allocator, n_rows) catch oom();
            n_rows += 1;
        }

        self.res.t_sampled.appendNTimes(allocator, null, n_rows) catch oom();
        self.res.t_sampled_probs.appendNTimes(allocator, null, n_rows) catch oom();
        self.res.t_sampled_logits.appendNTimes(allocator, null, n_rows) catch oom();
        self.res.t_candidates.appendNTimes(allocator, null, n_rows) catch oom();

        impl.assert(self.res.t_logits != null, "llama-graph.cpp", 3734, "res->t_logits != nullptr && \"missing t_logits tensor\"");

        // A dummy row keeps the single-output graph static whichever
        // samplers are active.
        const logits_t = c.ggml_pad(ctx, self.res.t_logits, 0, 1, 0, 0);

        for (self.samplers.items.items) |e| {
            if (e.sampler.*.iface.*.backend_reset) |f| f(e.sampler);
        }

        const dummy_row = [_]u32{0};

        for (self.samplers.items.items) |e| {
            var rows: []const u32 = &dummy_row;
            var active = false;
            for (rows_of.items) |r| {
                if (r.seq_id == e.seq_id) {
                    rows = r.rows.items;
                    active = true;
                }
            }
            // Inactive samplers always work on the first row.
            const i_out: c_int = if (active) 1 else 0;

            for (rows, 0..) |row, i| {
                const logits_seq = c.ggml_view_1d(ctx, logits_t, logits_t.*.ne[0], @as(usize, row) * logits_t.*.nb[1]);
                _ = c.ggml_format_name(logits_seq, "logits_seq_%d_%u", e.seq_id, @as(c_uint, @intCast(i)));

                var data: c.struct_llama_sampler_data = .{ .logits = logits_seq, .probs = null, .sampled = null, .candidates = null };
                std.debug.assert(e.sampler.*.iface.*.backend_apply != null);
                e.sampler.*.iface.*.backend_apply.?(e.sampler, ctx, self.gf, &data);

                if (data.sampled != null) {
                    if (active) self.res.t_sampled.items[row] = data.sampled;
                    outs[1] = data.sampled;
                    _ = c.ggml_build_forward_select(self.gf, @ptrCast(&outs), outs.len, i_out);
                }
                if (data.probs != null) {
                    if (active) self.res.t_sampled_probs.items[row] = data.probs;
                    outs[1] = data.probs;
                    _ = c.ggml_build_forward_select(self.gf, @ptrCast(&outs), outs.len, i_out);
                }
                if (data.logits != null) {
                    if (active) self.res.t_sampled_logits.items[row] = data.logits;
                    outs[1] = data.logits;
                    _ = c.ggml_build_forward_select(self.gf, @ptrCast(&outs), outs.len, i_out);
                }
                if (data.candidates != null) {
                    if (active) self.res.t_candidates.items[row] = data.candidates;
                    outs[1] = data.candidates;
                    _ = c.ggml_build_forward_select(self.gf, @ptrCast(&outs), outs.len, i_out);
                }
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Cross);
    std.testing.refAllDecls(Input);
    std.testing.refAllDecls(InpEmbd);
    std.testing.refAllDecls(InpEmbdH);
    std.testing.refAllDecls(InpPos);
    std.testing.refAllDecls(InpAttnTemp);
    std.testing.refAllDecls(InpOutIds);
    std.testing.refAllDecls(InpMean);
    std.testing.refAllDecls(InpCls);
    std.testing.refAllDecls(InpRs);
    std.testing.refAllDecls(InpAttnKv);
    std.testing.refAllDecls(InpMemHybrid);
    std.testing.refAllDecls(InpMemHybridK);
    std.testing.refAllDecls(InpSampling);
    std.testing.refAllDecls(SamplerMap);
    std.testing.refAllDecls(Callback);
    std.testing.refAllDecls(Params);
    std.testing.refAllDecls(FusedNode);
    std.testing.refAllDecls(Result);
    std.testing.refAllDecls(Qkv);
    std.testing.refAllDecls(Builder);
}
