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
//! All at v0.3.0 (`c1d0e7a00`).
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
const memory_recurrent = @import("memory_recurrent.zig");
const memory_hybrid = @import("memory_hybrid.zig");
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

/// `LLAMA_MAX_LAYERS` (llama-hparams.h:10 @c1d0e7a00).
const max_layers = hparams_mod.max_layers;

/// Mirrors `llm_graph_type` (llama-graph.h:36 @c1d0e7a00).
pub const GraphType = enum(c_int) {
    default,
    encoder,
    decoder,
    decoder_mtp,
};

/// Mirrors `llm_fused_op` (llama-graph.h:43 @c1d0e7a00).
pub const FusedOp = enum(c_int) {
    flash_attn,
    gdn_ar,
    gdn_ch,
    lightning_indexer,
    dsv4_hc_pre,
    dsv4_hc_comb,
    dsv4_hc_post,
};

/// Mirrors `llm_ffn_op_type` (llama-graph.h:53 @c1d0e7a00).
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

/// Mirrors `llm_ffn_gate_type` (llama-graph.h:66 @c1d0e7a00).
pub const FfnGate = enum(c_int) {
    seq,
    par,
};

/// Mirrors `llm_norm_type` (llama-graph.h:71 @c1d0e7a00).
pub const NormType = enum(c_int) {
    norm,
    rms,
    group,
};

/// Mirrors `llama_cross` (llama-graph.h:78 @c1d0e7a00): the encoder's
/// output, for a decoder's cross-attention.
pub const Cross = struct {
    n_embd: i64 = 0,
    n_enc: i64 = 0,
    v_embd: std.ArrayList(f32) = .empty,
};

// -----------------------------------------------------------------------------
// Inputs

/// Mirrors `llm_graph_input_i` (llama-graph.h:100 @c1d0e7a00).
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
/// (llama-graph.h:100 @c1d0e7a00).
fn inputDebug() c_int {
    const v = std.c.getenv("LLAMA_GRAPH_INPUT_DEBUG") orelse return 0;
    return impl.atoi(v);
}

/// Ports `llm_graph_input_embd` (llama-graph.h:126 @c1d0e7a00).
pub const InpEmbd = struct {
    /// I32 `[n_batch]`.
    tokens: ?*Tensor = null,
    /// F32 `[n_embd, n_batch]`.
    embd: ?*Tensor = null,
    n_embd: i64,

    /// Ports `llm_graph_input_embd`'s `set_input` (llama-graph.cpp:69 @c1d0e7a00).
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

    /// Ports `llm_graph_input_embd`'s `can_reuse` (llama-graph.cpp:85 @c1d0e7a00).
    pub fn canReuse(self: *InpEmbd, params: *const Params) bool {
        var res = true;
        res = res and ((params.ubatch.token == null) or (self.tokens != null and self.tokens.?.ne[0] == params.ubatch.n_tokens));
        res = res and ((params.ubatch.embd == null) or (self.embd != null and self.embd.?.ne[1] == params.ubatch.n_tokens));
        return res;
    }
};

/// Ports `llm_graph_input_embd_h` (llama-graph.h:142 @c1d0e7a00): the MTP
/// head's token embeddings and hidden states.
pub const InpEmbdH = struct {
    tokens: ?*Tensor = null,
    embd: ?*Tensor = null,
    h: ?*Tensor = null,
    n_embd: i64,

    /// Ports `llm_graph_input_embd_h`'s `set_input` (llama-graph.cpp:94 @c1d0e7a00).
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

    /// Ports `llm_graph_input_embd_h`'s `can_reuse` (llama-graph.cpp:117 @c1d0e7a00).
    pub fn canReuse(self: *InpEmbdH, params: *const Params) bool {
        var res = true;
        res = res and ((params.ubatch.token == null) or (self.tokens != null and self.tokens.?.ne[0] == params.ubatch.n_tokens));
        res = res and ((params.ubatch.embd == null) or (self.embd != null and self.embd.?.ne[1] == params.ubatch.n_tokens));
        res = res and ((params.ubatch.embd == null) or (self.h != null and self.h.?.ne[1] == params.ubatch.n_tokens));
        return res;
    }
};

/// Ports `llm_graph_input_pos` (llama-graph.h:158 @c1d0e7a00).
pub const InpPos = struct {
    /// I32 `[n_batch]`.
    pos: ?*Tensor = null,
    n_pos_per_embd: u32 = 1,

    /// Ports `llm_graph_input_pos`'s `set_input` (llama-graph.cpp:127 @c1d0e7a00).
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

    /// Ports `llm_graph_input_pos`'s `can_reuse` (llama-graph.cpp:149 @c1d0e7a00).
    pub fn canReuse(self: *InpPos, params: *const Params) bool {
        return self.pos.?.ne[0] == @as(i64, params.ubatch.n_tokens) * self.n_pos_per_embd;
    }
};

/// Ports `llm_graph_input_out_ids` (llama-graph.h:216 @c1d0e7a00).
pub const InpOutIds = struct {
    /// I32 `[n_outputs]`.
    out_ids: ?*Tensor = null,
    n_outputs: u32,

    /// Ports `llm_graph_input_out_ids`'s `set_input` (llama-graph.cpp:199 @c1d0e7a00).
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

    /// Ports `llm_graph_input_out_ids`'s `can_reuse` (llama-graph.cpp:226 @c1d0e7a00).
    pub fn canReuse(self: *InpOutIds, params: *const Params) bool {
        return self.n_outputs == params.n_outputs;
    }
};

/// Ports `llm_graph_input_mean` (llama-graph.h:236 @c1d0e7a00).
pub const InpMean = struct {
    /// F32 `[n_batch, n_batch]`.
    mean: ?*Tensor = null,
    cparams: Cparams,

    /// Ports `llm_graph_input_mean`'s `set_input` (llama-graph.cpp:234 @c1d0e7a00).
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

/// Ports `llm_graph_input_cls` (llama-graph.h:248 @c1d0e7a00).
pub const InpCls = struct {
    /// I32 `[n_batch]`.
    cls: ?*Tensor = null,
    cparams: Cparams,
    arch: Arch,

    /// Ports `llm_graph_input_cls`'s `set_input` (llama-graph.cpp:280 @c1d0e7a00).
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

/// Ports `llm_graph_input_rs` (llama-graph.h:261 @c1d0e7a00): which
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

    /// Ports `llm_graph_input_rs`'s `set_input` (llama-graph.cpp:329 @c1d0e7a00).
    pub fn setInput(self: *InpRs, _: ?*const Ubatch) void {
        const n_rs = self.mctx.getNRs();
        if (self.s_copy) |s_copy| {
            impl.assert(c.ggml_backend_buffer_is_host(s_copy.buffer), "llama-graph.cpp", 335, "ggml_backend_buffer_is_host(s_copy->buffer)");
            const data: [*]i32 = @ptrCast(@alignCast(s_copy.data));
            // Copy destinations are always between head and head + n.
            for (0..n_rs) |i| data[i] = self.mctx.sCopy(@intCast(i));
        }
    }

    /// Ports `llm_graph_input_rs`'s `can_reuse` (llama-graph.cpp:345 @c1d0e7a00).
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

/// Ports `llm_graph_input_attn_kv` (llama-graph.h:320 @c1d0e7a00).
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

    /// Ports `llm_graph_input_attn_kv`'s `set_input` (llama-graph.cpp:470 @c1d0e7a00).
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

    /// Ports `llm_graph_input_attn_kv`'s `can_reuse` (llama-graph.cpp:489 @c1d0e7a00).
    pub fn canReuse(self: *InpAttnKv, params: *const Params) bool {
        const mctx = params.mctx.?.cast(kv_cache.KvCacheContext);
        self.mctx = mctx;
        var res = true;
        res = res and self.self_k_idxs.?.ne[0] == params.ubatch.n_tokens;
        res = res and canReuseKqMask(self.self_kq_mask.?, mctx, &params.ubatch, &params.cparams);
        return res;
    }
};

/// Ports `llm_graph_input_mem_hybrid` (llama-graph.h:660 @c1d0e7a00).
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

    /// Ports `llm_graph_input_mem_hybrid`'s `set_input` (llama-graph.cpp:1086 @c1d0e7a00).
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

    /// Ports `llm_graph_input_mem_hybrid`'s `can_reuse` (llama-graph.cpp:1113 @c1d0e7a00).
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

/// Ports `llm_graph_input_sampling` (llama-graph.h:744 @c1d0e7a00).
pub const InpSampling = struct {
    samplers: SamplerMap,

    pub fn deinit(self: *InpSampling) void {
        self.samplers.deinit();
    }

    /// Ports `llm_graph_input_sampling`'s `set_input` (llama-graph.cpp:1267
    /// @c1d0e7a00): only the samplers of the ubatch's outputs.
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

    /// Ports `llm_graph_input_sampling`'s `can_reuse` (llama-graph.cpp:1290 @c1d0e7a00).
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

    /// Ports `llm_graph_params`'s `samplers_equal` (llama-graph.h:791 @c1d0e7a00).
    pub fn eql(lhs: *const SamplerMap, rhs: *const SamplerMap) bool {
        if (lhs.items.items.len != rhs.items.items.len) return false;
        for (lhs.items.items) |e| {
            if (rhs.get(e.seq_id) != e.sampler) return false;
        }
        return true;
    }
};

/// Ports `build_attn_inp_kq_mask` (llama-graph.cpp:29 @c1d0e7a00).
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

/// Ports `can_reuse_kq_mask` (llama-graph.cpp:48 @c1d0e7a00).
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

/// The `llm_graph_cb` callback (llama-graph.h:767 @c1d0e7a00): called on
/// every named tensor as the graph is built.
pub const Callback = struct {
    ctx: ?*anyopaque = null,
    func: ?*const fn (ctx: ?*anyopaque, ubatch: *const Ubatch, cur: *Tensor, name: [*:0]const u8, il: c_int) void = null,
};

/// Mirrors `llm_graph_params` (llama-graph.h:771 @c1d0e7a00).
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

    /// Ports `llm_graph_params`'s `allow_reuse` (llama-graph.h:814 @c1d0e7a00):
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

/// Mirrors `llm_graph_fused_node` (llama-graph.h:886 @c1d0e7a00).
pub const FusedNode = struct {
    op: FusedOp,
    tensor: *Tensor,
    il: c_int,
};

/// Ports `llm_graph_result` (llama-graph.h:892 @c1d0e7a00): a built graph,
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

    /// Ports the constructor of `llm_graph_result` (llama-graph.cpp:1308 @c1d0e7a00).
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

    /// Ports `llm_graph_result`'s `get_max_nodes` (llama-graph.cpp:1315 @c1d0e7a00).
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

    /// Ports `llm_graph_result`'s `reset` (llama-graph.cpp:1319 @c1d0e7a00).
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

    /// Ports `llm_graph_result`'s `set_inputs` (llama-graph.cpp:1353 @c1d0e7a00).
    pub fn setInputs(self: *Result, ubatch: ?*const Ubatch) void {
        for (self.inputs.items) |inp| inp.vtable.setInput(inp.ptr, ubatch);
    }

    /// Ports `llm_graph_result`'s `set_outputs` (llama-graph.cpp:1359 @c1d0e7a00).
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

    /// Ports `llm_graph_result`'s `can_reuse` (llama-graph.cpp:1403 @c1d0e7a00):
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

    /// Ports `llm_graph_result`'s `add_input` (llama-graph.cpp:1435 @c1d0e7a00):
    /// the result takes ownership.
    ///
    /// Return: `p`.
    pub fn addInput(self: *Result, comptime T: type, p: *T) *T {
        self.inputs.append(allocator, Input.of(T, p)) catch oom();
        return p;
    }

    /// Ports `llm_graph_result`'s `add_fused_node` (llama-graph.cpp:1440 @c1d0e7a00).
    pub fn addFusedNode(self: *Result, node: FusedNode) void {
        self.fused_nodes.append(allocator, node) catch oom();
    }

    /// Ports `llm_graph_result`'s `set_params` (llama-graph.cpp:1444 @c1d0e7a00):
    /// keep a copy of the parameters the graph was built with.
    pub fn setParams(self: *Result, params: *const Params) void {
        if (self.params) |*p| p.deinit();
        self.params = params.clone();
    }
};

/// Ports `llama_mul_mat_hadamard` (llama-impl.h:57 @c1d0e7a00).
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

/// Ports `llama_relative_position_bucket` (llama-graph.cpp:3824 @c1d0e7a00).
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

/// `llm_graph_get_rows_fn` (llama-graph.h:975 @c1d0e7a00): how `build_rs`
/// gathers the states a ubatch reads.
pub const GetRowsFn = *const fn (ctx: *c.ggml_context, states: *Tensor, ids: *Tensor) *Tensor;

fn ggmlGetRows(ctx: *c.ggml_context, states: *Tensor, ids: *Tensor) *Tensor {
    return c.ggml_get_rows(ctx, states, ids);
}

/// Mirrors `llm_graph_qkv` (llama-graph.h:977 @c1d0e7a00).
pub const Qkv = struct {
    /// `[n_embd_head, n_head, n_tokens]`.
    q: *Tensor,
    /// `[n_embd_head, n_head_kv, n_tokens]`.
    k: *Tensor,
    /// `[n_embd_head, n_head_kv, n_tokens]`.
    v: *Tensor,
};

/// Ports `llm_graph_context` (llama-graph.h:983 @c1d0e7a00): the state an
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
    /// @c1d0e7a00).
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
            .n_expert_used = if (cp.warmup) hp.n_expert else hp.n_expert_used,
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

    /// Ports `llm_graph_context`'s `cb` (llama-graph.cpp:1497 @c1d0e7a00).
    pub fn cb(self: *const Builder, cur: *Tensor, name: [*:0]const u8, il: c_int) void {
        if (self.cb_func.func) |f| f(self.cb_func.ctx, self.ubatch, cur, name, il);
    }

    /// Ports `llm_graph_context`'s `build_cvec` (llama-graph.cpp:1505 @c1d0e7a00).
    pub fn buildCvec(self: *const Builder, cur: *Tensor, il: c_int) *Tensor {
        return self.cvec.?.applyTo(self.ctx0, cur, il);
    }

    fn lorasOf(self: *const Builder) []const adapter.Loras.Entry {
        return if (self.loras) |l| l.items.items else &.{};
    }

    /// Ports `llm_graph_context`'s `build_lora_mm` (llama-graph.cpp:1511
    /// @c1d0e7a00): a matrix product, with the per-tensor scale and every
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

    /// Ports `llm_graph_context`'s `build_lora_mm_id` (llama-graph.cpp:1542 @c1d0e7a00).
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

    /// Ports `llm_graph_context`'s `build_norm` (llama-graph.cpp:1580 @c1d0e7a00).
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

    /// Ports `llm_graph_context`'s `build_qkv` (llama-graph.cpp:1616 @c1d0e7a00):
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

    /// Ports `llm_graph_context`'s `build_ffn` (llama-graph.cpp:1693 @c1d0e7a00).
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
        impl.assert(up_s == null or up_b == null or up == null or up.?.type != nvfp4, "llama-graph.cpp", 1723, "!up_s || !up_b || !up || up->type != GGML_TYPE_NVFP4");
        impl.assert(gate_s == null or gate_b == null or gate == null or gate.?.type != nvfp4, "llama-graph.cpp", 1724, "!gate_s || !gate_b || !gate || gate->type != GGML_TYPE_NVFP4");
        impl.assert(down_s == null or down_b == null or down == null or down.?.type != nvfp4, "llama-graph.cpp", 1725, "!down_s || !down_b || !down || down->type != GGML_TYPE_NVFP4");
        impl.assert(up_s == null or up == null or up.?.type != nvfp4 or !self.hasLora(up), "llama-graph.cpp", 1726, "!up_s || !up || up->type != GGML_TYPE_NVFP4 || !has_lora(up)");
        impl.assert(gate_s == null or gate == null or gate.?.type != nvfp4 or !self.hasLora(gate), "llama-graph.cpp", 1727, "!gate_s || !gate || gate->type != GGML_TYPE_NVFP4 || !has_lora(gate)");
        impl.assert(down_s == null or down == null or down.?.type != nvfp4 or !self.hasLora(down), "llama-graph.cpp", 1728, "!down_s || !down || down->type != GGML_TYPE_NVFP4 || !has_lora(down)");

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
                            tmp = c.ggml_clamp(ctx, tmp, -limit, limit);
                            self.cb(tmp, "ffn_up_clamped", il);
                            if (self.arch == .deepseek4 or (self.arch == .dflash and self.hparams.dsv4_hc_mult > 0)) {
                                cur = c.ggml_clamp(ctx, cur, -std.math.inf(f32), limit);
                                self.cb(cur, "ffn_gate_clamped", il);
                                cur = c.ggml_swiglu_split(ctx, cur, tmp);
                            } else {
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
                    impl.abort("llama-graph.cpp", 1849, "LLM_FFN_SWIGLU_OAI_MOE requires a parallel gate");
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
            .situ => impl.abort("llama-graph.cpp", 1863, "not yet supported"),
            .none => impl.abort("llama-graph.cpp", 1865, "fatal error"),
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

    // -------------------------------------------------------------------------
    // Inputs

    /// Ports `llm_graph_context`'s `build_inp_embd` (llama-graph.cpp:2308
    /// @c1d0e7a00): the token embeddings, or the embeddings given, chosen by
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

    /// Ports `llm_graph_context`'s `build_inp_pos` (llama-graph.cpp:2397 @c1d0e7a00).
    pub fn buildInpPos(self: *const Builder) *Tensor {
        const inp = allocator.create(InpPos) catch oom();
        inp.* = .{ .n_pos_per_embd = self.hparams.nPosPerEmbd() };
        inp.pos = c.ggml_new_tensor_1d(self.ctx0, c.GGML_TYPE_I32, self.n_tokens * self.hparams.nPosPerEmbd());
        c.ggml_set_input(inp.pos);
        _ = self.res.addInput(InpPos, inp);
        return inp.pos.?;
    }

    /// Ports `llm_graph_context`'s `build_inp_out_ids` (llama-graph.cpp:2425
    /// @c1d0e7a00). Always built, even when every token is an output, so the
    /// graph's topology does not depend on the output count.
    pub fn buildInpOutIds(self: *const Builder) *Tensor {
        const inp = allocator.create(InpOutIds) catch oom();
        inp.* = .{ .n_outputs = @intCast(self.n_outputs) };
        inp.out_ids = c.ggml_new_tensor_1d(self.ctx0, c.GGML_TYPE_I32, self.n_outputs);
        c.ggml_set_input(inp.out_ids);
        _ = self.res.addInput(InpOutIds, inp);
        return inp.out_ids.?;
    }

    /// Ports `llm_graph_context`'s `build_inp_mean` (llama-graph.cpp:2446 @c1d0e7a00).
    pub fn buildInpMean(self: *const Builder) *Tensor {
        const inp = allocator.create(InpMean) catch oom();
        inp.* = .{ .cparams = self.cparams.* };
        inp.mean = c.ggml_new_tensor_2d(self.ctx0, c.GGML_TYPE_F32, self.n_tokens, self.ubatch.n_seqs_unq);
        c.ggml_set_input(inp.mean);
        _ = self.res.addInput(InpMean, inp);
        return inp.mean.?;
    }

    /// Ports `llm_graph_context`'s `build_inp_cls` (llama-graph.cpp:2459 @c1d0e7a00).
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

    /// Ports `llm_graph_context`'s `build_attn_mha` (llama-graph.cpp:2541 @c1d0e7a00).
    pub fn buildAttnMha(self: *const Builder, q_: *Tensor, k_: *Tensor, v_: *Tensor, kq_b: ?*Tensor, kq_mask: ?*Tensor, sinks: ?*Tensor, v_mla: ?*Tensor, kq_scale: f32, il: c_int) *Tensor {
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
            impl.assert(kq_b == null, "llama-graph.cpp", 2565, "kq_b == nullptr && \"Flash attention does not support KQ bias yet\"");
            if (v_trans) v = c.ggml_transpose(ctx, v);
            // Without a KV cache, as in a non-causal embedding model.
            if (k.type == c.GGML_TYPE_F32) k = c.ggml_cast(ctx, k, c.GGML_TYPE_F16);
            if (v.type == c.GGML_TYPE_F32) v = c.ggml_cast(ctx, v, c.GGML_TYPE_F16);

            cur = c.ggml_flash_attn_ext(ctx, q, k, v, kq_mask, kq_scale, self.hparams.f_max_alibi_bias, if (self.hparams.attn_soft_cap) self.hparams.f_attn_logit_softcapping else 0.0);
            self.res.addFusedNode(.{ .op = .flash_attn, .tensor = cur, .il = il });

            c.ggml_flash_attn_ext_add_sinks(cur, sinks);
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

    /// Ports `build_attn_inp_kv_impl` (llama-graph.cpp:2753 @c1d0e7a00).
    fn buildAttnInpKvImpl(self: *const Builder, mctx_cur: *const kv_cache.KvCacheContext) *InpAttnKv {
        const inp = allocator.create(InpAttnKv) catch oom();
        inp.* = .{ .hparams = self.hparams, .cparams = self.cparams.*, .mctx = mctx_cur };
        impl.assert(@as(hparams_mod.SwaType, @fromBackingInt(@intCast(self.hparams.swa_type))) == .none, "llama-graph.cpp", 2762, "hparams.swa_type == LLAMA_SWA_TYPE_NONE && \"Use llama_kv_cache_iswa for SWA\"");
        inp.self_k_idxs = mctx_cur.buildInputKIdxs(self.ctx0, self.ubatch);
        inp.self_v_idxs = mctx_cur.buildInputVIdxs(self.ctx0, self.ubatch);
        inp.self_kq_mask = buildAttnInpKqMask(self.ctx0, mctx_cur, self.ubatch, self.cparams);
        inp.self_kq_mask_cnv = inp.self_kq_mask;
        inp.self_k_rot = mctx_cur.buildInputKRot(self.ctx0);
        inp.self_v_rot = mctx_cur.buildInputVRot(self.ctx0);
        return inp;
    }

    /// Ports `llm_graph_context`'s `build_attn_inp_kv` (llama-graph.cpp:2778 @c1d0e7a00).
    pub fn buildAttnInpKv(self: *const Builder) *InpAttnKv {
        const mctx_cur = self.mctx.?.cast(kv_cache.KvCacheContext);
        return self.res.addInput(InpAttnKv, self.buildAttnInpKvImpl(mctx_cur));
    }

    /// Ports `llm_graph_context`'s `build_attn` over a KV cache
    /// (llama-graph.cpp:2786 @c1d0e7a00): store K and V, attend, project.
    pub fn buildAttn(
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
        impl.assert(v_mla == null, "llama-graph.cpp", 2799, "v_mla == nullptr");
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

        var cur = self.buildAttnMha(q_cur, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
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

    // -------------------------------------------------------------------------
    // Recurrent

    /// Ports `llm_graph_context`'s `build_rs` over explicit state indices
    /// (llama-graph.cpp:3413 @c1d0e7a00).
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
        const output_states = get_state_rows(ctx, states, state_copy_main);
        c.ggml_build_forward_expand(self.gf, output_states);

        // Copy the extra states, which will not change further.
        const states_extra = c.ggml_get_rows(ctx, states, state_copy_extra);
        c.ggml_build_forward_expand(self.gf, c.ggml_cpy(ctx, states_extra, c.ggml_view_2d(ctx, s, state_size, @as(i64, n_rs) - n_seqs, s.nb[1], (@as(usize, rs_head) + @as(usize, @intCast(n_seqs))) * s.nb[1])));

        return output_states;
    }

    /// Ports `build_rs_inp_impl` (llama-graph.cpp:3449 @c1d0e7a00).
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

    /// Ports `llm_graph_context`'s `build_rs_inp` (llama-graph.cpp:3471 @c1d0e7a00).
    pub fn buildRsInp(self: *const Builder) *InpRs {
        const mctx_cur = self.mctx.?.cast(memory_recurrent.RecurrentContext);
        return self.res.addInput(InpRs, self.buildRsInpImpl(mctx_cur));
    }

    /// Ports `llm_graph_context`'s `build_rs` over an input (llama-graph.cpp:3479 @c1d0e7a00).
    pub fn buildRs(self: *const Builder, inp: *const InpRs, s: *Tensor, state_size: i32, n_seqs: i32, get_state_rows: ?GetRowsFn) *Tensor {
        const kv_state = inp.mctx;
        return self.buildRsRaw(s, inp.s_copy_main.?, inp.s_copy_extra.?, state_size, n_seqs, kv_state.getNRs(), kv_state.getHead(), kv_state.getSize(), kv_state.getRsZ(), get_state_rows orelse ggmlGetRows);
    }

    /// Ports `llm_graph_context`'s `build_inp_mem_hybrid` (llama-graph.cpp:3533 @c1d0e7a00).
    pub fn buildInpMemHybrid(self: *const Builder) *InpMemHybrid {
        const mctx_cur = self.mctx.?.cast(memory_hybrid.HybridContext);
        const inp_rs = self.buildRsInpImpl(mctx_cur.getRecr());
        const inp_attn = self.buildAttnInpKvImpl(mctx_cur.getAttn());
        const inp = allocator.create(InpMemHybrid) catch oom();
        inp.* = .{ .inp_attn = inp_attn, .inp_rs = inp_rs, .cparams = self.cparams.*, .mctx = mctx_cur };
        return self.res.addInput(InpMemHybrid, inp);
    }

    // -------------------------------------------------------------------------
    // Pooling, sampling, dense output

    /// Ports `llm_graph_context`'s `build_dense_out` (llama-graph.cpp:3586 @c1d0e7a00).
    pub fn buildDenseOut(self: *const Builder, dense_2: ?*Tensor, dense_2_b: ?*Tensor, dense_3: ?*Tensor) void {
        if (!self.cparams.embeddings or !(dense_2 != null or dense_2_b != null or dense_3 != null)) return;
        var cur = if (self.res.t_embd_pooled) |t| t else self.res.t_embd;
        impl.assert(cur != null, "llama-graph.cpp", 3594, "cur != nullptr && \"missing t_embd_pooled/t_embd\"");
        if (dense_2) |d| cur = c.ggml_mul_mat(self.ctx0, d, cur);
        if (dense_2_b) |b| cur = c.ggml_add(self.ctx0, cur, b);
        if (dense_3) |d| cur = c.ggml_mul_mat(self.ctx0, d, cur);
        self.cb(cur.?, "result_embd_pooled", -1);
        self.res.t_embd_pooled = cur;
        c.ggml_build_forward_expand(self.gf, cur);
    }

    /// Ports `llm_graph_context`'s `build_pooling` (llama-graph.cpp:3611 @c1d0e7a00).
    pub fn buildPooling(self: *const Builder, cls: ?*Tensor, cls_b: ?*Tensor, cls_out: ?*Tensor, cls_out_b: ?*Tensor, cls_norm: ?*Tensor) void {
        if (!self.cparams.embeddings) return;
        const ctx = self.ctx0;
        const inp = self.res.t_embd;
        impl.assert(inp != null, "llama-graph.cpp", 3634, "inp != nullptr && \"missing result_norm/result_embd tensor\"");

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
            else => impl.abort("llama-graph.cpp", 3703, "unknown pooling type"),
        }

        self.cb(cur, "result_embd_pooled", -1);
        self.res.t_embd_pooled = cur;
        c.ggml_build_forward_expand(self.gf, cur);
    }

    /// Ports `llm_graph_context`'s `build_sampling` (llama-graph.cpp:3711
    /// @c1d0e7a00): the backend samplers' graph, one row per output.
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

        impl.assert(self.res.t_logits != null, "llama-graph.cpp", 3735, "res->t_logits != nullptr && \"missing t_logits tensor\"");

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
    std.testing.refAllDecls(InpOutIds);
    std.testing.refAllDecls(InpMean);
    std.testing.refAllDecls(InpCls);
    std.testing.refAllDecls(InpRs);
    std.testing.refAllDecls(InpAttnKv);
    std.testing.refAllDecls(InpMemHybrid);
    std.testing.refAllDecls(InpSampling);
    std.testing.refAllDecls(SamplerMap);
    std.testing.refAllDecls(Callback);
    std.testing.refAllDecls(Params);
    std.testing.refAllDecls(FusedNode);
    std.testing.refAllDecls(Result);
    std.testing.refAllDecls(Qkv);
    std.testing.refAllDecls(Builder);
}
