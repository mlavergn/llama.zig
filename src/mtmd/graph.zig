//! The clip graph builder: the state every encoder graph is built with, and
//! the building blocks the per-model graphs under `models/` share.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/clip-graph.h` and the `clip_graph` methods of
//! `llama.cpp/tools/mtmd/clip.cpp` at v0.4.1 (`b29c606e2`). Each declaration
//! names what it ports and the line it began at.
//!
//! # A class hierarchy, flattened
//!
//! Upstream's `clip_graph` is an abstract class each model derives from,
//! overriding `build()` and, for three models, `build_mm()`. Here `Graph` is
//! the base's state and methods; a model is a struct holding a `Graph` and a
//! `build` function, chosen by `clip.zig`'s switch on the projector type
//! (upstream's `clip_get_graph_builder`). The `build_mm` override is the
//! `mm_hook` function pointer, and a `std::function` argument is a function
//! pointer with a context pointer beside it.

const std = @import("std");
const impl = @import("impl.zig");
const model_mod = @import("model.zig");
const names = @import("names.zig");
const c = impl.c;

/// A tensor, as every builder passes it: a C pointer that may be null.
pub const T = ?*c.ggml_tensor;

/// `DEFAULT_INTERPOLATION_MODE` (clip-graph.h:12 @b29c606e2).
pub const default_interpolation_mode: u32 = c.GGML_SCALE_MODE_BILINEAR | c.GGML_SCALE_FLAG_ANTIALIAS;

/// `CLIP_ALIGN` (clip-impl.h:438 @b29c606e2).
pub fn clipAlign(x: i64, n: i64) i64 {
    return @divTrunc(x + n - 1, n) * n;
}

/// `ggml_tensor::ne[i]` of a tensor known not to be null.
pub inline fn ne(t: T, i: usize) i64 {
    return t.?.ne[i];
}

/// `ggml_tensor::nb[i]` of a tensor known not to be null.
pub inline fn nb(t: T, i: usize) usize {
    return t.?.nb[i];
}

/// The `add_pos` argument of `build_vit`: a function and its context.
pub const AddPos = struct {
    ctx: ?*anyopaque = null,
    f: *const fn (ctx: ?*anyopaque, g: *Graph, cur: T, layer: *const model_mod.Layer) T,
};

/// The `callback_layer_out` hook of `build_vit_opts`.
pub const LayerOutHook = struct {
    ctx: ?*anyopaque = null,
    f: *const fn (ctx: ?*anyopaque, g: *Graph, cur: T, il: c_int) void,
};

/// Ports `build_vit_opts` (clip-graph.h:14 @b29c606e2).
pub const VitOpts = struct {
    attn_mask: T = null,
    attn_mask_layers: []const T = &.{}, // one per layer
    /// hook at layer output embeddings
    callback_layer_out: ?LayerOutHook = null,
    /// whether to skip the automatic post-layernorm (model.post_ln_w) applied at the end
    skip_post_ln: bool = false,
};

/// What a `Graph` is built from: the parts of `clip_ctx` its constructor
/// reads (mtmd/clip.cpp:248 @b29c606e2).
pub const Source = struct {
    model: *const model_mod.Model,
    buf_compute_meta: []u8,
    max_nodes: c_int,
    flash_attn_type: impl.FlashAttnType,
    n_mmproj_embd: c_int,
};

/// Ports `clip_graph` (clip-graph.h:26 @b29c606e2).
pub const Graph = struct {
    model: *const model_mod.Model,
    hparams: *const model_mod.Hparams,
    proj_type: names.ProjectorType,

    img: *const impl.ImageF32, // for backward compat
    img_batch: ?*const impl.ImageF32Batch = null,

    patch_size: c_int,
    n_patches_x: c_int,
    n_patches_y: c_int,
    n_patches: c_int,
    n_embd: c_int,
    n_head: c_int,
    n_head_kv: c_int,
    d_head: c_int,
    n_layer: c_int,
    n_mmproj_embd: c_int,
    eps: f32,
    kq_scale: f32,
    flash_attn_type: impl.FlashAttnType,

    n_batch: c_int = 1,

    /// Owned by the root graph; a child graph borrows its parent's.
    owns_ctx: bool,
    ctx0: ?*c.ggml_context,
    gf: ?*c.ggml_cgraph,

    /// `build_mm`'s override, for the three models that have one.
    mm_hook: ?*const fn (g: *const Graph, w: T, x: T) T = null,

    /// Ports `clip_graph` (mtmd/clip.cpp:248 @b29c606e2), its constructor over a
    /// context and an image.
    pub fn init(src: Source, img: *const impl.ImageF32) Graph {
        const hp = &src.model.hparams;
        const patch_size = hp.patch_size;
        const n_head = hp.n_head;
        const n_embd = hp.n_embd;
        const d_head: c_int = if (hp.n_embd_head > 0) hp.n_embd_head else (if (n_head > 0) @divTrunc(n_embd, n_head) else 0);
        const n_patches_x = @divTrunc(img.nx(), patch_size);
        const n_patches_y = @divTrunc(img.ny(), patch_size);
        const params: c.ggml_init_params = .{ .mem_size = src.buf_compute_meta.len, .mem_buffer = src.buf_compute_meta.ptr, .no_alloc = true };
        const ctx0 = c.ggml_init(params);
        return .{
            .model = src.model,
            .hparams = hp,
            .proj_type = src.model.proj_type,
            .img = img,
            .patch_size = patch_size,
            .n_patches_x = n_patches_x,
            .n_patches_y = n_patches_y,
            .n_patches = n_patches_x * n_patches_y,
            .n_embd = n_embd,
            .n_head = n_head,
            .n_head_kv = hp.n_head_kv,
            .d_head = d_head,
            .n_layer = hp.n_layer,
            .n_mmproj_embd = src.n_mmproj_embd,
            .eps = hp.eps,
            .kq_scale = if (d_head > 0) 1.0 / sqrtf(@floatFromInt(d_head)) else 0.0,
            .flash_attn_type = src.flash_attn_type,
            .owns_ctx = true,
            .ctx0 = ctx0,
            .gf = c.ggml_new_graph_custom(ctx0, @intCast(src.max_nodes), false),
        };
    }

    /// Ports `clip_graph` (mtmd/clip.cpp:276 @b29c606e2), its copy constructor: a
    /// sub-graph builder sharing its parent's context and graph.
    pub fn initChild(parent: *const Graph) Graph {
        var g = parent.*;
        g.img_batch = null;
        g.n_batch = 1;
        g.owns_ctx = false;
        g.mm_hook = null;
        return g;
    }

    /// The `ggml_context_ptr`'s destructor.
    pub fn deinit(self: *Graph) void {
        if (self.owns_ctx) c.ggml_free(self.ctx0);
        self.ctx0 = null;
    }

    /// Ports `build_mm` (mtmd/clip.cpp:299 @b29c606e2), a member of `clip_graph`, or the model's
    /// override.
    pub fn buildMm(self: *const Graph, w: T, x: T) T {
        if (self.mm_hook) |h| return h(self, w, x);
        return c.ggml_mul_mat(self.ctx0, w, x);
    }

    /// The base `build_mm`, for an override that falls back to it.
    pub fn buildMmBase(self: *const Graph, w: T, x: T) T {
        return c.ggml_mul_mat(self.ctx0, w, x);
    }

    /// Ports `cb` (mtmd/clip.cpp:303 @b29c606e2), a member of `clip_graph`.
    pub fn cb(self: *const Graph, cur: T, name: [*:0]const u8, il: c_int) void {
        _ = self;
        if (il >= 0) {
            _ = c.ggml_format_name(cur, "%s-%d", name, il);
        } else {
            _ = c.ggml_set_name(cur, name);
        }
    }

    /// Ports `get_img` (clip-graph.h:77 @b29c606e2), a member of `clip_graph`.
    pub fn getImg(self: *const Graph, idx: usize) *const impl.ImageF32 {
        impl.ggmlAssert(self.img_batch != null, "clip-graph.h", 78, "img_batch");
        impl.ggmlAssert(idx < self.img_batch.?.entries.items.len, "clip-graph.h", 79, "idx < img_batch->entries.size()");
        return &self.img_batch.?.entries.items[idx];
    }

    /// Ports `resize_position_embeddings` (mtmd/clip.cpp:312
    /// @b29c606e2), a member of `clip_graph`: siglip2 naflex.
    pub fn resizePositionEmbeddings(self: *Graph, interpolation_mode: u32) T {
        var pos_embd = self.model.position_embeddings;
        const height = @divTrunc(self.img.ny(), self.patch_size);
        const width = @divTrunc(self.img.nx(), self.patch_size);
        const mode = interpolation_mode;
        const n_per_side: c_int = @intFromFloat(@sqrt(@as(f64, @floatFromInt(ne(pos_embd, 1)))));

        impl.ggmlAssert(pos_embd != null, "clip.cpp", 319, "pos_embd");

        if (height == n_per_side and width == n_per_side) return pos_embd;

        const ctx0 = self.ctx0;
        pos_embd = c.ggml_reshape_3d(ctx0, pos_embd, self.n_embd, n_per_side, n_per_side); // -> (n_embd, n_per_side, n_per_side)
        pos_embd = c.ggml_permute(ctx0, pos_embd, 2, 0, 1, 3); // -> (n_per_side, n_per_side, n_embd)
        pos_embd = c.ggml_interpolate(ctx0, pos_embd, width, height, self.n_embd, 1, mode); // -> (width, height, n_embd)
        pos_embd = c.ggml_permute(ctx0, pos_embd, 1, 2, 0, 3); // -> (n_embd, width, height)
        pos_embd = c.ggml_cont_2d(ctx0, pos_embd, self.n_embd, width * height); // -> (n_embd, width * height)

        return pos_embd;
    }

    /// Ports `build_vit` (mtmd/clip.cpp:337 @b29c606e2), a member of `clip_graph`: the vision
    /// transformer most models use.
    pub fn buildVit(
        self: *Graph,
        inp_in: T,
        n_pos: i64,
        norm_t: model_mod.NormType,
        ffn_t: model_mod.FfnOpType,
        learned_pos_embd: T,
        add_pos: ?AddPos,
        opts: VitOpts,
    ) T {
        const ctx0 = self.ctx0;
        var inp = inp_in;
        // batch dim: inp is [n_embd, n_pos, B]
        const B = ne(inp, 2);

        if (learned_pos_embd != null) {
            inp = c.ggml_add(ctx0, inp, learned_pos_embd);
            self.cb(inp, "pos_embed", -1);
        }

        // flatten batch; unflatten again in attention
        inp = c.ggml_reshape_2d(ctx0, inp, self.n_embd, n_pos * B);

        var inpL = inp;

        // pre-layernorm
        if (self.model.pre_ln_w != null) {
            inpL = self.buildNorm(inpL, self.model.pre_ln_w, self.model.pre_ln_b, norm_t, self.eps, -1);
            self.cb(inpL, "pre_ln", -1);
        }

        const d_head = self.d_head;
        const n_head = self.n_head;
        const n_head_kv = self.n_head_kv;

        // loop over layers
        var il: c_int = 0;
        while (il < self.n_layer) : (il += 1) {
            const layer = &self.model.layers.items[@intCast(il)];
            var cur = inpL; // inpL = residual, cur = hidden_states

            var attn_mask = opts.attn_mask;
            if (opts.attn_mask_layers.len > @as(usize, @intCast(il))) attn_mask = opts.attn_mask_layers[@intCast(il)];

            // layernorm1
            cur = self.buildNorm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, self.eps, il);
            self.cb(cur, "layer_inp_normed", il);

            // self-attention
            {
                var Qcur: T = null;
                var Kcur: T = null;
                var Vcur: T = null;
                if (layer.qkv_w != null) {
                    // fused qkv
                    cur = self.buildMm(layer.qkv_w, cur);
                    if (layer.qkv_b != null) cur = c.ggml_add(ctx0, cur, layer.qkv_b);

                    // Q/K/V as [d_head, n_head, n_pos, B], the batch stride is cur->nb[1]*n_pos.
                    const rs = c.ggml_row_size(cur.?.type, d_head);
                    Qcur = c.ggml_view_4d(ctx0, cur, d_head, n_head, n_pos, B, rs, nb(cur, 1), nb(cur, 1) * @as(usize, @intCast(n_pos)), 0);
                    Kcur = c.ggml_view_4d(ctx0, cur, d_head, n_head, n_pos, B, rs, nb(cur, 1), nb(cur, 1) * @as(usize, @intCast(n_pos)), c.ggml_row_size(cur.?.type, n_head * d_head));
                    Vcur = c.ggml_view_4d(ctx0, cur, d_head, n_head, n_pos, B, rs, nb(cur, 1), nb(cur, 1) * @as(usize, @intCast(n_pos)), c.ggml_row_size(cur.?.type, 2 * n_head * d_head));

                    if (layer.q_norm != null) {
                        impl.ggmlAssert(ne(layer.q_norm, 0) == ne(Qcur, 0), "clip.cpp", 411, "layer.q_norm->ne[0] == Qcur->ne[0]");
                        Qcur = self.buildNorm(Qcur, layer.q_norm, null, norm_t, self.eps, il);
                        self.cb(Qcur, "Qcur_norm", il);
                    }

                    if (layer.k_norm != null) {
                        impl.ggmlAssert(ne(layer.k_norm, 0) == ne(Kcur, 0), "clip.cpp", 417, "layer.k_norm->ne[0] == Kcur->ne[0]");
                        Kcur = self.buildNorm(Kcur, layer.k_norm, null, norm_t, self.eps, il);
                        self.cb(Kcur, "Kcur_norm", il);
                    }
                } else {
                    // separate q, k, v
                    Qcur = self.buildMm(layer.q_w, cur);
                    if (layer.q_b != null) Qcur = c.ggml_add(ctx0, Qcur, layer.q_b);

                    Kcur = self.buildMm(layer.k_w, cur);
                    if (layer.k_b != null) Kcur = c.ggml_add(ctx0, Kcur, layer.k_b);

                    Vcur = self.buildMm(layer.v_w, cur);
                    if (layer.v_b != null) Vcur = c.ggml_add(ctx0, Vcur, layer.v_b);

                    // if true, norm must be applied after reshaping to (d_head, n_head, n_pos)
                    const norm_per_head = layer.q_norm != null and ne(layer.q_norm, 0) == d_head;

                    if (!norm_per_head) {
                        if (layer.q_norm != null) {
                            Qcur = self.buildNorm(Qcur, layer.q_norm, null, norm_t, self.eps, il);
                            self.cb(Qcur, "Qcur_norm", il);
                        }
                        if (layer.k_norm != null) {
                            Kcur = self.buildNorm(Kcur, layer.k_norm, null, norm_t, self.eps, il);
                            self.cb(Kcur, "Kcur_norm", il);
                        }
                    }

                    Qcur = c.ggml_reshape_4d(ctx0, Qcur, d_head, n_head, n_pos, B);
                    Kcur = c.ggml_reshape_4d(ctx0, Kcur, d_head, n_head_kv, n_pos, B);
                    Vcur = c.ggml_reshape_4d(ctx0, Vcur, d_head, n_head_kv, n_pos, B);

                    if (norm_per_head) {
                        if (layer.q_norm != null) {
                            Qcur = self.buildNorm(Qcur, layer.q_norm, null, norm_t, self.eps, il);
                            self.cb(Qcur, "Qcur_norm_per_head", il);
                        }
                        if (layer.k_norm != null) {
                            Kcur = self.buildNorm(Kcur, layer.k_norm, null, norm_t, self.eps, il);
                            self.cb(Kcur, "Kcur_norm_per_head", il);
                        }
                    }
                }

                self.cb(Qcur, "Qcur", il);
                self.cb(Kcur, "Kcur", il);
                self.cb(Vcur, "Vcur", il);

                if (add_pos) |ap| {
                    Qcur = ap.f(ap.ctx, self, Qcur, layer);
                    Kcur = ap.f(ap.ctx, self, Kcur, layer);
                    self.cb(Qcur, "Qcur_pos", il);
                    self.cb(Kcur, "Kcur_pos", il);
                }

                if (self.proj_type == .gemma4v) {
                    Vcur = c.ggml_rms_norm(ctx0, Vcur, self.eps);
                    self.cb(Vcur, "Vcur_normed", il);
                }

                // build_attn returns a flat 2D [n_embd, n_pos*B]
                cur = self.buildAttn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, attn_mask, self.kq_scale, il, null);
                self.cb(cur, "attn_out", il);
            }

            if (layer.ls_1_w != null) {
                cur = c.ggml_mul(ctx0, cur, layer.ls_1_w);
                self.cb(cur, "attn_out_scaled", il);
            }

            if (layer.attn_post_norm_w != null) {
                cur = self.buildNorm(cur, layer.attn_post_norm_w, null, norm_t, self.eps, il);
                self.cb(cur, "attn_post_normed", il);
            }

            // re-add the layer input, e.g., residual
            cur = c.ggml_add(ctx0, cur, inpL);

            inpL = cur; // inpL = residual, cur = hidden_states

            if (opts.callback_layer_out) |h| h.f(h.ctx, self, cur, il);

            self.cb(cur, "ffn_inp", il);

            // layernorm2 (pre-ffn norm)
            cur = self.buildNorm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, self.eps, il);
            self.cb(cur, "ffn_inp_normed", il);

            // ffn
            cur = if (layer.ff_gate_exps_w != null)
                self.buildMoeFfn(cur, layer, ffn_t, il)
            else
                self.buildFfn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w, layer.ff_down_b, ffn_t, il);

            self.cb(cur, "ffn_out", il);

            if (layer.ff_post_norm_w != null) {
                cur = self.buildNorm(cur, layer.ff_post_norm_w, null, norm_t, self.eps, il);
                self.cb(cur, "ffn_post_normed", il);
            }

            if (layer.ls_2_w != null) {
                cur = c.ggml_mul(ctx0, cur, layer.ls_2_w);
                self.cb(cur, "ffn_out_scaled", il);
            }

            // residual 2
            cur = c.ggml_add(ctx0, inpL, cur);
            self.cb(cur, "layer_out", il);

            if (layer.ls_out_w != null) {
                cur = c.ggml_mul(ctx0, cur, layer.ls_out_w);
                self.cb(cur, "layer_out_scaled", il);
            }

            inpL = cur;
        }

        if (self.model.audioHasAvgpool()) {
            var cur = inpL;
            cur = c.ggml_transpose(ctx0, cur);
            cur = c.ggml_cont(ctx0, cur);
            cur = c.ggml_pool_1d(ctx0, cur, c.GGML_OP_POOL_AVG, 2, 2, 0);
            cur = c.ggml_transpose(ctx0, cur);
            cur = c.ggml_cont(ctx0, cur);
            inpL = cur;
        }

        // post-layernorm
        if (self.model.post_ln_w != null and !opts.skip_post_ln) {
            inpL = self.buildNorm(inpL, self.model.post_ln_w, self.model.post_ln_b, norm_t, self.eps, -1);
        }

        // restore the batch dim
        impl.ggmlAssert(@rem(ne(inpL, 1), B) == 0, "clip.cpp", 565, "inpL->ne[1] % B == 0");
        inpL = c.ggml_reshape_3d(ctx0, inpL, self.n_embd, @divTrunc(ne(inpL, 1), B), B);
        return inpL;
    }

    /// Ports `build_inp` (mtmd/clip.cpp:572 @b29c606e2), a member of `clip_graph`: the patches
    /// after the conv2d, as `[n_embd, n_patches]`.
    pub fn buildInp(self: *Graph) T {
        const ctx0 = self.ctx0;
        const inp_raw = self.buildInpRaw(3);
        var inp = c.ggml_conv_2d(ctx0, self.model.patch_embeddings_0, inp_raw, self.patch_size, self.patch_size, 0, 0, 1, 1);
        inp = c.ggml_reshape_3d(ctx0, inp, self.n_patches, self.n_embd, self.n_batch);
        inp = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, inp));
        if (self.model.patch_bias != null) {
            inp = c.ggml_add(ctx0, inp, self.model.patch_bias);
            self.cb(inp, "patch_bias", -1);
        }
        return inp;
    }

    /// Ports `build_inp_raw` (mtmd/clip.cpp:584 @b29c606e2), a member of `clip_graph`.
    pub fn buildInpRaw(self: *Graph, channels: c_int) T {
        const inp_raw = c.ggml_new_tensor_4d(self.ctx0, c.GGML_TYPE_F32, self.img.nx(), self.img.ny(), channels, self.n_batch);
        _ = c.ggml_set_name(inp_raw, "inp_raw");
        c.ggml_set_input(inp_raw);
        return inp_raw;
    }

    /// Ports `build_norm` (mtmd/clip.cpp:591 @b29c606e2), a member of `clip_graph`.
    pub fn buildNorm(self: *const Graph, cur_in: T, mw: T, mb: T, t: model_mod.NormType, norm_eps: f32, il: c_int) T {
        var cur = if (t == .rms) c.ggml_rms_norm(self.ctx0, cur_in, norm_eps) else c.ggml_norm(self.ctx0, cur_in, norm_eps);
        if (mw != null) {
            cur = c.ggml_mul(self.ctx0, cur, mw);
            self.cb(cur, "norm_w", il);
        }
        if (mb != null) {
            cur = c.ggml_add(self.ctx0, cur, mb);
            self.cb(cur, "norm_b", il);
        }
        return cur;
    }

    /// Ports `build_ffn` (mtmd/clip.cpp:616 @b29c606e2), a member of `clip_graph`: only the
    /// parallel FFN.
    pub fn buildFfn(self: *const Graph, cur_in: T, up: T, up_b: T, gate: T, gate_b: T, down: T, down_b: T, type_op: model_mod.FfnOpType, il: c_int) T {
        const ctx0 = self.ctx0;
        var cur = cur_in;
        var tmp = if (up != null) self.buildMm(up, cur) else cur;
        self.cb(tmp, "ffn_up", il);

        if (up_b != null) {
            tmp = c.ggml_add(ctx0, tmp, up_b);
            self.cb(tmp, "ffn_up_b", il);
        }

        if (gate != null) {
            cur = self.buildMm(gate, cur);
            self.cb(cur, "ffn_gate", il);
            if (gate_b != null) {
                cur = c.ggml_add(ctx0, cur, gate_b);
                self.cb(cur, "ffn_gate_b", il);
            }
        } else {
            cur = tmp;
        }

        // we only support parallel ffn for now
        switch (type_op) {
            .silu => if (gate != null) {
                cur = c.ggml_swiglu_split(ctx0, cur, tmp);
                self.cb(cur, "ffn_swiglu", il);
            } else {
                cur = c.ggml_silu(ctx0, cur);
                self.cb(cur, "ffn_silu", il);
            },
            .gelu => if (gate != null) {
                cur = c.ggml_geglu_split(ctx0, cur, tmp);
                self.cb(cur, "ffn_geglu", il);
            } else {
                cur = c.ggml_gelu(ctx0, cur);
                self.cb(cur, "ffn_gelu", il);
            },
            .gelu_erf => if (gate != null) {
                cur = c.ggml_geglu_erf_split(ctx0, cur, tmp);
                self.cb(cur, "ffn_geglu_erf", il);
            } else {
                cur = c.ggml_gelu_erf(ctx0, cur);
                self.cb(cur, "ffn_gelu_erf", il);
            },
            .gelu_quick => if (gate != null) {
                cur = c.ggml_geglu_quick_split(ctx0, cur, tmp);
                self.cb(cur, "ffn_geglu_quick", il);
            } else {
                cur = c.ggml_gelu_quick(ctx0, cur);
                self.cb(cur, "ffn_gelu_quick", il);
            },
            .relu_sqr => {
                cur = c.ggml_relu(ctx0, cur);
                cur = c.ggml_sqr(ctx0, cur);
                self.cb(cur, "ffn_relu_sqr", il);
            },
        }

        if (down != null) cur = self.buildMm(down, cur);
        if (down_b != null) self.cb(cur, "ffn_down", il);
        if (down_b != null) cur = c.ggml_add(ctx0, cur, down_b);
        return cur;
    }

    /// Ports `build_moe_ffn` (mtmd/clip.cpp:706 @b29c606e2), a member of `clip_graph`: a sigmoid
    /// router with normalized top-k weights (dots3note vision).
    pub fn buildMoeFfn(self: *const Graph, cur_in: T, layer: *const model_mod.Layer, type_op: model_mod.FfnOpType, il: c_int) T {
        const ctx0 = self.ctx0;
        var cur = cur_in;
        const n_tokens = ne(cur, 1);
        const n_expert = ne(layer.ff_gate_exps_w, 2);
        const n_expert_used = @min(@as(i64, self.hparams.n_expert_used), n_expert);
        impl.ggmlAssert(n_expert_used > 0, "clip.cpp", 710, "n_expert_used > 0");
        impl.ggmlAssert(type_op == .silu, "clip.cpp", 711, "type_op == FFN_SILU");

        const probs = c.ggml_sigmoid(ctx0, self.buildMm(layer.ff_gate_inp_w, cur)); // [n_expert, n_tokens]
        self.cb(probs, "ffn_moe_probs", il);

        const sel = if (layer.ff_exp_probs_b != null) c.ggml_add(ctx0, probs, layer.ff_exp_probs_b) else probs;
        const selected = c.ggml_top_k(ctx0, sel, @intCast(n_expert_used)); // [n_expert_used, n_tokens]

        var weights = c.ggml_get_rows(ctx0, c.ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens), selected);
        weights = c.ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = c.ggml_div(ctx0, weights, c.ggml_sum_rows(ctx0, weights));
        weights = c.ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        self.cb(weights, "ffn_moe_weights", il);

        cur = c.ggml_reshape_3d(ctx0, cur, ne(cur, 0), 1, n_tokens);
        const gate = c.ggml_mul_mat_id(ctx0, layer.ff_gate_exps_w, cur, selected); // [n_ff, n_expert_used, n_tokens]
        const up = c.ggml_mul_mat_id(ctx0, layer.ff_up_exps_w, cur, selected);
        cur = c.ggml_mul(ctx0, c.ggml_silu(ctx0, gate), up);
        cur = c.ggml_mul_mat_id(ctx0, layer.ff_down_exps_w, cur, selected); // [n_embd, n_expert_used, n_tokens]
        cur = c.ggml_mul(ctx0, cur, weights);

        // sum over the selected experts
        var out: T = null;
        var i: i64 = 0;
        while (i < n_expert_used) : (i += 1) {
            const v = c.ggml_view_2d(ctx0, cur, ne(cur, 0), n_tokens, nb(cur, 2), @as(usize, @intCast(i)) * nb(cur, 1));
            out = if (out != null) c.ggml_add(ctx0, out, v) else v;
        }
        if (n_expert_used == 1) out = c.ggml_cont(ctx0, out);
        self.cb(out, "ffn_moe_out", il);
        return out;
    }

    /// Ports `build_attn` (mtmd/clip.cpp:748 @b29c606e2), a member of `clip_graph`.
    pub fn buildAttn(self: *const Graph, wo: T, wo_b: T, q_cur: T, k_cur: T, v_cur: T, kq_mask_in: T, kq_scale: f32, il: c_int, sinks: T) T {
        const ctx0 = self.ctx0;
        var kq_mask = kq_mask_in;
        // these nodes are added to the graph together so that they are not reordered
        // by doing so, the number of splits in the graph is reduced
        // the order is fixed without the compute flag, so an unselected branch stays out of the compute set
        c.ggml_build_forward_order(self.gf, q_cur);
        c.ggml_build_forward_order(self.gf, k_cur);
        c.ggml_build_forward_order(self.gf, v_cur);

        const q = c.ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
        var k = c.ggml_permute(ctx0, k_cur, 0, 2, 1, 3);

        var cur: T = undefined;

        if (self.flash_attn_type == .enabled) {
            var v = c.ggml_permute(ctx0, v_cur, 0, 2, 1, 3);

            k = c.ggml_cast(ctx0, k, c.GGML_TYPE_F16);
            v = c.ggml_cast(ctx0, v, c.GGML_TYPE_F16);
            if (kq_mask != null) kq_mask = c.ggml_cast(ctx0, kq_mask, c.GGML_TYPE_F16);

            cur = c.ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, 0.0, 0.0);
            _ = c.ggml_prec_set_acc(cur, c.GGML_PREC_F32);
            if (sinks != null) c.ggml_flash_attn_ext_add_sinks(cur, sinks);

            cur = c.ggml_reshape_2d(ctx0, cur, ne(cur, 0) * ne(cur, 1), ne(cur, 2) * ne(cur, 3));
        } else {
            var v = c.ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
            v = c.ggml_cont(ctx0, v);

            var kq = c.ggml_mul_mat(ctx0, k, q);
            kq = c.ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0);
            if (sinks != null) c.ggml_soft_max_add_sinks(kq, sinks);

            const kqv = c.ggml_mul_mat(ctx0, v, kq);
            cur = c.ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cur = c.ggml_cont_2d(ctx0, cur, ne(cur, 0) * ne(cur, 1), ne(cur, 2) * ne(cur, 3));
        }

        self.cb(cur, "kqv_out", il);

        if (wo != null) cur = self.buildMm(wo, cur);
        if (wo_b != null) cur = c.ggml_add(ctx0, cur, wo_b);
        return cur;
    }

    /// Ports `build_rope_2d` (mtmd/clip.cpp:822 @b29c606e2), a member of `clip_graph`: 2D RoPE as
    /// two `ggml_rope_ext` calls over the halves of each head, rotating
    /// adjacent pairs; `interleave_freq` gives the second half the odd
    /// frequencies (pixtral).
    pub fn buildRope2d(self: *const Graph, ctx0: ?*c.ggml_context, cur_in: T, pos_a: T, pos_b: T, freq_base: f32, interleave_freq: bool) T {
        _ = self;
        const n_dim = ne(cur_in, 0);
        const freq_scale_odd: f32 = if (interleave_freq) powf(freq_base, @as(f32, -2) / @as(f32, @floatFromInt(n_dim))) else 1.0;

        // first half, dims [0, n_dim/2)
        var cur = c.ggml_rope_ext(ctx0, cur_in, pos_a, null, @intCast(@divTrunc(n_dim, 2)), 0, 0, freq_base, 1.0, 0.0, 1.0, 0.0, 0.0);

        // second half, dims [n_dim/2, n_dim)
        cur = c.ggml_rope_ext(ctx0, cur, pos_b, null, @intCast(@divTrunc(n_dim, 2)), 0, 0, freq_base, freq_scale_odd, 0.0, 1.0, 0.0, 0.0);
        cur = c.ggml_rope_set_offset(cur, @intCast(@divTrunc(n_dim, 2)));
        return cur;
    }

    /// Ports `build_stack` (mtmd/clip.cpp:873 @b29c606e2), a member of `clip_graph`: ultravox's
    /// StackAudioFrames.
    pub fn buildStack(self: *const Graph, cur_in: T, stack_factor: i32, n_embed: i32) T {
        if (stack_factor <= 1) return cur_in;
        var cur = cur_in;
        const total_elements = c.ggml_nelements(cur);
        const stride: i64 = @as(i64, n_embed) * stack_factor;

        // Calculate padded length
        const padded_len = @divTrunc(total_elements + stride - 1, stride) * stride;
        const pad = padded_len - total_elements;

        if (pad > 0) {
            // Pad the tensor to make it divisible by stride
            cur = c.ggml_view_1d(self.ctx0, cur, total_elements, 0);
            cur = c.ggml_pad(self.ctx0, cur, @intCast(pad), 0, 0, 0);
        }

        // Reshape to [stride, padded_len / stride]
        cur = c.ggml_view_2d(self.ctx0, cur, stride, @divTrunc(padded_len, stride), c.ggml_row_size(cur.?.type, stride), 0);
        return cur;
    }

    /// Ports `build_patch_merge_permute` (mtmd/clip.cpp:899
    /// @b29c606e2), a member of `clip_graph`: pixel shuffle / patch merger, at any resolution.
    pub fn buildPatchMergePermute(self: *const Graph, cur_in: T, scale_factor: c_int) T {
        impl.ggmlAssert(scale_factor > 1, "clip.cpp", 900, "scale_factor > 1");
        const ctx0 = self.ctx0;
        var cur = cur_in;

        const n_embd: c_int = @intCast(ne(cur, 0));
        var width: i64 = @divTrunc(self.img.nx(), self.patch_size);
        var height: i64 = @divTrunc(self.img.ny(), self.patch_size);

        // pad width and height to factor
        const pad_width = clipAlign(width, scale_factor) - width;
        const pad_height = clipAlign(height, scale_factor) - height;
        cur = c.ggml_reshape_3d(ctx0, cur, n_embd, width, height);
        if (pad_width != 0 or pad_height != 0) {
            cur = c.ggml_pad(ctx0, cur, 0, @intCast(pad_width), @intCast(pad_height), 0);
            width += pad_width;
            height += pad_height;
        }

        // unshuffle h
        cur = c.ggml_reshape_3d(ctx0, cur, @as(i64, n_embd) * scale_factor, @divTrunc(width, scale_factor), height);
        cur = c.ggml_permute(ctx0, cur, 0, 2, 1, 3);

        // unshuffle w
        cur = c.ggml_cont_3d(ctx0, cur, @as(i64, n_embd) * scale_factor * scale_factor, @divTrunc(height, scale_factor), @divTrunc(width, scale_factor));
        cur = c.ggml_permute(ctx0, cur, 0, 2, 1, 3);

        cur = c.ggml_cont_2d(ctx0, cur, ne(cur, 0), ne(cur, 1) * ne(cur, 2));
        self.cb(cur, "pixel_shuffle", -1);
        return cur;
    }
};

extern fn sqrtf(x: f32) f32;
extern fn powf(x: f32, y: f32) f32;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    impl.refAllDeep(@This());
}
