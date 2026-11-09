//! Writing a loaded model back out as GGUF.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-model-saver.cpp` -- `llama_model_saver`'s
//!   and every member of `llama_model_saver` the model constructor reaches
//! - `llama.cpp/src/llama-model-saver.h` -- `llama_model_saver`
//! - `llama.cpp/src/llama-model.h` -- `llama_layer` and the structs it
//!   nests, for the order `add_tensors_from_model` walks them in
//! - `llama.cpp/src/llama-vocab.cpp` -- `llama_vocab::get_bpe_merges`
//!
//! All at v0.8.0 (`8642541b2`). `llama_model_save_to_file`, which lives in
//! `llama.cpp/src/llama.cpp`, is `api/quant.zig`'s.
//!
//! # Scope
//!
//! The saver built from a model, which is all `llama_model_save_to_file`
//! uses. The second constructor -- an architecture and a caller's
//! `gguf_context`, for `tests/test-llama-archs.cpp` -- and `save(FILE *)`
//! are C++-linkage and reached by nothing in `llama.h`, so they are left
//! out.
//!
//! # Which overload the C++ picks
//!
//! `add_kv` is overloaded, and which overload a call resolves to decides
//! the GGUF type written and whether an empty value is written at all:
//!
//! - a `std::string` -- the model's name, the architecture name, the
//!   tokenizer model and pre-tokenizer, the rope scaling type's name --
//!   goes through the container template, so an **empty** string writes
//!   **nothing**;
//! - a `std::vector<std::string>` -- the token list, the merges, the
//!   classifier labels -- takes the exact overload, which writes the array
//!   even when it is empty;
//! - a per-layer array whose layers all agree is written as one scalar of
//!   its element type;
//! - `std::vector<char>`, the precompiled charsmap, matches no branch of
//!   the template and aborts unless it is empty.
//!
//! # The layer walk
//!
//! `add_tensors_from_model` reads `llama_layer` as a flat array of tensor
//! pointers. `layer_fields` is that array's order, nested structs
//! included, and `scripts/quant-diff` checks every entry against
//! `offsetof` in the C++ and the count against `sizeof`. The port's
//! `Layer` declares only the tensors its models use: each entry is taken
//! from it when it has the field, and a comptime check refuses a `Layer`
//! field the list does not name.

const std = @import("std");
const impl = @import("impl.zig");
const arch = @import("arch.zig");
const model_mod = @import("model.zig");
const vm = @import("vocab/module.zig");
const hparams_mod = @import("hparams.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;
const Layer = model_mod.Layer;
const Tensor = c.ggml_tensor;

fn oom() noreturn {
    impl.abort("llama-model-saver.cpp", 0, "out of memory");
}

/// `add_tensors_from_model`'s model-level tensors, in its order
/// (llama-model-saver.cpp:475 @8642541b2), after `tok_embd`.
const model_fields = [_][]const u8{
    "type_embd",      "pos_embd",        "tok_norm",        "tok_norm_b",   "output_norm",   "output_norm_b",
    "output",         "output_b",        "output_norm_enc", "output_s",     "output_in_s",   "output_res_score",
    "nextn_proj_pre", "nextn_proj_post", "cls",             "cls_b",        "cls_out",       "cls_out_b",
    "cls_norm",       "hrm_z_l_init",    "hc_head_fn",      "hc_head_base", "hc_head_scale", "per_layer_tok_embd",
    "hc_head_norm",   "hc_head_down",    "hc_head_up",
};

/// `llama_model`'s tensors (llama-model.h:716 @8642541b2) that
/// `add_tensors_from_model` does not write.
const model_unsaved = [_][]const u8{
    "conv1d",                   "conv1d_b",                   "altup_proj",                  "altup_unembd_proj",
    "per_layer_model_proj",     "per_layer_proj_norm",        "fc",                          "fc_s",
    "fc_in_s",                  "d2t",                        "dspark_markov_w1",            "dspark_markov_w2",
    "dspark_markov_w2_s",       "dspark_conf_proj",           "dspark_conf_proj_b",          "dflash_selector_prev",
    "dflash_selector_next",     "dflash_selector_hidden",     "dense_2_out_layers",          "dense_2_out_layers_b",
    "dense_3_out_layers",       "cls_norm_b",
    // v0.7.0: DFlash's LiliCorr heads (models/dflash.cpp), not written either.
                    "lilicorr_token_proj",         "lilicorr_token_proj_b",
    "lilicorr_token_proj_s",    "lilicorr_pass_hidden_proj",  "lilicorr_pass_hidden_proj_b", "lilicorr_pass_hidden_proj_s",
    "lilicorr_feature_norm",    "lilicorr_feature_norm_b",    "lilicorr_feature_mlp_1",      "lilicorr_feature_mlp_1_b",
    "lilicorr_feature_mlp_3",   "lilicorr_feature_mlp_3_b",   "lilicorr_slot_embedding",     "lilicorr_rank_embedding",
    "lilicorr_attention_bias",  "lilicorr_context_proj",      "lilicorr_context_proj_b",     "lilicorr_context_proj_s",
    "lilicorr_output_norm",     "lilicorr_anchor_norm",       "lilicorr_factor_input_proj",  "lilicorr_factor_input_proj_b",
    "lilicorr_out_head",        "lilicorr_out_head_b",        "lilicorr_in_head",            "lilicorr_in_head_b",
    "lilicorr_anchor_out_head", "lilicorr_anchor_out_head_b",
};

/// Mirrors `llama_layer` (llama-model.h:264 @8642541b2) read as an array of
/// tensor pointers, with its nested `llama_layer_posnet` (llama-model.h:171
/// @8642541b2), `llama_layer_convnext` (llama-model.h:206 @8642541b2),
/// `llama_layer_shortconv` (llama-model.h:222 @8642541b2),
/// `llama_layer_nextn` (llama-model.h:228 @8642541b2) and
/// `llama_layer_switch_lora` (llama-model.h:246 @8642541b2) in place.
pub const layer_fields = [_][]const u8{
    // BEGIN layer_fields
    "attn_norm",                   "attn_norm_b",              "attn_norm_2",             "attn_norm_2_b",
    "attn_q_norm",                 "attn_q_norm_b",            "attn_k_norm",             "attn_k_norm_b",
    "attn_out_norm",               "attn_out_norm_b",          "attn_q_a_norm",           "attn_kv_a_norm",
    "attn_sub_norm",               "attn_post_norm",           "ffn_sub_norm",            "attn_norm_cross",
    "attn_norm_enc",               "ssm_norm",                 "ssm_dt_norm",             "ssm_b_norm",
    "ssm_c_norm",                  "wq",                       "wk",                      "wv",
    "wo",                          "wqkv",                     "wg",                      "wq_a",
    "wq_b",                        "wkv_a_mqa",                "wkv_b",                   "wkv",
    "wk_b",                        "wv_b",                     "wqkv_b",                  "wo_a",
    "wo_b",                        "wq_cross",                 "wk_cross",                "wv_cross",
    "wo_cross",                    "wq_enc",                   "wk_enc",                  "wv_enc",
    "wo_enc",                      "wqkv_gate",                "wqkv_gate_b",             "attn_v_gate",
    "attn_v_gate_b",               "attn_v_exps",              "attn_rel_b",              "attn_rel_b_enc",
    "attn_rel_b_cross",            "ffn_norm",                 "ffn_norm_b",              "ffn_post_norm",
    "ffn_post_norm_1",             "ffn_post_norm_2",          "ffn_pre_norm_2",          "ffn_norm_ada_down",
    "ffn_norm_ada_up",             "layer_out_norm",           "layer_out_norm_b",        "ffn_norm_exps",
    "ffn_norm_enc",                "ffn_gate",                 "ffn_down",                "ffn_up",
    "ffn_gate_enc",                "ffn_down_enc",             "ffn_up_enc",              "ffn_gate_inp",
    "ffn_gate_inp_s",              "ffn_gate_exps",            "ffn_down_exps",           "ffn_up_exps",
    "ffn_gate_up_exps",            "ffn_gate_inp_b",           "ffn_gate_exps_b",         "ffn_down_exps_b",
    "ffn_up_exps_b",               "ffn_gate_up_exps_b",       "ffn_gate_exps_s",         "ffn_down_exps_s",
    "ffn_up_exps_s",               "ffn_latent_down",          "ffn_latent_up",           "ffn_pre_exp_norm",
    "ffn_post_exp_norm",           "attn_resid_alpha",         "attn_resid_beta",         "ffn_resid_alpha",
    "ffn_resid_beta",              "ffn_gate_inp_shexp",       "ffn_gate_shexp",          "ffn_down_shexp",
    "ffn_up_shexp",                "ffn_gate_chexps",          "ffn_down_chexps",         "ffn_up_chexps",
    "ffn_gate_b",                  "ffn_down_b",               "ffn_up_b",                "ffn_act",
    "ffn_exp_probs_b",             "ffn_exp_probs_b_vl",       "ffn_gate_tid2eid",        "dflash_attn_conv_base",
    "dflash_attn_conv_proj",       "dflash_ffn_conv_base",     "dflash_ffn_conv_proj",    "lilicorr_attn_norm",
    "lilicorr_mlp_norm",           "lilicorr_attn_in_proj",    "lilicorr_attn_in_proj_b", "lilicorr_attn_out_proj",
    "lilicorr_attn_out_proj_b",    "lilicorr_attn_out_proj_s", "lilicorr_mlp_0",          "lilicorr_mlp_0_b",
    "lilicorr_mlp_0_s",            "lilicorr_mlp_2",           "lilicorr_mlp_2_b",        "lilicorr_mlp_2_s",
    "ssm_in",                      "ssm_x",                    "ssm_dt",                  "ssm_out",
    "ssm_conv1d",                  "ssm_a",                    "ssm_d",                   "ssm_conv1d_b",
    "ssm_dt_b",                    "ssm_beta_alpha",           "ssm_alpha",               "time_mix_w1",
    "time_mix_w2",                 "time_mix_lerp_x",          "time_mix_lerp_w",         "time_mix_lerp_k",
    "time_mix_lerp_v",             "time_mix_lerp_r",          "time_mix_lerp_g",         "time_mix_lerp_fused",
    "time_mix_first",              "time_mix_decay",           "time_mix_decay_w1",       "time_mix_decay_w2",
    "time_mix_key",                "time_mix_key_b",           "time_mix_value",          "time_mix_value_b",
    "time_mix_receptance",         "time_mix_receptance_b",    "time_mix_gate",           "time_mix_w0",
    "time_mix_a0",                 "time_mix_a1",              "time_mix_a2",             "time_mix_v0",
    "time_mix_v1",                 "time_mix_v2",              "time_mix_g1",             "time_mix_g2",
    "time_mix_k_k",                "time_mix_k_a",             "time_mix_r_k",            "time_mix_ln",
    "time_mix_ln_b",               "time_mix_output",          "channel_mix_lerp_k",      "channel_mix_lerp_r",
    "channel_mix_key",             "channel_mix_receptance",   "channel_mix_value",       "rope_long",
    "rope_short",                  "rope_freqs",               "wq_s",                    "wk_s",
    "wv_s",                        "wo_s",                     "wqkv_s",                  "wqkv_gate_s",
    "ffn_gate_s",                  "ffn_up_s",                 "ffn_down_s",              "ffn_gate_shexp_s",
    "ffn_up_shexp_s",              "ffn_down_shexp_s",         "ssm_in_s",                "ssm_out_s",
    "ssm_alpha_s",                 "ssm_beta_s",               "wq_in_s",                 "wk_in_s",
    "wv_in_s",                     "wo_in_s",                  "wqkv_in_s",               "wqkv_gate_in_s",
    "ffn_gate_in_s",               "ffn_up_in_s",              "ffn_down_in_s",           "ffn_gate_exps_in_s",
    "ffn_down_exps_in_s",          "ffn_up_exps_in_s",         "ffn_gate_shexp_in_s",     "ffn_up_shexp_in_s",
    "ffn_down_shexp_in_s",         "ssm_in_in_s",              "ssm_out_in_s",            "ssm_alpha_in_s",
    "ssm_beta_in_s",               "per_layer_inp_gate",       "per_layer_proj",          "per_layer_post_norm",
    "altup_correct_coef",          "altup_correct_scale",      "altup_predict_coef",      "altup_router",
    "altup_router_norm",           "laurel_l",                 "laurel_r",                "laurel_post_norm",
    "attn_sinks",                  "attn_kv_norm",             "hc_attn_fn",              "hc_attn_base",
    "hc_attn_scale",               "hc_ffn_fn",                "hc_ffn_base",             "hc_ffn_scale",
    "attn_comp_wkv",               "attn_comp_wgate",          "attn_comp_ape",           "attn_comp_norm",
    "indexer_comp_wkv",            "indexer_comp_wgate",       "indexer_comp_ape",        "indexer_comp_norm",
    "visexp_attn_wqkv",            "visexp_attn_wo",           "visexp_ffn_gate",         "visexp_ffn_down",
    "visexp_ffn_up",               "ffn_act_alpha_n",          "ffn_act_alpha_p",         "ffn_act_beta",
    "ffn_act_eps",                 "ssm_q_conv",               "ssm_k_conv",              "ssm_v_conv",
    "ssm_f_a",                     "ssm_f_b",                  "ssm_beta",                "ssm_g_a",
    "ssm_g_b",                     "ssm_o_norm",               "ssm_g",                   "attn_res_score",
    "ffn_res_score",               "ffn_routed_down",          "ffn_routed_up",           "ffn_routed_norm",
    "indexer_k_norm",              "indexer_k_norm_b",         "indexer_proj",            "indexer_attn_k",
    "indexer_attn_q_b",            "indexer_kpool_gate",       "indexer_kpool_ape",       "engram_kv",
    "engram_gate",                 "index_q_proj",             "index_k_proj",            "index_q_norm",
    "index_k_norm",                "hc_attn_norm",             "hc_attn_down",            "hc_attn_up",
    "hc_attn_inject",              "hc_ffn_norm",              "hc_ffn_down",             "hc_ffn_up",
    "hc_ffn_inject",               "hc_attn_down_inject",      "hc_ffn_down_inject",      "wqkv_z",
    "ssm_beta_alpha_cat",          "ple_key",                  "ple_value",               "ple_norm_key",
    "ple_norm_query",              "ple_norm_conv",            "ple_conv1d",              "out_scale",
    "posnet.norm1",                "posnet.norm1_b",           "posnet.conv1",            "posnet.conv1_b",
    "posnet.norm2",                "posnet.norm2_b",           "posnet.conv2",            "posnet.conv2_b",
    "posnet.attn_norm",            "posnet.attn_norm_b",       "posnet.attn_q",           "posnet.attn_q_b",
    "posnet.attn_k",               "posnet.attn_k_b",          "posnet.attn_v",           "posnet.attn_v_b",
    "posnet.attn_o",               "posnet.attn_o_b",          "posnet.norm",             "posnet.norm_b",
    "convnext.dw",                 "convnext.dw_b",            "convnext.norm",           "convnext.norm_b",
    "convnext.pw1",                "convnext.pw1_b",           "convnext.pw2",            "convnext.pw2_b",
    "convnext.gamma",              "shortconv.in_proj",        "shortconv.conv",          "shortconv.out_proj",
    "nextn.eh_proj",               "nextn.eh_proj_s",          "nextn.eh_proj_in_s",      "nextn.embed_tokens",
    "nextn.enorm",                 "nextn.hnorm",              "nextn.shared_head_head",  "nextn.shared_head_head_s",
    "nextn.shared_head_head_in_s", "nextn.shared_head_norm",   "nextn.hc_head_norm",      "nextn.hc_head_down",
    "nextn.hc_head_up",            "switch_lora.a_q",          "switch_lora.b_q",         "switch_lora.a_k",
    "switch_lora.b_k",             "switch_lora.a_v",          "switch_lora.b_v",         "switch_lora.a_o",
    "switch_lora.b_o",             "switch_lora.a_gate",       "switch_lora.b_gate",      "switch_lora.a_up",
    "switch_lora.b_up",            "switch_lora.a_down",       "switch_lora.b_down",
    // END layer_fields
};

/// The type at a dotted `path` inside `T`, or null when `T` lacks it.
fn PathType(comptime T: type, comptime path: []const u8) ?type {
    const dot = comptime std.mem.findScalar(u8, path, '.');
    const head = if (dot) |d| path[0..d] else path;
    if (!@hasField(T, head)) return null;
    const F = @FieldType(T, head);
    if (dot) |d| return PathType(F, path[d + 1 ..]);
    return F;
}

fn getPath(ptr: anytype, comptime path: []const u8) ?*Tensor {
    const dot = comptime std.mem.findScalar(u8, path, '.');
    if (dot) |d| return getPath(&@field(ptr.*, path[0..d]), path[d + 1 ..]);
    return @field(ptr.*, path);
}

fn listed(comptime list: []const []const u8, comptime name: []const u8) bool {
    for (list) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

/// Every tensor the port's `Layer` declares must be one the walk visits.
fn checkLayer(comptime T: type, comptime prefix: []const u8) void {
    for (@typeInfo(T).@"struct".field_names) |fname| {
        const path = prefix ++ fname;
        const F = @FieldType(T, fname);
        if (F == ?*Tensor) {
            if (!listed(&layer_fields, path)) @compileError("Layer." ++ path ++ " is not a member of llama_layer: add it to layer_fields in its declaration order");
        } else if (@typeInfo(F) == .@"struct") {
            checkLayer(F, path ++ ".");
        } else {
            @compileError("Layer." ++ path ++ " is not a tensor pointer, which add_tensors_from_model assumes");
        }
    }
}

comptime {
    @setEvalBranchQuota(1_000_000);
    checkLayer(Layer, "");
    for (@typeInfo(Model).@"struct".field_names) |fname| {
        if (@FieldType(Model, fname) == ?*Tensor and !std.mem.eql(u8, fname, "tok_embd") and !listed(&model_fields, fname) and !listed(&model_unsaved, fname)) {
            @compileError("Model." ++ fname ++ " is a tensor add_tensors_from_model does not name: decide whether the saver writes it");
        }
    }
}

/// A key's name, NUL-terminated.
const KeyZ = struct {
    buf: [arch.max_name + 1]u8 = undefined,

    fn ptr(self: *const KeyZ) [*:0]const u8 {
        return @ptrCast(&self.buf);
    }
};

/// Ports `llama_model_saver` (llama-model-saver.h:9 @8642541b2), built from
/// a model.
pub const Saver = struct {
    gguf_ctx: *c.gguf_context,
    model: *const Model,
    /// `llm_kv`: the architecture whose name the keys take.
    arch: arch.Arch,

    /// Ports the constructor of `llama_model_saver` from a model
    /// (llama-model-saver.cpp:17 @8642541b2).
    ///
    /// Parameters:
    /// - `model`: the model; borrowed for the saver's life.
    ///
    /// Return: the saver; free it with `deinit`.
    pub fn init(model: *const Model) Saver {
        const ctx = c.gguf_init_empty().?;
        impl.assert(arch.supportsSave(model.arch), "llama-model-saver.cpp", 19, "llama_arch_supports_save(model->arch)");
        return .{ .gguf_ctx = ctx, .model = model, .arch = model.arch };
    }

    /// Ports the destructor of `llama_model_saver` (llama-model-saver.cpp:22
    /// @8642541b2).
    ///
    /// Return: nothing.
    pub fn deinit(self: *Saver) void {
        c.gguf_free(self.gguf_ctx);
    }

    fn key(self: *const Saver, kv: arch.Kv) KeyZ {
        var k: KeyZ = .{};
        const n = arch.kvName(self.arch, kv, null);
        @memcpy(k.buf[0..n.len], n.slice());
        k.buf[n.len] = 0;
        return k;
    }

    /// Ports the scalar overloads `add_kv`, `add_kv`, `add_kv`, `add_kv` and
    /// `add_kv` (llama-model-saver.cpp:26, 30, 34, 38, 42 @8642541b2) -- for
    /// uint32_t, int32_t, uint64_t, float and bool -- chosen by the value's
    /// type.
    fn addScalar(self: *Saver, kv: arch.Kv, value: anytype) void {
        const k = self.key(kv);
        switch (@TypeOf(value)) {
            u32 => c.gguf_set_val_u32(self.gguf_ctx, k.ptr(), value),
            i32, c_int => c.gguf_set_val_i32(self.gguf_ctx, k.ptr(), value),
            u64 => c.gguf_set_val_u64(self.gguf_ctx, k.ptr(), value),
            f32 => c.gguf_set_val_f32(self.gguf_ctx, k.ptr(), value),
            bool => c.gguf_set_val_bool(self.gguf_ctx, k.ptr(), value),
            else => @compileError("no add_kv overload for " ++ @typeName(@TypeOf(value))),
        }
    }

    /// Ports the `const char *` overload of `add_kv` (llama-model-saver.cpp:46
    /// @8642541b2).
    fn addCStr(self: *Saver, kv: arch.Kv, value: [*:0]const u8) void {
        const k = self.key(kv);
        c.gguf_set_val_str(self.gguf_ctx, k.ptr(), value);
    }

    /// Ports the container `add_kv` (llama-model-saver.cpp:58 @8642541b2) for
    /// a `std::string`: nothing when it is empty.
    fn addString(self: *Saver, kv: arch.Kv, value: []const u8) void {
        if (value.len == 0) return;
        const z = allocator.dupeSentinel(u8, value, 0) catch oom();
        defer allocator.free(z);
        self.addCStr(kv, z.ptr);
    }

    /// Ports the container `add_kv` (llama-model-saver.cpp:58 @8642541b2) for
    /// an array: per layer, `n_layer()` values, written as one scalar when
    /// they all agree; nothing when there are none.
    fn addArr(self: *Saver, kv: arch.Kv, comptime T: type, value: []const T, per_layer: bool) void {
        const n_values: usize = if (per_layer) self.model.hparams.nLayer() else value.len;
        impl.assert(n_values <= value.len, "llama-model-saver.cpp", 61, "n_values <= value.size()");

        if (n_values == 0) return;

        if (per_layer) {
            var all_values_the_same = true;
            for (1..n_values) |i| {
                if (value[i] != value[0]) {
                    all_values_the_same = false;
                    break;
                }
            }
            if (all_values_the_same) {
                self.addScalar(kv, value[0]);
                return;
            }
        }

        const gtype = switch (T) {
            u8 => c.GGUF_TYPE_UINT8,
            i8 => c.GGUF_TYPE_INT8,
            u32 => c.GGUF_TYPE_UINT32,
            bool => c.GGUF_TYPE_BOOL,
            i32, c_int => c.GGUF_TYPE_INT32,
            u64 => c.GGUF_TYPE_UINT64,
            f32 => c.GGUF_TYPE_FLOAT32,
            else => @compileError("no add_kv branch for " ++ @typeName(T)),
        };
        const k = self.key(kv);
        c.gguf_set_arr_data(self.gguf_ctx, k.ptr(), gtype, value.ptr, n_values);
    }

    /// Ports the `std::vector<std::string>` overload of `add_kv`
    /// (llama-model-saver.cpp:102 @8642541b2): written even when empty.
    fn addStrArr(self: *Saver, kv: arch.Kv, value: []const []const u8) void {
        const tmp = allocator.alloc([*c]const u8, value.len) catch oom();
        defer allocator.free(tmp);
        var zs = allocator.alloc([:0]u8, value.len) catch oom();
        defer {
            for (zs) |z| allocator.free(z);
            allocator.free(zs);
        }
        for (value, 0..) |s, i| {
            zs[i] = allocator.dupeSentinel(u8, s, 0) catch oom();
            tmp[i] = zs[i].ptr;
        }
        const k = self.key(kv);
        c.gguf_set_arr_str(self.gguf_ctx, k.ptr(), if (tmp.len == 0) null else tmp.ptr, tmp.len);
    }

    /// Ports `add_tensor` (llama-model-saver.cpp:110 @8642541b2).
    fn addTensor(self: *Saver, tensor: ?*const Tensor) void {
        const t = tensor orelse return;
        if (c.gguf_find_tensor(self.gguf_ctx, &t.name) >= 0) {
            const name = std.mem.sliceTo(&t.name, 0);
            impl.assert(std.mem.eql(u8, name, "rope_freqs.weight") or std.mem.eql(u8, name, "rope_factors_long.weight") or
                std.mem.eql(u8, name, "rope_factors_short.weight"), "llama-model-saver.cpp", 118, "tensor_name == \"rope_freqs.weight\" || tensor_name == \"rope_factors_long.weight\" || tensor_name == \"rope_factors_short.weight\"");
            return;
        }
        c.gguf_add_tensor(self.gguf_ctx, t);
    }

    /// Ports `add_kv_from_model` (llama-model-saver.cpp:124 @8642541b2). The
    /// precision policy's names are written in `prec_src1`'s order, which
    /// in the C++ is a pointer-keyed `std::unordered_map`'s and so not
    /// reproducible from run to run; here it is the order the loader found
    /// them, `tensors_by_name`'s.
    ///
    /// Parameters:
    /// - `exc`: receives the `std::out_of_range` an unnamed rope scaling
    ///   type throws, or a token id past the vocabulary.
    ///
    /// Return: nothing, or `error.Thrown`.
    pub fn addKvFromModel(self: *Saver, exc: *Exception) Thrown!void {
        const model = self.model;
        const hp = &model.hparams;
        const vocab = &model.vocab;

        const n_vocab: i32 = @bitCast(vocab.nTokens());
        const nv: usize = @intCast(n_vocab);
        const tokens = allocator.alloc([]const u8, nv) catch oom();
        defer allocator.free(tokens);
        const scores = allocator.alloc(f32, nv) catch oom();
        defer allocator.free(scores);
        const token_types = allocator.alloc(i32, nv) catch oom();
        defer allocator.free(token_types);
        @memset(tokens, "");
        @memset(scores, 0);
        @memset(token_types, 0);

        if (vocab.type != vm.vocab.type_none) {
            var id: i32 = 0;
            while (id < n_vocab) : (id += 1) {
                const td = try vocab.getTokenData(id, exc);
                const i: usize = @intCast(id);
                tokens[i] = vocab.text(id);
                scores[i] = td.score;
                // Exact attribute values only; a combination is UNDEFINED.
                token_types[i] = switch (td.attr) {
                    c.LLAMA_TOKEN_ATTR_UNKNOWN => c.LLAMA_TOKEN_TYPE_UNKNOWN,
                    c.LLAMA_TOKEN_ATTR_UNUSED => c.LLAMA_TOKEN_TYPE_UNUSED,
                    c.LLAMA_TOKEN_ATTR_NORMAL => c.LLAMA_TOKEN_TYPE_NORMAL,
                    c.LLAMA_TOKEN_ATTR_CONTROL => c.LLAMA_TOKEN_TYPE_CONTROL,
                    c.LLAMA_TOKEN_ATTR_USER_DEFINED => c.LLAMA_TOKEN_TYPE_USER_DEFINED,
                    c.LLAMA_TOKEN_ATTR_BYTE => c.LLAMA_TOKEN_TYPE_BYTE,
                    else => c.LLAMA_TOKEN_TYPE_UNDEFINED,
                };
            }
        }

        self.addString(.general_architecture, model.archName());
        self.addString(.general_name, model.name.items);

        if (model.prec_policy.prec_src1.count() != 0) {
            const src1 = &model.prec_policy.prec_src1;
            const names = allocator.alloc([]const u8, src1.count()) catch oom();
            defer allocator.free(names);
            const values = allocator.alloc(i8, src1.count()) catch oom();
            defer allocator.free(values);
            for (src1.keys(), src1.values(), 0..) |w, prec, i| {
                names[i] = std.mem.sliceTo(c.ggml_get_name(w), 0);
                values[i] = if (prec == c.GGML_PREC_Q8) 0 else 1;
            }
            self.addStrArr(.general_tensor_extra_name, names);
            const k = self.key(.general_tensor_extra_prec_a4);
            c.gguf_set_arr_data(self.gguf_ctx, k.ptr(), c.GGUF_TYPE_BOOL, values.ptr, values.len);
        }

        self.addScalar(.vocab_size, vocab.nTokens());
        self.addScalar(.context_length, hp.n_ctx_train);
        self.addScalar(.embedding_length, hp.n_embd);
        if (hp.n_embd_out_impl > 0) self.addScalar(.embedding_length_out, hp.n_embd_out_impl);
        self.addScalar(.block_count, hp.n_layer_all);
        self.addScalar(.leading_dense_block_count, hp.n_layer_dense_lead);
        self.addArr(.feed_forward_length, u32, &hp.n_ff_arr, true);
        self.addArr(.expert_feed_forward_length, u32, &hp.n_ff_exp_arr, true);
        self.addScalar(.expert_latent_length, hp.n_expert_latent);
        self.addScalar(.moe_latent_size, hp.moe_latent_size);
        self.addScalar(.expert_shared_feed_forward_length, hp.n_ff_shexp);
        self.addScalar(.expert_chunk_feed_forward_length, hp.n_ff_chexp);
        self.addArr(.swiglu_clamp_exp, f32, hp.swiglu_clamp_exp[0..hp.n_layer_all], false);
        self.addArr(.swiglu_clamp_shexp, f32, hp.swiglu_clamp_shexp[0..hp.n_layer_all], false);
        self.addScalar(.use_parallel_residual, hp.use_par_res);
        self.addScalar(.expert_count, hp.n_expert);
        self.addScalar(.expert_used_count, hp.nExpertUsed(0));
        self.addScalar(.expert_shared_count, hp.n_expert_shared);
        self.addScalar(.expert_group_count, hp.n_expert_groups);
        self.addScalar(.expert_group_used_count, hp.n_group_used);
        self.addScalar(.expert_weights_scale, hp.expert_weights_scale);
        self.addScalar(.expert_weights_norm, hp.expert_weights_norm);
        self.addScalar(.expert_gating_func, hp.expert_gating_func);
        self.addScalar(.expert_group_scale, hp.expert_group_scale);
        self.addScalar(.experts_per_group, hp.n_group_experts);
        self.addScalar(.moe_every_n_layers, hp.moe_every_n_layers);
        self.addScalar(.nextn_predict_layers, hp.n_layer_nextn);
        self.addScalar(.num_deepstack_layers, hp.n_deepstack_layers);
        self.addArr(.deepstack_mapping, i32, &hp.deepstack_mapping_arr, false);
        self.addScalar(.pooling_type, @as(u32, @bitCast(hp.pooling_type)));
        self.addScalar(.logit_scale, hp.f_logit_scale);
        self.addScalar(.decoder_start_token_id, hp.dec_start_token_id);
        self.addScalar(.decoder_block_count, hp.dec_n_layer);
        self.addScalar(.attn_logit_softcapping, hp.f_attn_logit_softcapping);
        self.addScalar(.router_logit_softcapping, hp.f_router_logit_softcapping);
        self.addScalar(.final_logit_softcapping, hp.f_final_logit_softcapping);
        self.addScalar(.swin_norm, hp.swin_norm);
        self.addScalar(.rescale_every_n_layers, hp.rescale_every_n_layers);
        self.addScalar(.time_mix_extra_dim, hp.time_mix_extra_dim);
        self.addScalar(.time_decay_extra_dim, hp.time_decay_extra_dim);
        self.addScalar(.residual_scale, hp.f_residual_scale);
        self.addScalar(.embedding_scale, hp.f_embedding_scale);
        self.addScalar(.hrm_layers_per_stack, hp.n_hrm_layers_per_stack);
        self.addScalar(.hrm_h_cycles, hp.n_hrm_h_cycles);
        self.addScalar(.hrm_l_cycles, hp.n_hrm_l_cycles);
        self.addScalar(.hrm_prefix_lm, hp.hrm_prefix_lm);
        self.addScalar(.token_shift_count, hp.token_shift_count);
        self.addScalar(.interleave_moe_layer_step, hp.n_moe_layer_step);

        self.addArr(.attention_head_count, u32, &hp.n_head_arr, true);
        self.addArr(.attention_head_count_kv, u32, &hp.n_head_kv_arr, true);
        self.addScalar(.attention_max_alibi_bias, hp.f_max_alibi_bias);
        self.addScalar(.attention_clamp_kqv, hp.f_clamp_kqv);
        self.addScalar(.attention_key_length, hp.n_embd_head_k_full);
        self.addScalar(.attention_value_length, hp.n_embd_head_v_full);
        self.addScalar(.attention_layernorm_eps, hp.f_norm_eps);
        self.addScalar(.attention_layernorm_rms_eps, hp.f_norm_rms_eps);
        self.addScalar(.post_norm_eps, hp.f_post_norm_eps);
        self.addScalar(.attention_groupnorm_eps, hp.f_norm_group_eps);
        self.addScalar(.attention_groupnorm_groups, hp.n_norm_groups);
        self.addScalar(.attention_value_expert_count, hp.n_value_expert);
        self.addScalar(.attention_value_expert_used_count, hp.n_value_expert_used);
        self.addScalar(.attention_causal, hp.causal_attn);
        self.addScalar(.attention_q_lora_rank, hp.n_lora_q);
        self.addScalar(.attention_kv_lora_rank, hp.n_lora_kv);
        self.addScalar(.attention_decay_lora_rank, hp.n_lora_decay);
        self.addScalar(.attention_iclr_lora_rank, hp.n_lora_iclr);
        self.addScalar(.attention_value_residual_mix_lora_rank, hp.n_lora_value_res_mix);
        self.addScalar(.attention_gate_lora_rank, hp.n_lora_gate);
        self.addScalar(.attention_relative_buckets_count, hp.n_rel_attn_bkts);
        self.addArr(.attention_rope_pattern, u32, &hp.rope_pattern, true);
        self.addScalar(.attention_sliding_window, hp.n_swa);
        if (hp.swa_type != @backingInt(hparams_mod.SwaType.none)) {
            // Never collapsed to a scalar: the loaders read a scalar as a period.
            self.addArr(.attention_sliding_window_pattern, u32, hp.is_swa_impl[0..hp.n_layer_all], false);
        }
        self.addScalar(.attention_scale, hp.f_attention_scale);
        self.addScalar(.attention_output_scale, hp.f_attn_out_scale);
        self.addScalar(.attention_value_scale, hp.f_attn_value_scale);
        self.addScalar(.attention_temperature_length, hp.attn_temp_length);
        self.addScalar(.attention_temperature_scale, hp.f_attn_temp_scale);
        self.addScalar(.attention_key_length_mla, hp.n_embd_head_k_mla_impl);
        self.addScalar(.attention_value_length_mla, hp.n_embd_head_v_mla_impl);
        self.addScalar(.attention_key_length_swa, hp.n_embd_head_k_swa);
        self.addScalar(.attention_value_length_swa, hp.n_embd_head_v_swa);
        self.addScalar(.attention_key_length_mla_swa, hp.n_embd_head_k_mla_swa);
        self.addScalar(.attention_value_length_mla_swa, hp.n_embd_head_v_mla_swa);
        self.addScalar(.attention_kv_lora_rank_swa, hp.n_lora_kv_swa);
        self.addScalar(.attention_indexer_head_count, hp.indexer_n_head);
        self.addScalar(.attention_indexer_key_length, hp.indexer_head_size);
        self.addScalar(.attention_indexer_top_k, hp.indexer_top_k);
        self.addScalar(.attention_indexer_block_size, hp.indexer_block_size);
        self.addScalar(.attention_indexer_kpool, hp.indexer_kpool);
        self.addScalar(.attention_indexer_kpool_select_tail, hp.indexer_kpool_select_tail);
        self.addScalar(.attention_indexer_local_blocks, hp.indexer_local_blocks);
        self.addArr(.attention_indexer_types, u32, &hp.is_indexer_full_impl, true);
        self.addArr(.attention_recurrent_layers, u32, &hp.is_recr_impl, true);
        self.addScalar(.attention_output_group_count, hp.dsv4_o_group_count);
        self.addScalar(.attention_output_lora_rank, hp.dsv4_o_lora_rank);
        self.addScalar(.attention_compress_rope_freq_base, hp.dsv4_compress_rope_base);
        if (model.arch == .deepseek4 or hp.dsv4_hc_mult > 0) {
            // The loader wants one ratio per layer, nextn layers included.
            self.addArr(.attention_compress_ratios, u32, hp.dsv4_compress_ratios[0..hp.n_layer_all], false);
        } else {
            self.addArr(.attention_compress_ratios, u32, &hp.dsv4_compress_ratios, true);
        }
        self.addScalar(.hyper_connection_count, hp.dsv4_hc_mult);
        self.addScalar(.hyper_connection_sinkhorn_iterations, hp.dsv4_hc_sinkhorn_iters);
        self.addScalar(.hyper_connection_epsilon, hp.dsv4_hc_eps);
        self.addScalar(.hyper_connection_magnitude, hp.hc_magnitude);
        self.addScalar(.hash_layer_count, hp.dsv4_hash_layer_count);
        self.addScalar(.hyper_connection_low_rank, hp.hc_low_rank);

        if (model.arch == .deepseek41) {
            var kv_src: std.ArrayList(u32) = .empty;
            defer kv_src.deinit(allocator);
            var idx_src: std.ArrayList(u32) = .empty;
            defer idx_src.deinit(allocator);
            var il: u32 = 0;
            while (il < hp.nLayer()) : (il += 1) {
                if (hp.dsv41_kv_src[il] == @as(i32, @intCast(il))) kv_src.append(allocator, il) catch oom();
                if (hp.dsv41_topk_src[il] == @as(i32, @intCast(il))) idx_src.append(allocator, il) catch oom();
            }
            self.addArr(.attention_kv_source_layers, u32, kv_src.items, false);
            self.addArr(.attention_indexer_source_layers, u32, idx_src.items, false);
            if (hp.dsv41_cand_src >= 0) {
                self.addScalar(.attention_indexer_candidate_source_layer, hp.dsv41_cand_src);
                self.addScalar(.attention_indexer_candidate_top_k_blocks, hp.dsv41_cand_topk_blocks);
                self.addScalar(.attention_indexer_candidate_block_size, hp.dsv41_cand_block_size);
            }
            if (hp.ple_n_heads > 0) {
                self.addScalar(.engram_pad_token_id, hp.engram_pad_id);
                self.addArr(.engram_token_map, i32, model.engram_token_map.items, false);
            }
        }

        // The PLE group only means anything whole: all of it or none.
        if (hp.ple_n_heads > 0) {
            var ple_layers: std.ArrayList(u32) = .empty;
            defer ple_layers.deinit(allocator);
            var il: u32 = 0;
            while (il < hp.n_layer_all) : (il += 1) {
                // `is_ple_impl[il]`, the `std::bitset` read directly.
                if ((hp.is_ple_impl[il / 64] >> @intCast(il % 64)) & 1 != 0) ple_layers.append(allocator, il) catch oom();
            }
            self.addArr(.ple_layers, u32, ple_layers.items, false);
            self.addScalar(.ple_ngram_size, hp.ple_ngram_size);
            self.addScalar(.ple_heads_per_ngram, hp.ple_heads_per_ngram);
            self.addScalar(.ple_conv_kernel, hp.ple_conv_kernel);
            self.addScalar(.ple_eos_token_id, hp.ple_eos_token_id);
            self.addScalar(.embedding_length_per_layer, hp.ple_head_dim);
            self.addArr(.ple_layer_multipliers, u64, hp.ple_layer_multipliers[0 .. hp.ple_n_modules *% hp.ple_ngram_size], false);
            // `std::vector<uint64_t>` built from the `uint32_t` arrays.
            const n_ple_heads = hp.ple_n_modules *% hp.ple_n_heads;
            const offsets = allocator.alloc(u64, n_ple_heads) catch oom();
            defer allocator.free(offsets);
            for (offsets, hp.ple_head_offsets[0..n_ple_heads]) |*o, v| o.* = v;
            self.addArr(.ple_head_offsets, u64, offsets, false);
            const vocab_sizes = allocator.alloc(u64, n_ple_heads) catch oom();
            defer allocator.free(vocab_sizes);
            for (vocab_sizes, hp.ple_head_vocab_sizes[0..n_ple_heads]) |*o, v| o.* = v;
            self.addArr(.ple_head_vocab_sizes, u64, vocab_sizes, false);
        }

        const rope_scaling_factor: f32 = if (hp.rope_freq_scale_train == 1.0) 0.0 else 1.0 / hp.rope_freq_scale_train;

        self.addScalar(.rope_dimension_count, hp.n_rot_full);
        self.addScalar(.rope_dimension_count_swa, hp.n_rot_swa);
        self.addArr(.rope_dimension_sections, c_int, &hp.rope_sections, false);
        self.addString(.rope_section_order, try model_mod.ropeSectionOrderName(hp.rope_section_order, exc));
        self.addScalar(.rope_freq_base, hp.rope_freq_base_train);
        self.addScalar(.rope_freq_base_swa, hp.rope_freq_base_train_swa);
        self.addString(.rope_scaling_type, try model_mod.ropeScalingTypeName(hp.rope_scaling_type_train, exc));
        self.addScalar(.rope_scaling_factor, rope_scaling_factor);
        self.addScalar(.rope_scaling_attn_factor, hp.rope_attn_factor);
        self.addScalar(.rope_scaling_orig_ctx_len, hp.n_ctx_orig_yarn);
        self.addScalar(.rope_scaling_finetuned, hp.rope_finetuned);
        self.addScalar(.rope_scaling_yarn_log_mul, hp.rope_yarn_log_mul);
        self.addScalar(.rope_scaling_yarn_ext_factor, hp.yarn_ext_factor);
        self.addScalar(.rope_scaling_yarn_attn_factor, hp.yarn_attn_factor);
        self.addScalar(.rope_scaling_yarn_beta_fast, hp.yarn_beta_fast);
        self.addScalar(.rope_scaling_yarn_beta_slow, hp.yarn_beta_slow);

        self.addScalar(.ssm_inner_size, hp.ssm_d_inner);
        self.addScalar(.ssm_conv_kernel, hp.ssm_d_conv);
        self.addScalar(.ssm_state_size, hp.ssm_d_state);
        self.addScalar(.ssm_time_step_rank, hp.ssm_dt_rank);
        self.addScalar(.ssm_group_count, hp.ssm_n_group);
        self.addScalar(.ssm_dt_b_c_rms, hp.ssm_dt_b_c_rms);

        self.addScalar(.kda_head_dim, hp.n_embd_head_kda);
        self.addScalar(.kda_safe_gate, hp.kda_safe_gate);
        self.addScalar(.kda_gate_lower_bound, hp.kda_gate_lower_bound);

        self.addScalar(.wkv_head_size, hp.wkv_head_size);

        self.addString(.tokenizer_model, vocab.tokenizer_model.items);
        self.addString(.tokenizer_pre, vocab.tokenizer_pre.items);
        self.addStrArr(.tokenizer_list, tokens);
        self.addArr(.tokenizer_token_type, i32, token_types, false);
        self.addScalar(.tokenizer_token_type_count, vocab.n_token_types);
        self.addArr(.tokenizer_scores, f32, scores, false);
        {
            const merges = bpeMerges(vocab);
            defer freeMerges(merges);
            self.addStrArr(.tokenizer_merges, merges);
        }
        // Token ids are i32 but read back as u32; fine for writing.
        if (vocab.special_bos_id != vm.vocab.token_null) self.addScalar(.tokenizer_bos_id, @as(u32, @bitCast(vocab.special_bos_id)));
        if (vocab.special_eos_id != vm.vocab.token_null) self.addScalar(.tokenizer_eos_id, @as(u32, @bitCast(vocab.special_eos_id)));
        if (vocab.special_eot_id != vm.vocab.token_null) self.addScalar(.tokenizer_eot_id, @as(u32, @bitCast(vocab.special_eot_id)));
        if (vocab.special_eom_id != vm.vocab.token_null) self.addScalar(.tokenizer_eom_id, @as(u32, @bitCast(vocab.special_eom_id)));
        if (vocab.special_unk_id != vm.vocab.token_null) self.addScalar(.tokenizer_unk_id, @as(u32, @bitCast(vocab.special_unk_id)));
        if (vocab.special_sep_id != vm.vocab.token_null) self.addScalar(.tokenizer_sep_id, @as(u32, @bitCast(vocab.special_sep_id)));
        if (vocab.special_pad_id != vm.vocab.token_null) self.addScalar(.tokenizer_pad_id, @as(u32, @bitCast(vocab.special_pad_id)));
        self.addScalar(.tokenizer_add_bos, vocab.add_bos);
        self.addScalar(.tokenizer_add_eos, vocab.add_eos);
        self.addScalar(.tokenizer_add_sep, vocab.add_sep);
        self.addScalar(.tokenizer_add_prefix, vocab.add_space_prefix);
        self.addScalar(.tokenizer_remove_extra_ws, vocab.remove_extra_whitespaces);
        self.addScalar(.tokenizer_clean_spaces, vocab.clean_spaces);
        // `std::vector<char>`: no branch of the template matches it.
        if (vocab.precompiled_charsmap.items.len != 0) impl.abort("llama-model-saver.cpp", 98, "fatal error");
        if (vocab.special_fim_pre_id != vm.vocab.token_null) self.addScalar(.tokenizer_fim_pre_id, @as(u32, @bitCast(vocab.special_fim_pre_id)));
        if (vocab.special_fim_suf_id != vm.vocab.token_null) self.addScalar(.tokenizer_fim_suf_id, @as(u32, @bitCast(vocab.special_fim_suf_id)));
        if (vocab.special_fim_mid_id != vm.vocab.token_null) self.addScalar(.tokenizer_fim_mid_id, @as(u32, @bitCast(vocab.special_fim_mid_id)));
        if (vocab.special_fim_pad_id != vm.vocab.token_null) self.addScalar(.tokenizer_fim_pad_id, @as(u32, @bitCast(vocab.special_fim_pad_id)));
        if (vocab.special_fim_rep_id != vm.vocab.token_null) self.addScalar(.tokenizer_fim_rep_id, @as(u32, @bitCast(vocab.special_fim_rep_id)));
        if (vocab.special_fim_sep_id != vm.vocab.token_null) self.addScalar(.tokenizer_fim_sep_id, @as(u32, @bitCast(vocab.special_fim_sep_id)));

        self.addScalar(.posnet_embedding_length, hp.posnet.n_embd);
        self.addScalar(.posnet_block_count, hp.posnet.n_layer);

        self.addScalar(.convnext_embedding_length, hp.convnext.n_embd);
        self.addScalar(.convnext_block_count, hp.convnext.n_layer);

        {
            const labels = allocator.alloc([]const u8, model.classifier_labels.items.len) catch oom();
            defer allocator.free(labels);
            for (model.classifier_labels.items, labels) |l, *o| o.* = l;
            self.addStrArr(.classifier_output_labels, labels);
        }

        self.addScalar(.shortconv_l_cache, hp.n_shortconv_l_cache);

        self.addArr(.xielu_alpha_n, f32, &hp.xielu_alpha_n, false);
        self.addArr(.xielu_alpha_p, f32, &hp.xielu_alpha_p, false);
        self.addArr(.xielu_beta, f32, &hp.xielu_beta, false);
        self.addArr(.xielu_eps, f32, &hp.xielu_eps, false);

        self.addScalar(.attn_res_block_size, hp.attn_res_block_size);
        self.addScalar(.activation_situ_beta, hp.situ_beta);
        self.addScalar(.activation_situ_linear_beta, hp.situ_linear_beta);

        self.addScalar(.dense_2_feat_in, hp.dense_2_feat_in);
        self.addScalar(.dense_2_feat_out, hp.dense_2_feat_out);
        self.addScalar(.dense_3_feat_in, hp.dense_3_feat_in);
        self.addScalar(.dense_3_feat_out, hp.dense_3_feat_out);
    }

    /// Ports `add_tensors_from_model` (llama-model-saver.cpp:475 @8642541b2).
    /// Looped architectures alias physical tensors across cache slots, so
    /// each layer tensor is saved once, by pointer.
    ///
    /// Return: nothing.
    pub fn addTensorsFromModel(self: *Saver) void {
        @setEvalBranchQuota(1_000_000);
        const model = self.model;
        // Some models use one tensor for tok_embd and output.
        if (model.output != null and !std.mem.eql(u8, std.mem.sliceTo(&model.output.?.name, 0), std.mem.sliceTo(&model.tok_embd.?.name, 0))) {
            self.addTensor(model.tok_embd);
        }
        inline for (model_fields) |f| {
            if (@hasField(Model, f)) self.addTensor(@field(model, f));
        }

        // A different tensor with an existing name still asserts in `addTensor`.
        var seen: std.AutoHashMapUnmanaged(*const Tensor, void) = .empty;
        defer seen.deinit(allocator);
        for (model.layers) |*layer| {
            inline for (layer_fields) |path| {
                if (comptime PathType(Layer, path) != null) {
                    if (getPath(layer, path)) |t| {
                        const slot = seen.getOrPut(allocator, t) catch oom();
                        if (!slot.found_existing) self.addTensor(t);
                    }
                }
            }
        }
    }

    /// Ports `save` to a path (llama-model-saver.cpp:523 @8642541b2).
    ///
    /// Parameters:
    /// - `path_model`: where to write.
    ///
    /// Return: nothing.
    pub fn save(self: *Saver, path_model: [*:0]const u8) void {
        _ = c.gguf_write_to_file(self.gguf_ctx, path_model, false);
    }
};

/// Ports `llama_vocab`'s `get_bpe_merges` (llama-vocab.cpp:4614 @8642541b2):
/// each merge as `left right`, at its rank; a rank no pair holds stays
/// empty.
///
/// Return: the merges, owned by the caller; free with `freeMerges`.
fn bpeMerges(vocab: *const vm.Vocab) [][]const u8 {
    var max_rank: i32 = -1;
    var it = vocab.bpe_ranks.map.iterator();
    while (it.next()) |e| max_rank = @max(max_rank, e.value_ptr.*);
    const result = allocator.alloc([]const u8, @intCast(max_rank + 1)) catch oom();
    @memset(result, "");
    const buf = vocab.bpe_ranks.buf.items;
    it = vocab.bpe_ranks.map.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        result[@intCast(e.value_ptr.*)] = std.mem.concat(allocator, u8, &.{ buf[k.l_off..][0..k.l_len], " ", buf[k.r_off..][0..k.r_len] }) catch oom();
    }
    return result;
}

fn freeMerges(merges: [][]const u8) void {
    for (merges) |m| if (m.len != 0) allocator.free(m);
    allocator.free(merges);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Saver);
}

test "the layer walk names each field once" {
    for (layer_fields, 0..) |a, i| {
        for (layer_fields[i + 1 ..]) |b| try std.testing.expect(!std.mem.eql(u8, a, b));
    }
}
