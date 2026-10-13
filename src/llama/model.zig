//! A model, as far as its metadata: architecture, size class,
//! hyperparameters, vocabulary.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-model.cpp` -- `llama_expert_gating_func_name`,
//!   `LLAMA_ROPE_SCALING_TYPES` and its two lookups, `llama_model_create`,
//!   `llama_model_base`'s `load_stats`, `load_hparams` and `load_vocab`,
//!   `llama_model`'s `arch_name`, `type_name`, `desc` and `print_info`, and
//!   the conditional cases of `llama_model_rope_type`
//! - `llama.cpp/src/models/qwen35.cpp` -- `llama_model_qwen35`'s
//!   `load_arch_hparams`
//! - `llama.cpp/src/llama.cpp` -- the metadata half of `llama_model_load`
//!
//! All at v0.3.0 (`c1d0e7a00`). The size names and the fixed RoPE rules are
//! generated into `model_table.zig`.
//!
//! # Scope
//!
//! Everything `llama_model_load` does before `load_tensors`. The C++ makes
//! one subclass per architecture; the port has one `Model` and dispatches
//! the architecture-specific step on `arch`, and only Qwen3.5's exists so
//! far (PLAN.md, Decision 5). Any other architecture the C++ knows loads a
//! vocabulary-only model completely, and otherwise stops at `error.Unported`
//! once the base hyperparameters are read -- distinct from a C++ exception,
//! so the gate can tell "not yet ported" from "the C++ would have thrown".

const std = @import("std");
const impl = @import("impl.zig");
const arch = @import("arch.zig");
const libcxx = @import("libcxx.zig");
const gguf_meta = @import("gguf_meta.zig");
const hparams_mod = @import("hparams.zig");
const loader = @import("loader.zig");
const table = @import("model_table.zig");
const vm = @import("vocab/module.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Hparams = hparams_mod.Hparams;

pub const Type = table.Type;

fn oom() noreturn {
    impl.abort("llama-model.cpp", 0, "out of memory");
}

/// Ports `llama_expert_gating_func_name` (llama-model.cpp:955 @c1d0e7a00).
pub fn expertGatingFuncName(ty: u32) [:0]const u8 {
    return switch (ty) {
        1 => "softmax",
        2 => "sigmoid",
        4 => "sqrtsoftplus",
        else => "unknown",
    };
}

/// Ports `LLAMA_ROPE_SCALING_TYPES` (llama-model.cpp:964 @c1d0e7a00), in
/// the map's key order.
const rope_scaling_types = [_]struct { c_int, [:0]const u8 }{
    .{ c.LLAMA_ROPE_SCALING_TYPE_NONE, "none" },
    .{ c.LLAMA_ROPE_SCALING_TYPE_LINEAR, "linear" },
    .{ c.LLAMA_ROPE_SCALING_TYPE_YARN, "yarn" },
    .{ c.LLAMA_ROPE_SCALING_TYPE_LONGROPE, "longrope" },
};

/// Ports `llama_rope_scaling_type_name` (llama-model.cpp:971 @c1d0e7a00).
///
/// Return: the name, or `error.Thrown` with `std::out_of_range` for a type
/// the map lacks, as `map::at` throws.
pub fn ropeScalingTypeName(ty: c_int, exc: *Exception) Thrown![:0]const u8 {
    for (rope_scaling_types) |e| if (e[0] == ty) return e[1];
    return exc.throw(.out_of_range, "map::at:  key not found", .{});
}

/// Ports `llama_rope_scaling_type_from_string` (llama-model.cpp:975 @c1d0e7a00).
pub fn ropeScalingTypeFromString(name: []const u8) c_int {
    for (rope_scaling_types) |e| if (std.mem.eql(u8, e[1], name)) return e[0];
    return c.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

/// The model's parameters that this half of loading reads.
pub const Params = struct {
    vocab_only: bool = false,
    no_alloc: bool = false,
    split_mode: c.enum_llama_split_mode = c.LLAMA_SPLIT_MODE_LAYER,
};

/// `llama_model`, as far as `print_info`.
pub const Model = struct {
    type: Type = .unknown,
    arch: arch.Arch = .unknown,
    /// `general.name`.
    name: std.ArrayList(u8) = .empty,
    hparams: Hparams = .{},
    vocab: vm.Vocab,
    classifier_labels: std.ArrayList([]u8) = .empty,
    /// Every scalar metadata value as text, in `std::unordered_map` order.
    gguf_kv: libcxx.StringMap([]u8),

    // `llama_model::impl`
    n_elements: u64 = 0,
    n_bytes: usize = 0,
    desc_str: std.ArrayList(u8) = .empty,
    ftype: c_uint = c.LLAMA_FTYPE_ALL_F32,

    /// Ports `llama_model_create` (llama-model.cpp:355 @c1d0e7a00), through
    /// its architecture overload (llama-model.cpp:342 @c1d0e7a00).
    ///
    /// Every architecture the C++ has a class for is accepted: the one
    /// architecture-specific step is `loadArchHparams`, which is where an
    /// unported one stops.
    ///
    /// Return: the model, or `error.Thrown` where the C++ throws.
    pub fn create(ml: *const loader.Loader, params: Params, exc: *Exception) Thrown!Model {
        const a = ml.getArch();
        if (a == .unknown) return exc.throw(.runtime_error, "unknown model architecture: '{s}'", .{ml.getArchName()});
        if (!table.mapped[@intCast(@backingInt(a))]) {
            return exc.throw(.runtime_error, "unsupported model architecture: '{s}'", .{arch.archName(a)});
        }
        if (params.split_mode == c.LLAMA_SPLIT_MODE_TENSOR and !arch.supportsSmTensor(a)) {
            return exc.throw(.runtime_error, "LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '{s}'", .{arch.archName(a)});
        }
        var m: Model = .{
            .arch = a,
            .vocab = vm.Vocab.init() catch oom(),
            .gguf_kv = libcxx.StringMap([]u8).init(allocator) catch oom(),
        };
        m.name.appendSlice(allocator, "n/a") catch oom();
        return m;
    }

    /// Return: nothing.
    pub fn deinit(self: *Model) void {
        self.name.deinit(allocator);
        self.vocab.deinit();
        for (self.classifier_labels.items) |l| allocator.free(l);
        self.classifier_labels.deinit(allocator);
        var it = self.gguf_kv.iterator();
        while (it.next()) |e| allocator.free(e.value.*);
        self.gguf_kv.deinit();
        self.desc_str.deinit(allocator);
    }

    /// Ports `llama_model_base`'s `load_stats` (llama-model.cpp:1173 @c1d0e7a00).
    pub fn loadStats(self: *Model, ml: *const loader.Loader) void {
        self.n_elements = ml.n_elements;
        self.n_bytes = ml.n_bytes;
    }

    /// Ports `llama_model_base`'s `load_vocab` (llama-model.cpp:1366 @c1d0e7a00).
    pub fn loadVocab(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
        try self.vocab.load(&ml.meta, exc);
    }

    /// Ports `llama_model_base`'s `load_hparams` (llama-model.cpp:1178 @c1d0e7a00).
    pub fn loadHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) (Thrown || error{Unported})!void {
        const ctx = ml.meta.ctx;
        const meta = &ml.meta;
        const hp = &self.hparams;

        // Every scalar value as text, the first of a repeated key kept.
        var i: i64 = 0;
        while (i < c.gguf_get_n_kv(ctx)) : (i += 1) {
            if (c.gguf_get_kv_type(ctx, i) == c.GGUF_TYPE_ARRAY) continue;
            var value: std.ArrayList(u8) = .empty;
            loader.kvToStr(ctx, i, &value);
            const slot = self.gguf_kv.getOrPut(gguf_meta.cString(c.gguf_get_key(ctx, i)), &.{}) catch oom();
            if (slot.found) {
                value.deinit(allocator);
            } else {
                slot.value.* = value.toOwnedSlice(allocator) catch oom();
            }
        }

        if (try meta.getKey([]const u8, .general_name, false, exc)) |n| {
            self.name.clearRetainingCapacity();
            self.name.appendSlice(allocator, n) catch oom();
        }

        // Everything past this point is not vocabulary-related; a CLIP
        // model loads tensors only.
        if (hp.vocab_only or ml.getArch() == .clip) return;

        hp.n_ctx_train = (try meta.getKey(u32, .context_length, true, exc)).?;
        hp.n_embd = (try meta.getKey(u32, .embedding_length, true, exc)).?;
        if (try meta.getKey(u32, .embedding_length_out, false, exc)) |v| hp.n_embd_out_impl = v;
        if (try meta.getKey(bool, .attention_causal, false, exc)) |v| hp.causal_attn = v;
        hp.pooling_type = try meta.getPoolingType(.pooling_type, false, exc);
        hp.n_layer_all = (try meta.getKey(u32, .block_count, true, exc)).?;
        impl.assert(hp.n_layer_all > 0 and hp.n_layer_all <= hparams_mod.max_layers, "llama-model.cpp", 1207, "hparams.n_layer_all > 0 && hparams.n_layer_all <= LLAMA_MAX_LAYERS");
        if (try meta.getKey(u32, .expert_count, false, exc)) |v| hp.n_expert = v;
        if (try meta.getKey(u32, .expert_used_count, false, exc)) |v| hp.n_expert_used = v;
        if (try meta.getKey(u32, .expert_group_count, false, exc)) |v| hp.n_expert_groups = v;
        if (try meta.getKey(u32, .expert_group_used_count, false, exc)) |v| hp.n_group_used = v;

        if (self.arch == .hunyuan_vl or self.arch == .hunyuan_dense) {
            if (hp.n_expert <= 1) {
                hp.n_expert = 0;
                hp.n_expert_used = 0;
            }
        }

        if (self.arch == .wavtokenizer_dec) {
            hp.n_embd = (try meta.getKey(u32, .features_length, true, exc)).?;
            hp.n_embd_out_impl = (try meta.getKey(u32, .embedding_length, true, exc)).?;
            hp.posnet.n_embd = (try meta.getKey(u32, .posnet_embedding_length, true, exc)).?;
            hp.posnet.n_layer = (try meta.getKey(u32, .posnet_block_count, true, exc)).?;
            hp.convnext.n_embd = (try meta.getKey(u32, .convnext_embedding_length, true, exc)).?;
            hp.convnext.n_layer = (try meta.getKey(u32, .convnext_block_count, true, exc)).?;
            impl.assert(hp.posnet.n_layer <= hp.n_layer_all, "llama-model.cpp", 1230, "hparams.posnet.n_layer <= hparams.n_layer_all");
            impl.assert(hp.convnext.n_layer <= hp.n_layer_all, "llama-model.cpp", 1231, "hparams.convnext.n_layer <= hparams.n_layer_all");
        }

        impl.assert(hp.n_expert <= hparams_mod.max_experts, "llama-model.cpp", 1234, "hparams.n_expert <= LLAMA_MAX_EXPERTS");
        impl.assert(hp.n_expert_used <= hp.n_expert, "llama-model.cpp", 1235, "hparams.n_expert_used <= hparams.n_expert");
        if (hp.n_expert > 0) {
            impl.assert(hp.n_expert_used > 0, "llama-model.cpp", 1237, "hparams.n_expert_used > 0");
            impl.assert(hp.n_expert_groups < hp.n_expert, "llama-model.cpp", 1238, "hparams.n_expert_groups < hparams.n_expert");
            if (hp.n_expert_groups > 1) {
                impl.assert(hp.n_expert % hp.n_expert_groups == 0, "llama-model.cpp", 1240, "hparams.n_expert % hparams.n_expert_groups == 0");
                impl.assert(hp.n_group_used > 0, "llama-model.cpp", 1241, "hparams.n_group_used > 0");
                impl.assert(hp.n_group_used < hp.n_expert_groups, "llama-model.cpp", 1242, "hparams.n_group_used < hparams.n_expert_groups");
            }
        } else {
            impl.assert(hp.n_expert_used == 0, "llama-model.cpp", 1245, "hparams.n_expert_used == 0");
            impl.assert(hp.n_expert_groups == 0, "llama-model.cpp", 1246, "hparams.n_expert_groups == 0");
        }

        hp.n_head_arr = @splat(0);
        hp.n_head_kv_arr = @splat(0);
        hp.n_ff_arr = @splat(0);
        hp.rope_sections = @splat(0);
        hp.rope_pattern = @splat(1);
        hp.is_swa_impl = @splat(0);
        hp.is_recr_impl = @splat(@intFromBool(arch.isRecurrent(ml.getArch())));
        hp.is_indexer_full_impl = @splat(0);
        hp.xielu_alpha_n = @splat(0);
        hp.xielu_alpha_p = @splat(0);
        hp.xielu_beta = @splat(0);
        hp.xielu_eps = @splat(0);
        hp.swiglu_clamp_exp = @splat(0);
        hp.swiglu_clamp_shexp = @splat(0);

        _ = try meta.getKeyOrArr(u32, .feed_forward_length, &hp.n_ff_arr, hp.nLayer(), false, exc);
        _ = try meta.getKeyOrArr(u32, .attention_head_count, &hp.n_head_arr, hp.nLayer(), false, exc);

        hp.deepstack_mapping_arr = @splat(-1);

        // n_head_kv is optional, and defaults to n_head.
        hp.n_head_kv_arr = hp.n_head_arr;
        _ = try meta.getKeyOrArr(u32, .attention_head_count_kv, &hp.n_head_kv_arr, hp.nLayer(), false, exc);

        hp.rope_finetuned = (try meta.getKey(bool, .rope_scaling_finetuned, false, exc)) orelse false;

        hp.n_ctx_orig_yarn = hp.n_ctx_train;
        if (try meta.getKey(u32, .rope_scaling_orig_ctx_len, false, exc)) |v| hp.n_ctx_orig_yarn = v;

        hp.rope_freq_base_train = 10000.0;
        if (try meta.getKey(f32, .rope_freq_base, false, exc)) |v| hp.rope_freq_base_train = v;

        const rope_scaling = (try meta.getKey([]const u8, .rope_scaling_type, false, exc)) orelse "linear";
        hp.rope_scaling_type_train = ropeScalingTypeFromString(rope_scaling);
        impl.assert(hp.rope_scaling_type_train != c.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED, "llama-model.cpp", 1292, "hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED");

        // rope_freq_scale is the inverse of the stored factor; either key.
        var ropescale: f32 = 0.0;
        if (try meta.getKey(f32, .rope_scaling_factor, false, exc)) |v| {
            ropescale = v;
        } else if (try meta.getKey(f32, .rope_scale_linear, false, exc)) |v| {
            ropescale = v;
        }
        hp.rope_freq_scale_train = if (ropescale == 0.0) 1.0 else 1.0 / ropescale;

        if (try meta.getKey(f32, .rope_scaling_attn_factor, false, exc)) |v| hp.rope_attn_factor = v;
        if (try meta.getKey(f32, .rope_scaling_alpha, false, exc)) |v| hp.rope_scaling_alpha = v;

        // Non-transformer models have no attention heads.
        if (hp.nHead(0) > 0) {
            hp.n_embd_head_k_full = hp.n_embd / hp.nHead(0);
            if (try meta.getKey(u32, .attention_key_length, false, exc)) |v| hp.n_embd_head_k_full = v;
            hp.n_embd_head_v_full = hp.n_embd / hp.nHead(0);
            if (try meta.getKey(u32, .attention_value_length, false, exc)) |v| hp.n_embd_head_v_full = v;

            hp.n_rot_full = hp.n_embd_head_k_full;
            if (try meta.getKey(u32, .rope_dimension_count, false, exc)) |v| hp.n_rot_full = v;

            if (self.arch == .llama or self.arch == .deci or self.arch == .falcon or self.arch == .llama_embed) {
                if (hp.n_rot_full != hp.n_embd_head_k_full) {
                    return exc.throw(.runtime_error, "invalid n_rot: {d}, expected {d}", .{ hp.n_rot_full, hp.n_embd_head_k_full });
                }
            }
        } else {
            hp.n_rot_full = 0;
            hp.n_embd_head_k_full = 0;
            hp.n_embd_head_v_full = 0;
        }

        // Head size and n_rot for SWA layers.
        hp.n_embd_head_k_swa = hp.n_embd_head_k_full;
        hp.n_embd_head_v_swa = hp.n_embd_head_v_full;
        if (try meta.getKey(u32, .attention_key_length_swa, false, exc)) |v| hp.n_embd_head_k_swa = v;
        if (try meta.getKey(u32, .attention_value_length_swa, false, exc)) |v| hp.n_embd_head_v_swa = v;
        hp.n_rot_swa = hp.n_rot_full;
        if (try meta.getKey(u32, .rope_dimension_count_swa, false, exc)) |v| hp.n_rot_swa = v;

        // Classifier models.
        if (try meta.getArrList([]const u8, .classifier_output_labels, false, exc)) |labels| {
            defer allocator.free(labels);
            for (self.classifier_labels.items) |l| allocator.free(l);
            self.classifier_labels.clearRetainingCapacity();
            for (labels) |l| self.classifier_labels.append(allocator, allocator.dupe(u8, l) catch oom()) catch oom();
        }
        if (self.classifier_labels.items.len != 0) hp.n_cls_out = @truncate(self.classifier_labels.items.len);

        try self.loadArchHparams(ml, exc);

        self.n_bytes = ml.n_bytes;

        self.desc_str.clearRetainingCapacity();
        self.desc_str.print(allocator, "{s} {s} {s}", .{ self.archName(), self.typeName(), ml.ftypeNameOf() }) catch oom();

        self.ftype = ml.ftype;

        if (hp.f_max_alibi_bias > 0.0) hp.use_alibi = true;

        hp.rope_type = self.ropeType();
    }

    /// The architecture's own `load_arch_hparams`.
    fn loadArchHparams(self: *Model, ml: *const loader.Loader, exc: *Exception) (Thrown || error{Unported})!void {
        switch (self.arch) {
            .qwen35 => try self.loadArchHparamsQwen35(ml, exc),
            else => return error.Unported,
        }
    }

    /// Ports `llama_model_qwen35`'s `load_arch_hparams` (qwen35.cpp:4 @c1d0e7a00).
    fn loadArchHparamsQwen35(self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void {
        const meta = &ml.meta;
        const hp = &self.hparams;
        hp.f_norm_rms_eps = (try meta.getKey(f32, .attention_layernorm_rms_eps, true, exc)).?;
        _ = try meta.getKeyOrArr(c_int, .rope_dimension_sections, &hp.rope_sections, 4, true, exc);

        // Linear attention (gated delta net).
        hp.ssm_d_conv = (try meta.getKey(u32, .ssm_conv_kernel, true, exc)).?;
        hp.ssm_d_inner = (try meta.getKey(u32, .ssm_inner_size, true, exc)).?;
        hp.ssm_d_state = (try meta.getKey(u32, .ssm_state_size, true, exc)).?;
        hp.ssm_dt_rank = (try meta.getKey(u32, .ssm_time_step_rank, true, exc)).?;
        hp.ssm_n_group = (try meta.getKey(u32, .ssm_group_count, true, exc)).?;

        // NextN/MTP: an extra decoder block appended beyond the main stack.
        if (try meta.getKey(u32, .nextn_predict_layers, false, exc)) |v| hp.n_layer_nextn = v;
        impl.assert(hp.n_layer_nextn < hp.n_layer_all, "qwen35.cpp", 17, "hparams.n_layer_nextn < hparams.n_layer_all && \"n_layer_nextn must be < n_layer_impl\"");

        // Recurrent (linear attention) layers; MTP layers are dense.
        if (!try meta.getKeyOrArr(u32, .attention_recurrent_layers, &hp.is_recr_impl, hp.n_layer_all, false, exc)) {
            var full_attn_interval: u32 = 4;
            if (try meta.getKey(u32, .full_attention_interval, false, exc)) |v| full_attn_interval = v;
            var i: u32 = 0;
            while (i < hp.n_layer_all) : (i += 1) {
                hp.is_recr_impl[i] = @intFromBool(i < hp.nLayer() and remU32(i + 1, full_attn_interval) != 0);
            }
        }

        self.type = switch (hp.nLayer()) {
            24 => if (hp.n_embd == 1024) .@"0_8b" else .@"2b",
            32 => if (hp.n_embd == 2560) .@"4b" else .@"9b",
            64 => .@"27b",
            else => .unknown,
        };
    }

    /// Ports `llama_model`'s `arch_name` (llama-model.cpp:1808 @c1d0e7a00).
    pub fn archName(self: *const Model) [:0]const u8 {
        return arch.archName(self.arch);
    }

    /// Ports `llama_model`'s `type_name` (llama-model.cpp:1812 @c1d0e7a00).
    pub fn typeName(self: *const Model) [:0]const u8 {
        return table.type_names[@intCast(@backingInt(self.type))];
    }

    /// Ports `llama_model`'s `desc` (llama-model.cpp:1816 @c1d0e7a00).
    pub fn desc(self: *const Model) []const u8 {
        return self.desc_str.items;
    }

    /// Ports `llama_model_rope_type` (llama-model.cpp:2736 @c1d0e7a00): the
    /// generated rules, and the four that read the hyperparameters.
    pub fn ropeType(self: *const Model) c.enum_llama_rope_type {
        return switch (table.rope_rules[@intCast(@backingInt(self.arch))]) {
            .fixed => |t| t,
            .abort => impl.abort("llama-model.cpp", 2907, "unknown architecture"),
            .conditional => switch (self.arch) {
                .dflash => if (self.hparams.dsv4_hc_mult > 0) c.LLAMA_ROPE_TYPE_NORM else c.LLAMA_ROPE_TYPE_NEOX,
                .glm4 => if (self.hparams.useMrope()) c.LLAMA_ROPE_TYPE_MROPE else c.LLAMA_ROPE_TYPE_NORM,
                .glm4_moe, .hunyuan_vl => if (self.hparams.useMrope()) c.LLAMA_ROPE_TYPE_MROPE else c.LLAMA_ROPE_TYPE_NEOX,
                else => unreachable,
            },
        };
    }

    /// Ports `llama_model`'s `print_info` (llama-model.cpp:1872 @c1d0e7a00).
    pub fn printInfo(self: *const Model, exc: *Exception) Thrown!void {
        const hp = &self.hparams;
        const f = "print_info";
        const rope_scaling_type = try ropeScalingTypeName(hp.rope_scaling_type_train, exc);

        impl.logInfo("%s: arch                  = %s\n", .{ f, self.archName().ptr });
        impl.logInfo("%s: vocab_only            = %d\n", .{ f, @as(c_int, @intFromBool(hp.vocab_only)) });
        impl.logInfo("%s: no_alloc              = %d\n", .{ f, @as(c_int, @intFromBool(hp.no_alloc)) });

        if (!hp.vocab_only) {
            var buf: std.ArrayList(u8) = .empty;
            defer buf.deinit(allocator);
            const n = hp.n_layer_all;

            impl.logInfo("%s: n_ctx_train           = %u\n", .{ f, hp.n_ctx_train });
            impl.logInfo("%s: n_embd_inp            = %u\n", .{ f, hp.nEmbdInp() });
            impl.logInfo("%s: n_embd                = %u\n", .{ f, hp.n_embd });
            impl.logInfo("%s: n_embd_out            = %u\n", .{ f, hp.nEmbdOut() });
            impl.logInfo("%s: n_layer               = %u\n", .{ f, hp.nLayer() });
            impl.logInfo("%s: n_layer_all           = %u\n", .{ f, hp.n_layer_all });
            impl.logInfo("%s: n_head                = %s\n", .{ f, printF(&buf, hp, Hparams.nHead, n) });
            impl.logInfo("%s: n_head_kv             = %s\n", .{ f, printF(&buf, hp, Hparams.nHeadKv, n) });
            impl.logInfo("%s: n_rot                 = %u\n", .{ f, hp.n_rot_full });
            impl.logInfo("%s: n_swa                 = %u\n", .{ f, hp.n_swa });
            impl.logInfo("%s: is_swa_any            = %u\n", .{ f, @as(c_uint, @intFromBool(hp.isSwaAny())) });
            impl.logInfo("%s: n_embd_head_k         = %u\n", .{ f, hp.n_embd_head_k_full });
            impl.logInfo("%s: n_embd_head_v         = %u\n", .{ f, hp.n_embd_head_v_full });
            impl.logInfo("%s: n_gqa                 = %s\n", .{ f, printF(&buf, hp, Hparams.nGqa, n) });
            impl.logInfo("%s: n_embd_k_gqa          = %s\n", .{ f, printF(&buf, hp, Hparams.nEmbdKGqa, n) });
            impl.logInfo("%s: n_embd_v_gqa          = %s\n", .{ f, printF(&buf, hp, Hparams.nEmbdVGqa, n) });
            impl.logInfo("%s: f_norm_eps            = %.1e\n", .{ f, @as(f64, hp.f_norm_eps) });
            impl.logInfo("%s: f_norm_rms_eps        = %.1e\n", .{ f, @as(f64, hp.f_norm_rms_eps) });
            impl.logInfo("%s: f_clamp_kqv           = %.1e\n", .{ f, @as(f64, hp.f_clamp_kqv) });
            impl.logInfo("%s: f_max_alibi_bias      = %.1e\n", .{ f, @as(f64, hp.f_max_alibi_bias) });
            impl.logInfo("%s: f_logit_scale         = %.1e\n", .{ f, @as(f64, hp.f_logit_scale) });
            impl.logInfo("%s: f_attn_scale          = %.1e\n", .{ f, @as(f64, hp.f_attention_scale) });
            impl.logInfo("%s: f_attn_value_scale    = %.4f\n", .{ f, @as(f64, hp.f_attn_value_scale) });
            impl.logInfo("%s: n_ff                  = %s\n", .{ f, printF(&buf, hp, Hparams.nFf, n) });
            impl.logInfo("%s: n_expert              = %u\n", .{ f, hp.n_expert });
            impl.logInfo("%s: n_expert_used         = %u\n", .{ f, hp.n_expert_used });
            impl.logInfo("%s: n_expert_groups       = %d\n", .{ f, hp.n_expert_groups });
            impl.logInfo("%s: n_group_used          = %d\n", .{ f, hp.n_group_used });
            impl.logInfo("%s: causal attn           = %d\n", .{ f, @as(c_int, @intFromBool(hp.causal_attn)) });
            impl.logInfo("%s: pooling type          = %d\n", .{ f, hp.pooling_type });
            impl.logInfo("%s: rope type             = %d\n", .{ f, hp.rope_type });
            impl.logInfo("%s: rope scaling          = %s\n", .{ f, rope_scaling_type.ptr });
            impl.logInfo("%s: freq_base_train       = %.1f\n", .{ f, @as(f64, hp.rope_freq_base_train) });
            impl.logInfo("%s: freq_scale_train      = %g\n", .{ f, @as(f64, hp.rope_freq_scale_train) });
            if (hp.swa_type != 0) {
                impl.logInfo("%s: freq_base_swa         = %.1f\n", .{ f, @as(f64, hp.rope_freq_base_train_swa) });
                impl.logInfo("%s: freq_scale_swa        = %g\n", .{ f, @as(f64, hp.rope_freq_scale_train_swa) });
                impl.logInfo("%s: n_embd_head_k_swa     = %u\n", .{ f, hp.n_embd_head_k_swa });
                impl.logInfo("%s: n_embd_head_v_swa     = %u\n", .{ f, hp.n_embd_head_v_swa });
                impl.logInfo("%s: n_rot_swa             = %u\n", .{ f, hp.n_rot_swa });
            }
            impl.logInfo("%s: n_ctx_orig_yarn       = %u\n", .{ f, hp.n_ctx_orig_yarn });
            impl.logInfo("%s: rope_yarn_log_mul     = %.4f\n", .{ f, @as(f64, hp.rope_yarn_log_mul) });
            impl.logInfo("%s: rope_finetuned        = %s\n", .{ f, @as([*:0]const u8, if (hp.rope_finetuned) "yes" else "unknown") });
            if (self.arch == .granite) {
                var any = false;
                for (hp.deepstack_mapping_arr) |e| any = any or e >= 0;
                if (any) impl.logInfo("%s: deepstack_mapping_arr = %s\n", .{ f, printF(&buf, hp, deepstackAt, n) });
            }
            const s = hp.rope_sections;
            if (s[0] != 0 or s[1] != 0 or s[2] != 0 or s[3] != 0) {
                impl.logInfo("%s: mrope sections        = [%d, %d, %d, %d]\n", .{ f, s[0], s[1], s[2], s[3] });
            }
            if (self.classifier_labels.items.len != 0) {
                impl.logInfo("%s: n_cls_out             = %u\n", .{ f, hp.n_cls_out });
                for (self.classifier_labels.items, 0..) |label, i| {
                    impl.logInfo("%s: cls_label[%2zu]         = %.*s\n", .{ f, i, @as(c_int, @intCast(label.len)), label.ptr });
                }
            }

            switch (self.arch) {
                .mamba, .mamba2, .jamba, .falcon_h1, .plamo2, .granite_hybrid, .qwen3next, .qwen35, .qwen35moe, .nemotron_h, .nemotron_h_moe => {
                    impl.logInfo("%s: ssm_d_conv            = %u\n", .{ f, hp.ssm_d_conv });
                    impl.logInfo("%s: ssm_d_inner           = %u\n", .{ f, hp.ssm_d_inner });
                    impl.logInfo("%s: ssm_d_state           = %u\n", .{ f, hp.ssm_d_state });
                    impl.logInfo("%s: ssm_dt_rank           = %u\n", .{ f, hp.ssm_dt_rank });
                    impl.logInfo("%s: ssm_n_group           = %u\n", .{ f, hp.ssm_n_group });
                    impl.logInfo("%s: ssm_dt_b_c_rms        = %d\n", .{ f, @as(c_int, @intFromBool(hp.ssm_dt_b_c_rms)) });
                },
                else => {},
            }

            impl.logInfo("%s: model type            = %s\n", .{ f, self.typeName().ptr });
            const ne: f64 = @floatFromInt(self.n_elements);
            if (ne >= 1e12) {
                impl.logInfo("%s: model params          = %.2f T\n", .{ f, ne * 1e-12 });
            } else if (ne >= 1e9) {
                impl.logInfo("%s: model params          = %.2f B\n", .{ f, ne * 1e-9 });
            } else if (ne >= 1e6) {
                impl.logInfo("%s: model params          = %.2f M\n", .{ f, ne * 1e-6 });
            } else {
                impl.logInfo("%s: model params          = %.2f K\n", .{ f, ne * 1e-3 });
            }

            impl.logInfo("%s: general.name          = %.*s\n", .{ f, @as(c_int, @intCast(self.name.items.len)), self.name.items.ptr });

            self.printArchInfo(exc);
        }

        self.vocab.printInfo();
    }

    /// The architecture-specific tail of `print_info`.
    fn printArchInfo(self: *const Model, _: *Exception) void {
        const hp = &self.hparams;
        const f = "print_info";
        const a = self.arch;
        if (a == .deepseek) {
            impl.logInfo("%s: n_layer_dense_lead    = %d\n", .{ f, hp.n_layer_dense_lead });
            impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.n_ff_exp });
            impl.logInfo("%s: n_expert_shared       = %d\n", .{ f, hp.n_expert_shared });
            impl.logInfo("%s: expert_weights_scale  = %.1f\n", .{ f, @as(f64, hp.expert_weights_scale) });
        }
        if (a == .deepseek2 or a == .deepseek2ocr or a == .deepseek32 or a == .glm_dsa or a == .dots3note or a == .mistral4) {
            impl.logInfo("%s: n_layer_dense_lead    = %d\n", .{ f, hp.n_layer_dense_lead });
            impl.logInfo("%s: n_lora_q              = %d\n", .{ f, hp.n_lora_q });
            impl.logInfo("%s: n_lora_kv             = %d\n", .{ f, hp.n_lora_kv });
            impl.logInfo("%s: n_embd_head_k_mla     = %d\n", .{ f, hp.nEmbdHeadKMla() });
            impl.logInfo("%s: n_embd_head_v_mla     = %d\n", .{ f, hp.nEmbdHeadVMla() });
            impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.n_ff_exp });
            impl.logInfo("%s: n_expert_shared       = %d\n", .{ f, hp.n_expert_shared });
            impl.logInfo("%s: expert_weights_scale  = %.1f\n", .{ f, @as(f64, hp.expert_weights_scale) });
            impl.logInfo("%s: expert_weights_norm   = %d\n", .{ f, @as(c_int, @intFromBool(hp.expert_weights_norm)) });
            impl.logInfo("%s: expert_gating_func    = %s\n", .{ f, expertGatingFuncName(hp.expert_gating_func).ptr });
        }
        if (a == .qwen2moe) {
            impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.n_ff_exp });
            impl.logInfo("%s: n_ff_shexp            = %d\n", .{ f, hp.n_ff_shexp });
        }
        if (a == .mellum or a == .cohere2moe or a == .qwen3moe or a == .openai_moe or a == .qwen3vlmoe or a == .rnd1) {
            impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.n_ff_exp });
        }
        if (a == .minicpm or a == .granite or a == .granite_moe or a == .granite_hybrid or a == .granite_switch or a == .nemotron_h_moe) {
            impl.logInfo("%s: f_embedding_scale     = %f\n", .{ f, @as(f64, hp.f_embedding_scale) });
            impl.logInfo("%s: f_residual_scale      = %f\n", .{ f, @as(f64, hp.f_residual_scale) });
            impl.logInfo("%s: f_attention_scale     = %f\n", .{ f, @as(f64, hp.f_attention_scale) });
            impl.logInfo("%s: n_ff_shexp            = %d\n", .{ f, hp.n_ff_shexp });
        }
        if (a == .bailingmoe) {
            impl.logInfo("%s: n_layer_dense_lead    = %d\n", .{ f, hp.n_layer_dense_lead });
            impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.n_ff_exp });
            impl.logInfo("%s: n_expert_shared       = %d\n", .{ f, hp.n_expert_shared });
            impl.logInfo("%s: expert_weights_scale  = %.1f\n", .{ f, @as(f64, hp.expert_weights_scale) });
            impl.logInfo("%s: expert_weights_norm   = %d\n", .{ f, @as(c_int, @intFromBool(hp.expert_weights_norm)) });
        }
        if (a == .bailingmoe2 or a == .bailingmoe3) {
            impl.logInfo("%s: n_layer_dense_lead    = %d\n", .{ f, hp.n_layer_dense_lead });
            impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.n_ff_exp });
            impl.logInfo("%s: n_ff_shexp            = %d\n", .{ f, hp.n_ff_shexp });
            impl.logInfo("%s: n_expert_shared       = %d\n", .{ f, hp.n_expert_shared });
            impl.logInfo("%s: expert_weights_scale  = %.1f\n", .{ f, @as(f64, hp.expert_weights_scale) });
            impl.logInfo("%s: expert_weights_norm   = %d\n", .{ f, @as(c_int, @intFromBool(hp.expert_weights_norm)) });
            impl.logInfo("%s: expert_gating_func    = %s\n", .{ f, expertGatingFuncName(hp.expert_gating_func).ptr });
            impl.logInfo("%s: n_layer_nextn         = %d\n", .{ f, hp.n_layer_nextn });
        }
        if (a == .smallthinker or a == .lfm2moe) {
            impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.n_ff_exp });
            impl.logInfo("%s: expert_gating_func    = %s\n", .{ f, expertGatingFuncName(hp.expert_gating_func).ptr });
        }
        if (a == .grovemoe) {
            impl.logInfo("%s: n_ff_exp              = %d\n", .{ f, hp.n_ff_exp });
            impl.logInfo("%s: n_ff_chexp            = %d\n", .{ f, hp.n_ff_chexp });
            impl.logInfo("%s: n_group_experts       = %d\n", .{ f, hp.n_group_experts });
            impl.logInfo("%s: expert_group_scale    = %.2f\n", .{ f, @as(f64, hp.expert_group_scale) });
        }
    }
};

/// `uint32_t % uint32_t` as aarch64 computes it: `udiv` then `msub`, and
/// `udiv` by zero gives zero, so `a % 0 == a`. The C++ is undefined there;
/// this is what the reference binary does.
fn remU32(a: u32, b: u32) u32 {
    return if (b == 0) a else a % b;
}

fn deepstackAt(hp: *const Hparams, il: u32) u32 {
    return @bitCast(hp.deepstack_mapping_arr[il]);
}

/// `print_f` in `print_info`: one value for every layer, or all of them in
/// brackets when they differ. The lambda returns `int32_t`, so a value past
/// `INT32_MAX` prints negative.
fn printF(buf: *std.ArrayList(u8), hp: *const Hparams, comptime get: fn (*const Hparams, u32) u32, n: u32) [*:0]const u8 {
    buf.clearRetainingCapacity();
    var is_var = false;
    const v0: i32 = @bitCast(get(hp, 0));
    var il: u32 = 0;
    while (il < n) : (il += 1) if (@as(i32, @bitCast(get(hp, il))) != v0) {
        is_var = true;
    };
    if (is_var) {
        buf.append(allocator, '[') catch oom();
        il = 0;
        while (il < n) : (il += 1) {
            buf.print(allocator, "{d}", .{@as(i32, @bitCast(get(hp, il)))}) catch oom();
            if (il < n - 1) buf.appendSlice(allocator, ", ") catch oom();
        }
        buf.append(allocator, ']') catch oom();
    } else {
        buf.print(allocator, "{d}", .{v0}) catch oom();
    }
    buf.append(allocator, 0) catch oom();
    return @ptrCast(buf.items.ptr);
}

/// What loading a model's metadata produced.
pub const Loaded = union(enum) {
    /// Loaded; the model is the caller's to `deinit`.
    model: Model,
    /// The C++ would have thrown; `exc` holds the message `llama_model_load` logs.
    thrown,
    /// The architecture is one the port does not have yet.
    unported,
};

/// Ports the metadata half of `llama_model_load` (src/llama.cpp:315 @c1d0e7a00):
/// the loader and its `print_info`, the model, its hyperparameters,
/// vocabulary and statistics, and its `print_info` -- everything before
/// `load_tensors`, with the same messages for what it catches.
///
/// Parameters:
/// - `source`: where the model comes from.
/// - `splits`: the split paths, as the loader takes them.
/// - `params`: the parameters this half reads.
/// - `overrides`: metadata overrides, or null.
/// - `exc`: receives what is thrown, with the C++'s wrapping.
///
/// Return: the outcome; a loaded model is the caller's.
pub fn loadMetadata(
    source: loader.Loader.Source,
    splits: *std.ArrayList([:0]const u8),
    params: Params,
    overrides: ?[*]const c.llama_model_kv_override,
    exc: *Exception,
) Loaded {
    var ml = loader.Loader.init(source, splits, c.LLAMA_LOAD_MODE_MMAP, false, params.no_alloc, false, overrides, exc) catch return .thrown;
    defer ml.deinit();
    ml.printInfo();

    var model = Model.create(&ml, params, exc) catch return .thrown;

    model.hparams.vocab_only = params.vocab_only;
    model.hparams.no_alloc = params.no_alloc;

    model.loadHparams(&ml, exc) catch |e| {
        model.deinit();
        if (e == error.Unported) return .unported;
        rewrap(exc, "error loading model hyperparameters: ");
        return .thrown;
    };
    if (model.arch == .clip) {
        model.deinit();
        _ = exc.throw(.runtime_error, "CLIP cannot be used as main model, use it with --mmproj instead", .{}) catch {};
        return .thrown;
    }
    model.loadVocab(&ml, exc) catch {
        model.deinit();
        rewrap(exc, "error loading model vocabulary: ");
        return .thrown;
    };

    model.loadStats(&ml);
    model.printInfo(exc) catch {
        model.deinit();
        return .thrown;
    };

    if (params.vocab_only) impl.logInfo("%s: vocab only - skipping tensors\n", .{"llama_model_load"});
    return .{ .model = model };
}

/// `throw std::runtime_error(prefix + std::string(e.what()))`.
fn rewrap(exc: *Exception, comptime prefix: []const u8) void {
    var what: [1024]u8 = undefined;
    const w = exc.what();
    @memcpy(what[0..w.len], w);
    exc.set(.runtime_error, prefix ++ "{s}", .{what[0..w.len]});
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "rope scaling names round-trip and an unknown one is unspecified" {
    var exc: Exception = .{};
    try std.testing.expectEqualStrings("yarn", try ropeScalingTypeName(c.LLAMA_ROPE_SCALING_TYPE_YARN, &exc));
    try std.testing.expectEqual(@as(c_int, c.LLAMA_ROPE_SCALING_TYPE_LONGROPE), ropeScalingTypeFromString("longrope"));
    try std.testing.expectEqual(@as(c_int, c.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED), ropeScalingTypeFromString("Linear"));
}

test "a modulus by zero is the dividend, as udiv makes it" {
    try std.testing.expectEqual(@as(u32, 7), remU32(7, 0));
    try std.testing.expectEqual(@as(u32, 3), remU32(7, 4));
}
