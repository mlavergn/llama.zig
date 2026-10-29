//! The model-specific arms of the clip loader's hyperparameter pass.
//!
//! # Provenance
//!
//! Ports the per-projector `switch` inside `clip_model_loader::load_hparams`
//! of `llama.cpp/tools/mtmd/clip.cpp` (lines 1416-2008) at v0.4.1
//! (`b29c606e2`). The code before and after the switch is `clip.zig`'s
//! `Loader.loadHparams`, which calls `load` here. Each arm keeps upstream's
//! keys, required flags, defaults, log lines and messages verbatim; a comment
//! names the `case` labels it covers.
//!
//! # `wa_layer_indexes`
//!
//! Upstream keeps the YoutuVL full-attention layer list in a
//! `std::unordered_set<int32_t>`, and `load_hparams` logs it by iterating.
//! `model.zig` keeps a hash set for membership; the iteration order is
//! libc++'s, reproduced by `waLayerIndexesInSetOrder` from the insertion
//! sequence `load` records, through `libcxx.U32Set`. `std::hash<int>` is
//! the value sign-extended to `size_t`; `U32Set` keys on the value as
//! `uint32_t`, which agrees for every non-negative index -- a negative layer
//! index would bucket differently, and no real model has one.

const std = @import("std");
const impl = @import("impl.zig");
const clip = @import("clip.zig");
const names = @import("names.zig");
const model_mod = @import("model.zig");
const libcxx = @import("llama").libcxx;
const gpa = impl.gpa;

/// `load_hparams`'s locals the switch reads and writes.
pub const Info = struct {
    proj_type: [*:0]const u8,
    is_vision: bool,
    is_audio: bool,
    is_gen_audio: bool,
    log_ffn_op: *[*:0]const u8,
};

/// The insertion sequence of the last `wa_layer_indexes` `load` filled, and
/// the hparams it belongs to.
var wa_insertions: std.ArrayList(i32) = .empty;
var wa_owner: ?*const model_mod.Hparams = null;
var wa_ordered: std.ArrayList(i32) = .empty;

const func = "load_hparams";

/// Ports the model-specific switch of `load_hparams` (mtmd/clip.cpp:1244
/// @b29c606e2), a member of `clip_model_loader`.
pub fn load(ld: *clip.Loader, model: *model_mod.Model, info: Info) clip.Error!void {
    const hparams = &model.hparams;
    var kb: [256]u8 = undefined;

    switch (model.proj_type) {
        // PROJECTOR_TYPE_MLP, MLP_NORM, LDP, LDPV2, COGVLM
        .mlp, .mlp_norm, .ldp, .ldpv2, .cogvlm => {
            hparams.has_llava_projector = model.proj_type != .cogvlm;
            hparams.image_pad_color = .{ 122, 116, 104 };
            if (hparams.image_res_candidates.items.len != 0) {
                hparams.image_resize_pad = .ceil;
                hparams.image_resize_algo = .bicubic;
            } else {
                // llava-1.6 default params
                hparams.image_pad_ov = .none;
                hparams.image_pad_rf = .ceil;
                hparams.image_pad_color_rf = .{ 122, 116, 104 };
            }
        },
        // PROJECTOR_TYPE_GLM_EDGE
        .glm_edge => {
            hparams.image_resize_pad = .ceil;
            hparams.image_resize_algo = .bicubic;
        },
        // PROJECTOR_TYPE_MINICPMV
        .minicpmv => {
            // use default llava-uhd preprocessing params
            if (hparams.minicpmv_version == 0) {
                hparams.minicpmv_version = 2; // default to 2 if not set
            }
        },
        // PROJECTOR_TYPE_MINICPMV4_6
        .minicpmv4_6 => {
            // ViT merger 2x2 + final merger 2x2 = 4x spatial merge per dimension
            hparams.n_merge = 4;
            try ld.getU32(names.KEY_PROJ_SCALE_FACTOR, &hparams.n_merge, false);
            impl.ggmlAssert(hparams.n_merge == 2 or hparams.n_merge == 4, "clip.cpp", 1453, "hparams.n_merge == 2 || hparams.n_merge == 4");

            // borrow wa_layer_indexes for vit_merger insertion point
            var wa_layer_indexes_vec: std.ArrayList(i32) = .empty;
            defer wa_layer_indexes_vec.deinit(gpa);
            try ld.getArrInt(names.KEY_WIN_ATTN_LAYER_INDEXES, &wa_layer_indexes_vec, false);
            if (wa_layer_indexes_vec.items.len != 0) hparams.insert_layer_id = wa_layer_indexes_vec.items[0];
        },
        // PROJECTOR_TYPE_INTERNVL
        .internvl => {
            // older version of internvl doesn't have min/max tiles, we need to provide default values for them to avoid issues
            hparams.preproc_min_tiles = 1;
            hparams.preproc_max_tiles = 12;
            try ld.getU32(names.KEY_PROJ_SCALE_FACTOR, &hparams.n_merge, false);
            try ld.getU32(names.KEY_PREPROC_MIN_TILES, &hparams.preproc_min_tiles, false);
            try ld.getU32(names.KEY_PREPROC_MAX_TILES, &hparams.preproc_max_tiles, false);
            impl.ggmlAssert(hparams.preproc_min_tiles <= hparams.preproc_max_tiles and hparams.preproc_max_tiles < std.math.maxInt(i32), "clip.cpp", 1471, "hparams.preproc_min_tiles <= hparams.preproc_max_tiles && hparams.preproc_max_tiles < INT32_MAX");
            try clip.Loader.setInternvlDhrResCandidates(model);
        },
        // PROJECTOR_TYPE_NEMOTRON_V2_VL
        .nemotron_v2_vl => {
            try ld.getU32(names.KEY_PROJ_SCALE_FACTOR, &hparams.n_merge, false);
        },
        // PROJECTOR_TYPE_PARAKEET
        .parakeet => {
            try ld.getU32(names.KEY_AUDIO_SUBSMPL_FACTOR, &hparams.subsampling_factor, true);
            impl.ggmlAssert(hparams.subsampling_factor == 8, "clip.cpp", 1481, "hparams.subsampling_factor == 8 && \"subsampling_factor must match the conv strides in clip_graph_parakeet::build()\"");
            try ld.getU32(names.KEY_A_CONV_KERNEL_SIZE, &hparams.audio_conv_kernel_size, true);
            impl.ggmlAssert(hparams.audio_conv_kernel_size > 0 and @rem(hparams.audio_conv_kernel_size, 2) == 1, "clip.cpp", 1484, "hparams.audio_conv_kernel_size > 0 && hparams.audio_conv_kernel_size % 2 == 1 && \"audio_conv_kernel_size must be a positive odd integer\"");
            hparams.audio_chunk_len = 0;
            hparams.audio_sample_rate = 16000;
            hparams.audio_n_fft = 512;
            hparams.audio_window_len = 400;
            hparams.audio_hop_len = 160;
        },
        // PROJECTOR_TYPE_IDEFICS3
        .idefics3 => {
            // use default llava-uhd preprocessing params
            hparams.image_resize_algo = .lanczos;
            try ld.getU32(names.KEY_PROJ_SCALE_FACTOR, &hparams.n_merge, false);
            try ld.getU32(names.KEY_PREPROC_IMAGE_SIZE, &hparams.image_longest_edge, false);
            hparams.setLimitImageTokensCustom();
        },
        // PROJECTOR_TYPE_LFM2
        .lfm2 => {
            hparams.image_resize_algo = .bilinear;
            hparams.image_resize_algo_rf = .bilinear;
            hparams.image_resize_algo_ov = .bilinear;
            try ld.getU32(names.KEY_PROJ_SCALE_FACTOR, &hparams.n_merge, false);
            // ref: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/blob/main/processor_config.json
            hparams.setLimitImageTokens(64, 256);
        },
        // PROJECTOR_TYPE_PHI4
        .phi4 => {
            hparams.n_merge = 1;
            hparams.image_resize_algo = .bilinear;
            try ld.getU32(names.KEY_IMAGE_MIN_PIXELS, &hparams.image_min_pixels, true);
            try ld.getU32(names.KEY_IMAGE_MAX_PIXELS, &hparams.image_max_pixels, true);
            hparams.setWarmupNTokens(16 * 16);
        },
        // PROJECTOR_TYPE_PIXTRAL
        .pixtral => {
            // ref: https://huggingface.co/mistral-community/pixtral-12b/blob/main/preprocessor_config.json
            hparams.n_merge = 1; // the original pixtral does not use patch merging
            hparams.image_resize_algo = .bicubic;
            hparams.rope_theta = 10000.0;
            try ld.getU32(names.KEY_SPATIAL_MERGE_SIZE, &hparams.n_merge, false);
            hparams.setLimitImageTokens(8, 1024);
            hparams.setWarmupNTokens(256); // avoid OOM on warmup
        },
        // PROJECTOR_TYPE_LIGHTONOCR
        .lightonocr => {
            hparams.n_merge = 1;
            hparams.image_resize_algo = .bicubic;
            hparams.rope_theta = 10000.0;
            try ld.getU32(names.KEY_SPATIAL_MERGE_SIZE, &hparams.n_merge, false);
            hparams.image_longest_edge = hparams.image_size;
            try ld.getU32(names.KEY_PREPROC_IMAGE_SIZE, &hparams.image_longest_edge, false);
            hparams.setLimitImageTokensCustom();
            hparams.setWarmupNTokens(256); // avoid OOM on warmup
        },
        // PROJECTOR_TYPE_DOTS_OCR
        .dots_ocr => {
            hparams.rope_theta = 10000.0;
            try ld.getU32(names.KEY_PROJ_SCALE_FACTOR, &hparams.n_merge, true);
            try ld.getU32(names.KEY_IMAGE_MIN_PIXELS, &hparams.image_min_pixels, true);
            try ld.getU32(names.KEY_IMAGE_MAX_PIXELS, &hparams.image_max_pixels, true);
            hparams.setWarmupNTokens(46 * 46); // avoid OOM on warmup
        },
        // PROJECTOR_TYPE_DOTS3NOTE_V
        .dots3note_v => {
            hparams.rope_theta = 10000.0;
            hparams.image_resize_algo = .bicubic;
            try ld.getU32(names.KEY_SPATIAL_MERGE_SIZE, &hparams.n_merge, true);
            try ld.getU32(names.KEY_IMAGE_MIN_PIXELS, &hparams.image_min_pixels, true);
            try ld.getU32(names.KEY_IMAGE_MAX_PIXELS, &hparams.image_max_pixels, true);
            try ld.getU32(names.KEY_VISION_N_EXPERT_USED, &hparams.n_expert_used, true);
            hparams.setWarmupNTokens(46 * 46); // avoid OOM on warmup
        },
        // PROJECTOR_TYPE_DOTS3NOTE_A
        .dots3note_a => {
            hparams.rope_theta = 10000.0;
            hparams.audio_chunk_len = 60; // in seconds
            hparams.audio_sample_rate = 16000;
            hparams.audio_n_fft = 400;
            hparams.audio_window_len = 400;
            hparams.audio_hop_len = 160;
        },
        // PROJECTOR_TYPE_KIMIVL
        .kimivl => {
            hparams.image_resize_algo = .bicubic;
            hparams.rope_theta = 10000.0;
            try ld.getU32(names.KEY_PROJ_SCALE_FACTOR, &hparams.n_merge, false);
            // TODO: check kimivl preprocessor for exact values
            hparams.setLimitImageTokens(8, 1024);
            hparams.setWarmupNTokens(256); // avoid OOM on warmup
        },
        // PROJECTOR_TYPE_KIMIK25
        .kimik25 => {
            hparams.image_resize_algo = .bicubic;
            hparams.rope_theta = 10000.0;
            try ld.getU32(names.KEY_PROJ_SCALE_FACTOR, &hparams.n_merge, false);

            var min_pixels: i32 = 0;
            var max_pixels: i32 = 0;
            try ld.getU32(names.KEY_IMAGE_MIN_PIXELS, &min_pixels, false);
            try ld.getU32(names.KEY_IMAGE_MAX_PIXELS, &max_pixels, false);
            if (min_pixels > 0 and max_pixels > 0) {
                hparams.image_min_pixels = min_pixels;
                hparams.image_max_pixels = max_pixels;
                hparams.warmup_image_size = @intFromFloat(@sqrt(@as(f64, @floatFromInt(max_pixels))));
            } else {
                hparams.setLimitImageTokens(2, 4096);
            }
        },
        // PROJECTOR_TYPE_DEEPSEEK4V
        .deepseek4v => {
            hparams.image_resize_algo = .bicubic;
            hparams.image_pad_color = .{ 127, 127, 127 };
            hparams.rope_theta = 10000.0;
            try ld.getU32(names.KEY_PROJ_SCALE_FACTOR, &hparams.n_merge, true);
            try ld.getU32(names.KEY_IMAGE_MIN_PIXELS, &hparams.image_min_pixels, true);
            hparams.dsv4_max_n_token = 384;
            hparams.dsv4_max_wh_ratio = 8;
            const patch_area = hparams.patch_size * hparams.patch_size * hparams.n_merge * hparams.n_merge;
            // handle min/max token counts from CLI
            if (hparams.custom_image_min_tokens > 0) {
                hparams.image_min_pixels = hparams.custom_image_min_tokens * patch_area;
            }
            if (hparams.custom_image_max_tokens > 0) {
                // the cap is on the whole token block, keep some room for the resize solver
                hparams.dsv4_max_n_token = @max(hparams.custom_image_max_tokens, 16);
            }
            hparams.image_max_pixels = hparams.dsv4_max_n_token * patch_area;
            // a small custom max token count also lowers the min-pixel upscale threshold
            hparams.image_min_pixels = @min(hparams.image_min_pixels, hparams.image_max_pixels);
            // avoid OOM on warmup
            const warmup_side: c_int = @intFromFloat(@sqrt(@as(f64, @floatFromInt(@min(256, hparams.dsv4_max_n_token)))));
            hparams.setWarmupNTokens(warmup_side * warmup_side);
        },
        // PROJECTOR_TYPE_GEMMA3
        .gemma3 => {
            // default value (used by all model sizes in gemma 3 family)
            // number of patches for each **side** is reduced by a factor of 4
            hparams.n_merge = 4;
            hparams.image_resize_algo = .bilinear;
            // test model (tinygemma3) has a different value, we optionally read it
            try ld.getU32(names.KEY_PROJ_SCALE_FACTOR, &hparams.n_merge, false);
        },
        // PROJECTOR_TYPE_GEMMA4V, GEMMA4UV
        .gemma4v, .gemma4uv => {
            hparams.rope_theta = 100.0;
            hparams.n_merge = 3; // pooling_kernel_size
            hparams.image_resize_algo = .bicubic;
            try ld.getU32(names.KEY_PROJ_SCALE_FACTOR, &hparams.n_merge, false);
            if (model.proj_type == .gemma4uv) {
                // for "unified" variant, we directly use a bigger patch size, because the "token merging" is done directly on conv layer
                hparams.patch_size = hparams.patch_size * hparams.n_merge;
                hparams.n_merge = 1;
            }
            hparams.setLimitImageTokens(70, 1120);
            hparams.setWarmupNTokens(256); // avoid OOM on warmup
        },
        // PROJECTOR_TYPE_GEMMA3NV
        .gemma3nv => {
            // Gemma3n uses MobileNetV5 which produces 256 tokens (16x16)
            hparams.n_merge = 1; // MobileNetV5 handles resizing internally
            hparams.image_resize_algo = .bilinear;
            try ld.getU32(names.KEY_PROJ_SCALE_FACTOR, &hparams.n_merge, false);
        },
        // PROJECTOR_TYPE_QWEN2VL, QWEN25VL, QWEN3VL
        .qwen2vl, .qwen25vl, .qwen3vl => {
            hparams.n_merge = 2; // default value for Qwen 2 and 2.5
            hparams.image_resize_algo = .bicubic;
            try ld.getU32(names.KEY_SPATIAL_MERGE_SIZE, &hparams.n_merge, false);
            try ld.getU32(names.KEY_WIN_ATTN_PATTERN, &hparams.n_wa_pattern, model.proj_type == .qwen25vl); // only 2.5 requires it
            // ref: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json
            hparams.setLimitImageTokens(8, 4096);
            hparams.setWarmupNTokens(46 * 46); // avoid OOM on warmup
            const warn_min_pixels = 1024 * hparams.n_merge * hparams.n_merge * hparams.patch_size * hparams.patch_size;
            if (hparams.image_min_pixels < warn_min_pixels) {
                impl.logWrn("%s: Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks\n", .{func});
                impl.logWrn("%s: if you encounter problems with accuracy, try adding --image-min-tokens 1024\n", .{func});
                impl.logWrn("%s: more info: https://github.com/ggml-org/llama.cpp/issues/16842\n\n", .{func});
            }
        },
        // PROJECTOR_TYPE_MINIMAX_M3
        .minimax_m3 => {
            hparams.n_merge = 2; // spatial_merge_size
            hparams.image_resize_algo = .bicubic;
            hparams.image_resize_pad = .none;
            try ld.getU32(names.KEY_SPATIAL_MERGE_SIZE, &hparams.n_merge, false);
            // n_merge is used as a divisor; reject 0 to avoid int div-by-zero
            impl.ggmlAssert(hparams.n_merge > 0, "clip.cpp", 1677, "hparams.n_merge > 0");
            hparams.rope_theta = 10000.0; // vision_config.rope_theta
            // MiniMax-M3: max_pixels 451584 (=672^2) -> 576 merged tokens (image_seq_length)
            hparams.setLimitImageTokens(8, 576);
            hparams.setWarmupNTokens(16 * 16);
        },
        // PROJECTOR_TYPE_MUSE_GLIMMER
        .muse_glimmer => {
            hparams.n_merge = 2; // pixel-shuffle downsample after the ViT
            hparams.image_resize_algo = .lanczos;
            hparams.rope_theta = 10000.0;
            hparams.muse_glimmer_patch_temporal = 2;
            hparams.muse_glimmer_sparse_factor = 4; // 3 sparse layers + 1 global, repeating
            try ld.getU32(names.KEY_SPATIAL_MERGE_SIZE, &hparams.n_merge, false);
            hparams.setLimitImageTokens(1, 4096);
            hparams.setWarmupNTokens(32 * 32);
        },
        // PROJECTOR_TYPE_MIMOVL
        .mimovl => {
            hparams.n_merge = 2; // spatial_merge_size
            hparams.image_resize_algo = .bicubic;
            try ld.getU32(names.KEY_SPATIAL_MERGE_SIZE, &hparams.n_merge, false);
            try ld.getU32(clip.fmtKey(&kb, names.KEY_N_HEAD_KV, "vision"), &hparams.n_head_kv, true);
            // 1D banded sliding-window radius (visual_token_window_size); required
            try ld.getU32(names.KEY_ATTN_WINDOW_SIZE, &hparams.attn_window_size, true);
            var pat: std.ArrayList(i32) = .empty;
            defer pat.deinit(gpa);
            try ld.getArrInt(names.KEY_WA_PATTERN_MODE, &pat, true);
            impl.ggmlAssert(@as(i32, @intCast(pat.items.len)) == hparams.n_layer, "clip.cpp", 1704, "(int) pat.size() == hparams.n_layer && \"mimovl wa_pattern_mode length must equal n_layer\"");
            hparams.wa_pattern_mode.clearRetainingCapacity();
            try hparams.wa_pattern_mode.appendSlice(gpa, pat.items);
            try ld.getU32(names.KEY_IMAGE_MIN_PIXELS, &hparams.image_min_pixels, true);
            try ld.getU32(names.KEY_IMAGE_MAX_PIXELS, &hparams.image_max_pixels, true);
            hparams.setWarmupNTokens(46 * 46); // avoid OOM on warmup
        },
        // PROJECTOR_TYPE_STEP3VL
        .step3vl => {
            hparams.n_merge = 4; // two stride-2 downsamplers after patching
            try ld.getU32(names.KEY_PROJ_SCALE_FACTOR, &hparams.n_merge, false);
            hparams.rope_theta = 10000.0;
            try ld.getU32(names.KEY_PREPROC_IMAGE_SIZE, &hparams.image_longest_edge, false);
            if (hparams.image_longest_edge == 0) hparams.image_longest_edge = 3024;
            // note: the step3vl preprocessor slices based on a fixed window grid, so it does not support custom min/max image tokens
            hparams.warmup_image_size = hparams.image_size;
        },
        // PROJECTOR_TYPE_YOUTUVL
        .youtuvl => {
            hparams.n_merge = 2;
            hparams.image_resize_algo = .bilinear;
            hparams.image_resize_pad = .none;
            try ld.getU32(names.KEY_SPATIAL_MERGE_SIZE, &hparams.n_merge, false);
            try ld.getU32(names.KEY_ATTN_WINDOW_SIZE, &hparams.attn_window_size, true);
            var wa_layer_indexes_vec: std.ArrayList(i32) = .empty;
            defer wa_layer_indexes_vec.deinit(gpa);
            try ld.getArrInt(names.KEY_WIN_ATTN_LAYER_INDEXES, &wa_layer_indexes_vec, true);
            for (wa_layer_indexes_vec.items) |layer| try insertWaLayer(hparams, layer);
            // support max_height * max_width = 8000 * 8000. 8000/16/2 = 250 image tokens
            hparams.setLimitImageTokens(1, 62500);
            hparams.setWarmupNTokens(16 * 16); // avoid OOM on warmup
        },
        // PROJECTOR_TYPE_YASA2
        .yasa2 => {
            hparams.ffn_op = .gelu_erf;
            info.log_ffn_op.* = "gelu_erf";
            hparams.image_resize_algo = .bicubic;
            // reka model performs better when the image is stretched to fit
            // fixed square size (no padding)
            hparams.image_resize_pad = .none;
        },
        // PROJECTOR_TYPE_GLM4V
        .glm4v => {
            hparams.rope_theta = 10000.0;
            hparams.n_merge = 2; // default value for GLM4-V
            hparams.image_resize_algo = .bicubic;
            try ld.getU32(names.KEY_SPATIAL_MERGE_SIZE, &hparams.n_merge, false);
            hparams.setLimitImageTokens(8, 4096);
            hparams.setWarmupNTokens(46 * 46); // avoid OOM on warmup
        },
        // PROJECTOR_TYPE_LLAMA4
        .llama4 => {
            hparams.rope_theta = 10000.0;
            hparams.image_resize_algo = .bilinear;
            try ld.getU32(names.KEY_PROJ_SCALE_FACTOR, &hparams.n_merge, false);
            try clip.Loader.setLlavaUhdResCandidates(model, 3);
        },
        // PROJECTOR_TYPE_ULTRAVOX, QWEN2A, QWEN3A, GLMA, VOXTRAL, MERALION, MUSIC_FLAMINGO
        .ultravox, .qwen2a, .qwen3a, .glma, .voxtral, .meralion, .music_flamingo => {
            const require_stack = model.proj_type == .ultravox or model.proj_type == .voxtral or
                model.proj_type == .meralion or model.proj_type == .glma;
            try ld.getU32(names.KEY_A_PROJ_STACK_FACTOR, &hparams.proj_stack_factor, require_stack);
            hparams.ffn_op = .gelu_erf;
            info.log_ffn_op.* = "gelu_erf"; // temporary solution for logging

            // audio preprocessing params
            hparams.audio_chunk_len = 30; // in seconds
            hparams.audio_sample_rate = 16000;
            hparams.audio_n_fft = 400;
            hparams.audio_window_len = 400;
            hparams.audio_hop_len = 160;
        },
        // PROJECTOR_TYPE_MIMO_AUDIO
        .mimo_audio => {
            try ld.getU32(names.KEY_A_RVQ_NUM_QUANTIZERS, &hparams.rvq_num_quantizers, false);
            try ld.getArrInt(names.KEY_A_RVQ_CODEBOOK_SIZE, &hparams.rvq_codebook_size, false);
            if (hparams.rvq_num_quantizers <= 0) {
                return clip.throwErr("%s: mimo_audio: missing %s\n", .{ func, @as([*:0]const u8, names.KEY_A_RVQ_NUM_QUANTIZERS) });
            }
            if (@as(i32, @intCast(hparams.rvq_codebook_size.items.len)) != hparams.rvq_num_quantizers) {
                return clip.throwErr("%s: mimo_audio: %s length (%zu) must equal %s (%d)\n", .{
                    func,                                               @as([*:0]const u8, names.KEY_A_RVQ_CODEBOOK_SIZE), hparams.rvq_codebook_size.items.len,
                    @as([*:0]const u8, names.KEY_A_RVQ_NUM_QUANTIZERS), hparams.rvq_num_quantizers,
                });
            }
            hparams.ffn_op = .gelu_erf; // PyTorch F.gelu default (approximate="none")
            hparams.rope_theta = 10000.0;

            // audio preprocessing params (mel spectrogram)
            hparams.audio_sample_rate = 24000;
            hparams.audio_n_fft = 960;
            hparams.audio_window_len = 960;
            hparams.audio_hop_len = 240;

            try ld.getU32(names.KEY_A_ATTN_WINDOW_SIZE, &hparams.attn_window_size, true);
            var wa_pattern: std.ArrayList(i32) = .empty;
            defer wa_pattern.deinit(gpa);
            try ld.getArrInt(names.KEY_A_WA_PATTERN_MODE, &wa_pattern, true);
            if (@as(i32, @intCast(wa_pattern.items.len)) != hparams.n_layer) {
                return clip.throwErr("%s: mimo_audio: %s length (%zu) must equal n_layer (%d)\n", .{ func, @as([*:0]const u8, names.KEY_A_WA_PATTERN_MODE), wa_pattern.items.len, hparams.n_layer });
            }
            hparams.wa_pattern_mode.clearRetainingCapacity();
            try hparams.wa_pattern_mode.appendSlice(gpa, wa_pattern.items);

            try ld.getU32(names.KEY_A_LOCAL_BLOCK_COUNT, &hparams.audio_local_n_layer, true);
            try ld.getU32(names.KEY_A_LOCAL_GROUP_SIZE, &hparams.audio_local_group_size, true);
            if (hparams.audio_local_group_size <= 0) {
                return clip.throwErr("%s: mimo_audio: %s must be > 0\n", .{ func, @as([*:0]const u8, names.KEY_A_LOCAL_GROUP_SIZE) });
            }
        },
        // PROJECTOR_TYPE_QWEN3TTS_SPKENC
        .qwen3tts_spkenc => {
            // ECAPA-TDNN speaker encoder, mel front-end uses the Slaney default (fmin=0, fmax=sr/2)
            hparams.audio_sample_rate = 24000;
            hparams.audio_n_fft = 1024;
            hparams.audio_window_len = 1024;
            hparams.audio_hop_len = 256;
        },
        // PROJECTOR_TYPE_QWEN3TTS_GEN
        .qwen3tts_gen => {
            // TODO: hardcoded for now, read from code_predictor_config instead
            hparams.rope_theta = 1000000.0;

            // code2wav params
            hparams.wav_tfm_n_layer = 8;
            hparams.wav_tfm_n_embd = 512;
            hparams.wav_tfm_n_ff = 1024;
            hparams.wav_tfm_n_head = 16;
            hparams.wav_tfm_n_head_kv = 16;
            hparams.wav_tfm_eps = 1e-5;
            hparams.wav_tfm_rope_theta = 10000.0;
            hparams.wav_upsample_n_block = 2;
            hparams.wav_dac_n_block = 4;
            hparams.wav_dac_n_res = 3;
            // matches the reference decoder's sliding_window (speech_tokenizer/config.json)
            hparams.wav_tfm_swa = 72;
        },
        // PROJECTOR_TYPE_POCKETTTS_SPKENC, POCKETTTS_GEN
        .pockettts_spkenc, .pockettts_gen => {
            // mimi front-end takes the raw waveform, no mel
            hparams.audio_sample_rate = 24000;
            // seanet ratios are [6,5,4] in the config, the encoder reverses them
            hparams.seanet_ratios.clearRetainingCapacity();
            try hparams.seanet_ratios.appendSlice(gpa, &.{ 4, 5, 6 });
            hparams.seanet_n_stage = @intCast(hparams.seanet_ratios.items.len);
            hparams.mimi_downsample = 16;
            // matches the reference transformer's "context"
            hparams.mimi_tfm_context = 250;
            hparams.rope_theta = 10000.0;
            // flow_lm defaults, see pocket_tts/default_parameters.py
            hparams.flow_n_step = 1;
            hparams.gen_eos_threshold = -4.0;
        },
        // PROJECTOR_TYPE_PADDLEOCR
        .paddleocr => {
            hparams.n_merge = 2;
            hparams.image_resize_algo = .bicubic;
            try ld.getU32(names.KEY_IMAGE_MIN_PIXELS, &hparams.image_min_pixels, true);
            try ld.getU32(names.KEY_IMAGE_MAX_PIXELS, &hparams.image_max_pixels, true);
            hparams.setWarmupNTokens(28 * 28); // avoid OOM on warmup
        },
        // PROJECTOR_TYPE_DEEPSEEKOCR, DEEPSEEKOCR2
        .deepseekocr, .deepseekocr2 => {
            hparams.patch_size = 16;
            hparams.image_size = 1024;
            hparams.warmup_image_size = 1024;
            hparams.image_resize_algo = .bicubic;
            hparams.image_pad_color = .{ 127, 127, 127 };

            try ld.getU32(names.KEY_SAM_N_BLOCK, &hparams.sam_n_layer, true);
            try ld.getU32(names.KEY_SAM_N_HEAD, &hparams.sam_n_head, true);
            try ld.getU32(names.KEY_SAM_N_EMBD, &hparams.sam_n_embd, true);
            try ld.getU32(names.KEY_ATTN_WINDOW_SIZE, &hparams.attn_window_size, true);
            hparams.preproc_min_tiles = 2;
            if (model.proj_type == .deepseekocr) {
                hparams.preproc_max_tiles = 9;
                hparams.preproc_tile_size = 640;
                // the CLIP/ViT body runs its layernorms at 1e-5 (the SAM stage uses 1e-6)
                hparams.eps = 1e-5;
            }
            if (model.proj_type == .deepseekocr2) {
                hparams.preproc_max_tiles = 6;
                hparams.preproc_tile_size = 768;
                // qwen2 encoder is GQA, requires KEY_N_HEAD_KV
                try ld.getU32(clip.fmtKey(&kb, names.KEY_N_HEAD_KV, "vision"), &hparams.n_head_kv, true);
            }
            // unlimited-ocr shares the v1 projector but tiles up to 32
            try ld.getU32(names.KEY_PREPROC_MIN_TILES, &hparams.preproc_min_tiles, false);
            try ld.getU32(names.KEY_PREPROC_MAX_TILES, &hparams.preproc_max_tiles, false);
            impl.ggmlAssert(hparams.preproc_min_tiles >= 0 and hparams.preproc_min_tiles <= hparams.preproc_max_tiles and hparams.preproc_max_tiles <= 256, "clip.cpp", 1907, "hparams.preproc_min_tiles >= 0 && hparams.preproc_min_tiles <= hparams.preproc_max_tiles && hparams.preproc_max_tiles <= 256");
        },
        // PROJECTOR_TYPE_HUNYUANVL
        .hunyuanvl => {
            hparams.n_merge = 2;
            hparams.image_resize_algo = .lanczos;
            hparams.image_resize_pad = .none;
            hparams.ffn_op = .gelu;
            hparams.setLimitImageTokens(256, 16384);
            try ld.getU32(names.KEY_SPATIAL_MERGE_SIZE, &hparams.n_merge, false);
            try ld.getU32(names.KEY_IMAGE_MIN_PIXELS, &hparams.image_min_pixels, false);
            try ld.getU32(names.KEY_IMAGE_MAX_PIXELS, &hparams.image_max_pixels, false);
            hparams.setWarmupNTokens(32 * 32);
        },
        // PROJECTOR_TYPE_LFM2A
        .lfm2a => {
            // audio preprocessing params
            hparams.audio_chunk_len = 1; // in seconds
            hparams.audio_sample_rate = 16000;
            hparams.audio_n_fft = 512;
            hparams.audio_window_len = 400;
            hparams.audio_hop_len = 160;
        },
        // PROJECTOR_TYPE_EXAONE4_5
        .exaone4_5 => {
            hparams.n_merge = 2;
            try ld.getU32(names.KEY_SPATIAL_MERGE_SIZE, &hparams.n_merge, false);
            try ld.getU32(names.KEY_WIN_ATTN_PATTERN, &hparams.n_wa_pattern, false);
            try ld.getU32(names.KEY_IMAGE_MIN_PIXELS, &hparams.image_min_pixels, true);
            try ld.getU32(names.KEY_IMAGE_MAX_PIXELS, &hparams.image_max_pixels, true);
            hparams.setWarmupNTokens(46 * 46);
            if (hparams.rope_theta <= 0.0) hparams.rope_theta = 10000.0;
            try ld.getU32(clip.fmtKey(&kb, names.KEY_N_HEAD_KV, "vision"), &hparams.n_head_kv, true);
        },
        // PROJECTOR_TYPE_GEMMA4A
        .gemma4a => {
            // Gemma4 feature_extraction_gemma4.py:
            // frame_length_ms=20 -> 320 samples, n_fft=512, hop=10ms -> 160
            hparams.audio_chunk_len = 0; // no fixed-length padding
            hparams.audio_sample_rate = 16000;
            hparams.audio_n_fft = 512;
            hparams.audio_window_len = 320; // 20ms frame (NOT 25ms/400)
            hparams.audio_hop_len = 160;
            // all gemma4a models use 1e-6; the original conversion wrote a wrong rms_norm_eps
            hparams.eps = 1e-6;
        },
        // PROJECTOR_TYPE_GEMMA4UA
        .gemma4ua => {
            // Encoder-free: raw 16 kHz waveform chunked into 640-sample frames.
            hparams.audio_chunk_len = 0;
            hparams.audio_sample_rate = 16000;
            hparams.eps = 1e-6;
            hparams.n_mel_bins = 640;
        },
        // PROJECTOR_TYPE_GRANITE_SPEECH
        .granite_speech => {
            hparams.audio_chunk_len = 0;
            hparams.audio_sample_rate = 16000;
            hparams.audio_n_fft = 512;
            hparams.audio_window_len = 400;
            hparams.audio_hop_len = 160;
            try ld.getU32(names.KEY_A_CHUNK_SIZE, &hparams.audio_chunk_size, true);
            // context_size is squared for the attn_dists/mask buffers; cap to prevent int32 overflow
            impl.ggmlAssert(hparams.audio_chunk_size > 0 and hparams.audio_chunk_size <= 8192, "clip.cpp", 1976, "hparams.audio_chunk_size > 0 && hparams.audio_chunk_size <= 8192");
            try ld.getU32(names.KEY_A_CONV_KERNEL_SIZE, &hparams.audio_conv_kernel_size, true);
            try ld.getU32(names.KEY_A_MAX_POS_EMB, &hparams.audio_max_pos_emb, true);
            try ld.getU32(names.KEY_A_PROJ_WINDOW_SIZE, &hparams.audio_proj_window_size, true);
            try ld.getU32(names.KEY_A_PROJ_DOWNSAMPLE_RATE, &hparams.audio_proj_downsample_rate, true);
            try ld.getU32(names.KEY_A_PROJ_HEAD_COUNT, &hparams.audio_proj_head_count, true);
            // NOTE: feature layers loaded above in common path
        },
        // PROJECTOR_TYPE_JANUS_PRO
        .janus_pro => {
            hparams.image_pad_color = .{ 127, 127, 127 };
            hparams.image_resize_algo = .bicubic;
        },
        // PROJECTOR_TYPE_GRANITE4_VISION
        .granite4_vision => {
            // SigLIP tower.
            hparams.image_resize_algo = .bicubic;
            hparams.image_resize_pad = .ceil;

            // NOTE: feature_layers loaded in common path as optional
            try ld.getArrInt(names.KEY_PROJ_SPATIAL_OFFSETS, &hparams.proj_spatial_offsets, true);
            if (hparams.feature_layers.items.len != hparams.proj_spatial_offsets.items.len) {
                return granite4FeatureLayerMismatch(hparams.feature_layers.items.len, hparams.proj_spatial_offsets.items.len);
            }

            try ld.getU32(names.KEY_PROJ_SAMPLE_QUERY_SIDE, &hparams.downsample_query_side, true);
            try ld.getU32(names.KEY_PROJ_SAMPLE_WINDOW_SIDE, &hparams.downsample_window_side, true);
            hparams.warmup_image_size = hparams.image_size;
        },
        else => return clip.throwErr("%s: unknown vision projector type %s\n", .{ func, info.proj_type }),
    }
}

/// The granite4_vision mismatch throw, line 1970 of clip.cpp.
///
/// Upstream formats `"%s: feature_layers.size() %d != proj_spatial_offsets.size() %d"`
/// with only the two `size_t` sizes -- `__func__` is missing -- so `%s`
/// reads `feature_layers.size()` as a `char *` and the last `%d` reads past
/// the arguments: undefined behaviour. The defined part is reproduced: a
/// size of 0 is a null pointer, which the C library prints as `(null)`,
/// followed by the second size, and the missing argument is taken as 0;
/// any other size is a wild pointer, on which upstream's `snprintf` faults,
/// so the port raises SIGSEGV there.
fn granite4FeatureLayerMismatch(n_feature_layers: usize, n_offsets: usize) clip.Error {
    if (n_feature_layers != 0) {
        _ = std.c.raise(.SEGV);
        std.process.abort();
    }
    return clip.throwErr("%s: feature_layers.size() %d != proj_spatial_offsets.size() %d", .{
        @as(?[*:0]const u8, null), @as(c_int, @truncate(@as(i64, @bitCast(n_offsets)))), @as(c_int, 0),
    });
}

/// `hparams.wa_layer_indexes.insert(layer)`, recording the insertion order.
fn insertWaLayer(hparams: *model_mod.Hparams, layer: i32) clip.Error!void {
    if (wa_owner != hparams) {
        wa_insertions.clearRetainingCapacity();
        wa_owner = hparams;
    }
    try wa_insertions.append(gpa, layer);
    try hparams.wa_layer_indexes.put(gpa, layer, {});
}

/// `hparams.wa_layer_indexes` in the order a `std::unordered_set<int32_t>`
/// iterates it, for `load_hparams`'s log line. Borrowed until the next call.
pub fn waLayerIndexesInSetOrder(hparams: *const model_mod.Hparams) clip.Error![]const i32 {
    wa_ordered.clearRetainingCapacity();
    if (wa_owner != hparams) return wa_ordered.items;
    var set = try libcxx.U32Set.init(gpa);
    defer set.deinit();
    for (wa_insertions.items) |v| _ = try set.insert(@bitCast(v));
    var it = set.iterator();
    while (it.next()) |v| try wa_ordered.append(gpa, @bitCast(v));
    return wa_ordered.items;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the set order of wa_layer_indexes is libc++'s, not insertion order" {
    var hp: model_mod.Hparams = .{};
    defer hp.deinit();
    for ([_]i32{ 31, 7, 15, 23 }) |v| try insertWaLayer(&hp, v);
    const got = try waLayerIndexesInSetOrder(&hp);
    try std.testing.expectEqual(@as(usize, 4), got.len);
    var sorted: [4]i32 = undefined;
    @memcpy(&sorted, got);
    std.mem.sort(i32, &sorted, {}, std.sort.asc(i32));
    try std.testing.expectEqualSlices(i32, &.{ 7, 15, 23, 31 }, &sorted);
}
