//! The clip model's hyperparameters and weights.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/clip-model.h` at v0.9.0 (`cd959bae9`), field
//! for field. The tensor-pointer fields were extracted from the header
//! mechanically, in the header's order, so a struct here reads against the
//! C++ line by line; the rest is written by hand. Each declaration names
//! what it ports and the line it began at.

const std = @import("std");
const impl = @import("impl.zig");
const names = @import("names.zig");
const c = impl.c;
const gpa = impl.gpa;

/// Ports `ffn_op_type` (clip-model.h:15 @cd959bae9).
pub const FfnOpType = enum(c_int) { gelu, gelu_erf, silu, gelu_quick, relu_sqr };

/// Ports `norm_type` (clip-model.h:23 @cd959bae9).
pub const NormType = enum(c_int) { normal, rms };

/// Ports `patch_merge_type` (clip-model.h:28 @cd959bae9).
pub const PatchMergeType = enum(c_int) { flat, spatial_unpad };

/// Ports `resize_algo` (clip-model.h:35 @cd959bae9): the low bits
/// select the algorithm, Pillow-compatible (matching `PIL.Image.resize`);
/// `flag_torch` matches torchvision's antialiased resize of `uint8` images on
/// the CPU (int16 weights), bilinear and bicubic only. Non-exhaustive, because
/// the C ORs the flag in: `base` and `isTorch` read the two halves apart.
pub const ResizeAlgo = enum(c_int) {
    bilinear = 0,
    bicubic = 1,
    lanczos = 2,
    _,

    pub const flag_torch: c_int = 1 << 8;

    /// The algorithm without the torchvision flag.
    /// Return: bilinear, bicubic or lanczos.
    pub fn base(self: ResizeAlgo) ResizeAlgo {
        return @fromBackingInt(@backingInt(self) & ~flag_torch);
    }

    /// Return: whether `RESIZE_ALGO_FLAG_TORCH` is set.
    pub fn isTorch(self: ResizeAlgo) bool {
        return (@backingInt(self) & flag_torch) != 0;
    }
};

/// Ports `pad_style` (clip-model.h:47 @cd959bae9): how `img_tool::resize`
/// pads.
pub const PadStyle = enum(c_int) {
    /// no padding; direct resize to target dimensions
    none,
    /// aspect-preserving pad (default)
    ceil,
    /// aspect-preserving pad with nearest-integer rounding (Pillow byte-parity)
    nearest,
};

/// `std::unordered_set<int32_t>`, as far as the C++ uses it: membership.
pub const IntSet = std.AutoHashMapUnmanaged(i32, void);

/// Ports `clip_hparams` (clip-model.h:53 @cd959bae9).
pub const Hparams = struct {
    image_size: i32 = 0,
    patch_size: i32 = 0,
    n_embd: i32 = 0,
    n_ff: i32 = 0,
    projection_dim: i32 = 0,
    n_head: i32 = 0,
    n_head_kv: i32 = 0,
    // 0 = derive from n_embd; set when qkv width != n_embd
    n_embd_head: i32 = 0,
    n_layer: i32 = 0,
    n_merge: i32 = 1, // number of patch merges **per-side**

    // for preprocessor
    image_longest_edge: i32 = 0,
    image_min_pixels: i32 = -1,
    image_max_pixels: i32 = -1,
    image_resize_algo: ResizeAlgo = .bicubic,
    image_resize_pad: PadStyle = .ceil, // padding style when resizing
    image_round_half_even: bool = false, // smart_resize rounds .5 to even, like Python round() in the HF processor
    image_floor_before_align: bool = false,
    image_pad_color: [3]u8 = .{ 0, 0, 0 },

    // (preprocessor) for llava-uhd style models
    image_res_candidates: std.ArrayList(impl.ImageSize) = .empty,
    preproc_min_tiles: i32 = 0,
    preproc_max_tiles: i32 = 0,
    max_slice_nums: i32 = 9, // llava-uhd slice cap; per-model, carried in the GGUF
    preproc_tile_size: i32 = 0, // local tile size (deepseek-ocr, lfm2)
    preproc_max_pixels_tolerance: f32 = 1.0,
    image_resize_algo_rf: ResizeAlgo = .bicubic,
    image_resize_algo_ov: ResizeAlgo = .bicubic,
    image_pad_rf: PadStyle = .ceil, // padding style for the refined image (e.g. llava-1.6)
    image_pad_ov: PadStyle = .none, // padding style for the overview image (e.g. llava-1.6)
    image_pad_color_rf: [3]u8 = .{ 0, 0, 0 }, // padding color for refined image
    image_pad_color_ov: [3]u8 = .{ 0, 0, 0 }, // padding color for overview image

    // uninitialized in the C++; every loader path sets both
    image_mean: [3]f32 = .{ 0, 0, 0 },
    image_std: [3]f32 = .{ 0, 0, 0 },

    // for models using dynamic image size, we need to have a smaller image size to warmup
    // otherwise, user will get OOM every time they load the model
    warmup_image_size: i32 = 0,
    warmup_audio_size: i32 = 3000,

    ffn_op: FfnOpType = .gelu,

    mm_patch_merge_type: PatchMergeType = .flat,

    eps: f32 = 1e-6,
    rope_theta: f32 = 0.0,

    swiglu_clamp: f32 = 0.0,
    n_expert_used: i32 = 0,
    feature_layers: std.ArrayList(i32) = .empty,
    attn_window_size: i32 = 0,
    n_wa_pattern: i32 = 0,
    wa_layer_indexes: IntSet = .empty, // explicit layer indexes that use full attention (for irregular patterns like YoutuVL)
    wa_pattern_mode: std.ArrayList(i32) = .empty, // mimovl: per-layer window-attention mode

    // deepseek4v: resize solver caps the LLM token count of the aligner grid
    dsv4_max_n_token: i32 = 0,
    dsv4_max_wh_ratio: i32 = 0,

    // deepseek-ocr (sam)
    sam_n_layer: i32 = 0,
    sam_n_head: i32 = 0,
    sam_n_embd: i32 = 0,

    // Granite4 Vision
    proj_spatial_offsets: std.ArrayList(i32) = .empty,
    // uninitialized in the C++
    downsample_query_side: i32 = 0,
    downsample_window_side: i32 = 0,

    // Muse Glimmer vision (per-block sparse-window pattern, learned pos-emb, patch-temporal)
    muse_glimmer_patch_temporal: i32 = 0,
    muse_glimmer_sparse_factor: i32 = 0,

    // audio
    n_mel_bins: i32 = 0, // whisper preprocessor
    proj_stack_factor: i32 = 0, // ultravox
    subsampling_factor: i32 = 0, // parakeet

    audio_chunk_size: i32 = 0,
    audio_conv_kernel_size: i32 = 0,
    audio_max_pos_emb: i32 = 0,
    audio_proj_window_size: i32 = 0,
    audio_proj_downsample_rate: i32 = 0,
    audio_proj_head_count: i32 = 0,

    // audio pyramid (muse-spark): widths, folded samples and blocks per stage
    audio_hidden_sizes: std.ArrayList(i32) = .empty,
    audio_patch_sizes: std.ArrayList(i32) = .empty,
    audio_depths: std.ArrayList(i32) = .empty,

    // audio-to-mel preprocessor params
    audio_chunk_len: i32 = -1, // in seconds
    audio_sample_rate: i32 = -1,
    audio_n_fft: i32 = -1,
    audio_window_len: i32 = -1,
    audio_hop_len: i32 = -1,

    // parakeet
    mel_filters: std.ArrayList(f32) = .empty,
    window: std.ArrayList(f32) = .empty,

    // nemotron transcribe
    audio_rope_dims: i32 = 0,
    spk_n_embd: i32 = 0,
    spk_n_layer: i32 = 0,
    spk_n_head: i32 = 0,

    // mimo-audio-tokenizer: residual vector quantizer
    rvq_num_quantizers: i32 = 0,
    rvq_codebook_size: std.ArrayList(i32) = .empty, // per-quantizer bin count (ragged)

    // threshold for the "out_eos_score" graph output
    gen_eos_threshold: f32 = 0.0,

    // name of the weight variant, some pipelines tune themselves on it
    gen_model_variant: []const u8 = "",

    // sampling defaults of the generated audio: greedy, no top-k, no repetition penalty
    gen_temp: f32 = 0.0,
    gen_top_k: i32 = 0,
    gen_penalty_repeat: f32 = 1.0,
    gen_penalty_last_n: i32 = 0,

    // pocket-tts
    seanet_n_stage: i32 = 0,
    seanet_ratios: std.ArrayList(i32) = .empty, // encoder order (reversed compared to the config)
    mimi_downsample: i32 = 0, // encoder frame rate / model frame rate
    mimi_tfm_context: i32 = 0, // attention window of the mimi transformers, in frames
    flow_n_step: i32 = 1, // lsd_decode steps

    // qwen3tts code2wav
    wav_tfm_n_layer: i32 = 0,
    wav_tfm_n_embd: i32 = 0,
    wav_tfm_n_ff: i32 = 0,
    wav_tfm_n_head: i32 = 0,
    wav_tfm_n_head_kv: i32 = 0,
    wav_tfm_eps: f32 = 1e-5,
    wav_tfm_rope_theta: f32 = 10000.0,
    wav_upsample_n_block: i32 = 0,
    wav_dac_n_block: i32 = 0,
    wav_dac_n_res: i32 = 0,
    wav_tfm_swa: i32 = 0, // pre_transformer's KV cache size, in frames
    wav_frame_stack: i32 = 0, // lfm3a codec: decoder positions per frame

    // mimo-v2.5: LLM-side connector (input_local_transformer)
    audio_local_n_layer: i32 = 0,
    audio_local_group_size: i32 = 0,

    // legacy
    has_llava_projector: bool = false,
    minicpmv_version: c_int = 0,
    minicpmv_query_num: i32 = 0, // MiniCPM-V query number
    insert_layer_id: i32 = 0, // MiniCPM-V 4.6 ViT merger insertion layer

    // custom value provided by user, can be undefined if not set
    custom_image_min_tokens: i32 = -1,
    custom_image_max_tokens: i32 = -1,

    /// `pockettts_max_spk_seconds` (clip-model.h:192 @cd959bae9), a member of `clip_hparams`.
    pub const pockettts_max_spk_seconds: i32 = 30;

    pub fn deinit(self: *Hparams) void {
        self.image_res_candidates.deinit(gpa);
        self.feature_layers.deinit(gpa);
        self.wa_layer_indexes.deinit(gpa);
        self.wa_pattern_mode.deinit(gpa);
        self.proj_spatial_offsets.deinit(gpa);
        self.mel_filters.deinit(gpa);
        self.window.deinit(gpa);
        self.rvq_codebook_size.deinit(gpa);
        self.seanet_ratios.deinit(gpa);
        self.audio_hidden_sizes.deinit(gpa);
        self.audio_patch_sizes.deinit(gpa);
        self.audio_depths.deinit(gpa);
    }

    /// Ports `audio_samples_per_token` (clip-model.h:149 @cd959bae9), a member of
    /// `clip_hparams`: the audio pyramid's fold, every stage multiplied.
    pub fn audioSamplesPerToken(self: *const Hparams) i32 {
        var n: i64 = 1;
        for (self.audio_patch_sizes.items) |p| {
            n *= p;
            impl.ggmlAssert(n >= 1 and n <= (1 << 20), "clip-model.h", 153, "n >= 1 && n <= (1 << 20)");
        }
        return @intCast(n);
    }

    /// Ports `set_image_resize_algo` (clip-model.h:227 @cd959bae9), a member of `clip_hparams`.
    pub fn setImageResizeAlgo(self: *Hparams, algo: ResizeAlgo, torch: bool) void {
        impl.ggmlAssert(!(torch and algo == .lanczos), "clip-model.h", 228, "!(torch && algo == RESIZE_ALGO_LANCZOS) && \"torchvision has no lanczos\"");
        self.image_resize_algo = if (torch) @fromBackingInt(@backingInt(algo) | ResizeAlgo.flag_torch) else algo;
    }

    /// Ports `set_limit_image_tokens` (clip-model.h:232 @cd959bae9), a member of `clip_hparams`.
    pub fn setLimitImageTokens(self: *Hparams, n_tokens_min: c_int, n_tokens_max: c_int) void {
        const patch_area = self.patch_size * self.patch_size * self.n_merge * self.n_merge;
        self.image_min_pixels = (if (self.custom_image_min_tokens > 0) self.custom_image_min_tokens else n_tokens_min) * patch_area;
        self.image_max_pixels = (if (self.custom_image_max_tokens > 0) self.custom_image_max_tokens else n_tokens_max) * patch_area;
        self.warmup_image_size = @intFromFloat(@sqrt(@as(f64, @floatFromInt(self.image_max_pixels))));
    }

    /// Ports `set_limit_image_tokens` (clip-model.h:240 @cd959bae9), a member of `clip_hparams`,
    /// the overload the longest-edge preprocessor uses.
    pub fn setLimitImageTokensCustom(self: *Hparams) void {
        const patch_area = self.patch_size * self.patch_size * self.n_merge * self.n_merge;
        if (self.custom_image_min_tokens > 0) self.image_min_pixels = self.custom_image_min_tokens * patch_area;
        if (self.custom_image_max_tokens > 0) self.image_max_pixels = self.custom_image_max_tokens * patch_area;
    }

    /// Ports `set_warmup_n_tokens` (clip-model.h:250 @cd959bae9), a member of `clip_hparams`:
    /// since v0.6.0 never more tokens than `custom_image_max_tokens` allows.
    pub fn setWarmupNTokens(self: *Hparams, n_tokens: c_int) void {
        var n_tok_per_side: c_int = @intFromFloat(@sqrt(@as(f64, @floatFromInt(n_tokens))));
        impl.ggmlAssert(n_tok_per_side * n_tok_per_side == n_tokens, "clip-model.h", 252, "n_tok_per_side * n_tok_per_side == n_tokens && \"n_tokens must be n*n\"");
        // do not warmup with more tokens than the max allowed
        if (self.custom_image_max_tokens > 0 and n_tokens > self.custom_image_max_tokens) {
            n_tok_per_side = @max(1, @as(c_int, @intFromFloat(@sqrt(@as(f64, @floatFromInt(self.custom_image_max_tokens))))));
        }
        self.warmup_image_size = n_tok_per_side * self.patch_size * self.n_merge;
    }

    /// Ports `is_global_attn` (clip-model.h:263 @cd959bae9), a member of `clip_hparams`, over
    /// `global_attn_indices`, the SAM ViT's {2, 5, 8, 11}.
    pub fn isGlobalAttn(self: *const Hparams, layer: i32) bool {
        _ = self;
        return layer == 2 or layer == 5 or layer == 8 or layer == 11;
    }

    /// Ports `is_feature_layer` (clip-model.h:275 @cd959bae9), a member of `clip_hparams`.
    pub fn isFeatureLayer(self: *const Hparams, layer: i32) bool {
        return std.mem.indexOfScalar(i32, self.feature_layers.items, layer) != null;
    }
};

/// Ports `clip_layer` (clip-model.h:280 @cd959bae9).
pub const Layer = struct {
    // layernorm 1 (or layer input norm, or pre-attention norm)
    ln_1_w: ?*c.ggml_tensor = null,
    ln_1_b: ?*c.ggml_tensor = null,
    // attention
    k_w: ?*c.ggml_tensor = null,
    k_b: ?*c.ggml_tensor = null,
    q_w: ?*c.ggml_tensor = null,
    q_b: ?*c.ggml_tensor = null,
    v_w: ?*c.ggml_tensor = null,
    v_b: ?*c.ggml_tensor = null,
    qkv_w: ?*c.ggml_tensor = null,
    qkv_b: ?*c.ggml_tensor = null,
    o_w: ?*c.ggml_tensor = null,
    o_b: ?*c.ggml_tensor = null,
    attn_sinks: ?*c.ggml_tensor = null,
    k_norm: ?*c.ggml_tensor = null,
    q_norm: ?*c.ggml_tensor = null,
    k_norm_b: ?*c.ggml_tensor = null,
    q_norm_b: ?*c.ggml_tensor = null,
    // lfm3: per-head sigmoid gate on the attention output
    attn_gate_w: ?*c.ggml_tensor = null,
    attn_post_norm_w: ?*c.ggml_tensor = null,
    ff_up_w: ?*c.ggml_tensor = null,
    ff_up_b: ?*c.ggml_tensor = null,
    ff_gate_w: ?*c.ggml_tensor = null,
    ff_gate_b: ?*c.ggml_tensor = null,
    ff_down_w: ?*c.ggml_tensor = null,
    ff_down_b: ?*c.ggml_tensor = null,
    // MoE FFN (dots3note vision pyramid blocks)
    ff_gate_inp_w: ?*c.ggml_tensor = null,
    ff_gate_exps_w: ?*c.ggml_tensor = null,
    ff_up_exps_w: ?*c.ggml_tensor = null,
    ff_down_exps_w: ?*c.ggml_tensor = null,
    ff_exp_probs_b: ?*c.ggml_tensor = null,
    // layernorm 2 (or pre-FFN norm)
    ln_2_w: ?*c.ggml_tensor = null,
    ln_2_b: ?*c.ggml_tensor = null,
    ff_post_norm_w: ?*c.ggml_tensor = null,
    // depthwise causal conv block (muse-spark audio), conv_norm_w is declared below
    conv_w: ?*c.ggml_tensor = null,
    conv_b: ?*c.ggml_tensor = null,
    // layer scale (no bias)
    ls_1_w: ?*c.ggml_tensor = null,
    ls_2_w: ?*c.ggml_tensor = null,
    ls_out_w: ?*c.ggml_tensor = null, // gemma4
    // qwen3vl deepstack merger
    deepstack_norm_w: ?*c.ggml_tensor = null,
    deepstack_norm_b: ?*c.ggml_tensor = null,
    deepstack_fc1_w: ?*c.ggml_tensor = null,
    deepstack_fc1_b: ?*c.ggml_tensor = null,
    deepstack_fc2_w: ?*c.ggml_tensor = null,
    deepstack_fc2_b: ?*c.ggml_tensor = null,
    // lfm3vl conv layers
    conv2d_in_proj_w: ?*c.ggml_tensor = null,
    conv2d_spatial_w: ?*c.ggml_tensor = null, // may be null
    conv2d_out_proj_w: ?*c.ggml_tensor = null,
    // sam rel_pos
    rel_pos_w: ?*c.ggml_tensor = null,
    rel_pos_h: ?*c.ggml_tensor = null,
    // lfm2
    ff_norm_w: ?*c.ggml_tensor = null,
    ff_norm_b: ?*c.ggml_tensor = null,
    ff_norm_1_w: ?*c.ggml_tensor = null,
    ff_norm_1_b: ?*c.ggml_tensor = null,
    ff_up_1_w: ?*c.ggml_tensor = null,
    ff_up_1_b: ?*c.ggml_tensor = null,
    ff_down_1_w: ?*c.ggml_tensor = null,
    ff_down_1_b: ?*c.ggml_tensor = null,
    pos_bias_u: ?*c.ggml_tensor = null,
    pos_bias_v: ?*c.ggml_tensor = null,
    norm_conv_w: ?*c.ggml_tensor = null,
    norm_conv_b: ?*c.ggml_tensor = null,
    linear_pos_w: ?*c.ggml_tensor = null,
    conv_norm_w: ?*c.ggml_tensor = null,
    conv_norm_b: ?*c.ggml_tensor = null,
    conv_norm_mean: ?*c.ggml_tensor = null, // parakeet
    conv_norm_var: ?*c.ggml_tensor = null, // parakeet
    conv_dw_w: ?*c.ggml_tensor = null,
    conv_dw_b: ?*c.ggml_tensor = null,
    conv_pw1_w: ?*c.ggml_tensor = null,
    conv_pw1_b: ?*c.ggml_tensor = null,
    conv_pw2_w: ?*c.ggml_tensor = null,
    conv_pw2_b: ?*c.ggml_tensor = null,
    // gemma4 audio conformer per-layer
    attn_pre_norm_w: ?*c.ggml_tensor = null,
    attn_k_rel_w: ?*c.ggml_tensor = null,
    per_dim_scale_w: ?*c.ggml_tensor = null,
    per_dim_k_scale_w: ?*c.ggml_tensor = null,
    ff_post_norm_1_w: ?*c.ggml_tensor = null,
    // granite_speech conformer per-layer
    attn_rel_pos_emb: ?*c.ggml_tensor = null,
    // granite_speech qformer cross-attention
    cross_attn_q_w: ?*c.ggml_tensor = null,
    cross_attn_q_b: ?*c.ggml_tensor = null,
    cross_attn_k_w: ?*c.ggml_tensor = null,
    cross_attn_k_b: ?*c.ggml_tensor = null,
    cross_attn_v_w: ?*c.ggml_tensor = null,
    cross_attn_v_b: ?*c.ggml_tensor = null,
    cross_attn_o_w: ?*c.ggml_tensor = null,
    cross_attn_o_b: ?*c.ggml_tensor = null,
    cross_attn_norm_w: ?*c.ggml_tensor = null,
    cross_attn_norm_b: ?*c.ggml_tensor = null,
    // qwen3tts speaker encoder: SE-Res2Net block, tdnn1/tdnn2 reuse conv_pw1_w/b and conv_pw2_w/b above
    se_conv1_w: ?*c.ggml_tensor = null,
    se_conv1_b: ?*c.ggml_tensor = null,
    se_conv2_w: ?*c.ggml_tensor = null,
    se_conv2_b: ?*c.ggml_tensor = null,

    res2_conv_w: std.ArrayList(?*c.ggml_tensor) = .empty, // Res2Net hierarchical branches
    res2_conv_b: std.ArrayList(?*c.ggml_tensor) = .empty,

    pub fn hasDeepstack(self: *const Layer) bool {
        return self.deepstack_fc1_w != null;
    }

    pub fn deinit(self: *Layer) void {
        self.res2_conv_w.deinit(gpa);
        self.res2_conv_b.deinit(gpa);
    }
};

/// Ports `mobilenetv5_block` (clip-model.h:417 @cd959bae9): the expanded
/// MobileNetV5 block of the Gemma3n vision encoder.
pub const MobilenetV5Block = struct {
    // Stage 0 (Edge Residual)
    s0_conv_exp_w: ?*c.ggml_tensor = null,
    s0_bn1_w: ?*c.ggml_tensor = null,
    s0_conv_pwl_w: ?*c.ggml_tensor = null,
    s0_bn2_w: ?*c.ggml_tensor = null,
    // Stage 1+ (Universal Inverted Residual)
    dw_start_w: ?*c.ggml_tensor = null,
    dw_start_bn_w: ?*c.ggml_tensor = null,
    pw_exp_w: ?*c.ggml_tensor = null,
    pw_exp_bn_w: ?*c.ggml_tensor = null,
    dw_mid_w: ?*c.ggml_tensor = null,
    dw_mid_bn_w: ?*c.ggml_tensor = null,
    pw_proj_w: ?*c.ggml_tensor = null,
    pw_proj_bn_w: ?*c.ggml_tensor = null,
    layer_scale_w: ?*c.ggml_tensor = null,
    // Attention (MQA) components
    attn_q_w: ?*c.ggml_tensor = null,
    attn_k_w: ?*c.ggml_tensor = null,
    attn_v_w: ?*c.ggml_tensor = null,
    attn_o_w: ?*c.ggml_tensor = null,
    // Optional downsampling/norm in attention
    attn_k_dw_w: ?*c.ggml_tensor = null,
    attn_k_norm_w: ?*c.ggml_tensor = null,
    attn_v_dw_w: ?*c.ggml_tensor = null,
    attn_v_norm_w: ?*c.ggml_tensor = null,
    // Block norm (often present in attention blocks)
    attn_norm_w: ?*c.ggml_tensor = null,
};

/// Ports `yasa2_block` (clip-model.h:455 @cd959bae9).
pub const Yasa2Block = struct {
    dw_w: ?*c.ggml_tensor = null,
    dw_b: ?*c.ggml_tensor = null,
    ln_w: ?*c.ggml_tensor = null,
    ln_b: ?*c.ggml_tensor = null,
    pw1_w: ?*c.ggml_tensor = null,
    pw1_b: ?*c.ggml_tensor = null,
    grn_w: ?*c.ggml_tensor = null,
    grn_b: ?*c.ggml_tensor = null,
    pw2_w: ?*c.ggml_tensor = null,
    pw2_b: ?*c.ggml_tensor = null,
};

/// Ports `yasa2_stage` (clip-model.h:468 @cd959bae9).
pub const Yasa2Stage = struct {
    down_ln_w: ?*c.ggml_tensor = null,
    down_ln_b: ?*c.ggml_tensor = null,
    down_conv_w: ?*c.ggml_tensor = null,
    down_conv_b: ?*c.ggml_tensor = null,
    blocks: std.ArrayList(Yasa2Block) = .empty,
};

/// Ports `qf_block` (clip-model.h:478 @cd959bae9): a QFormer projector block
/// (Granite Speech, Granite4 Vision).
pub const QfBlock = struct {
    qf_proj_query: ?*c.ggml_tensor = null,
    qf_proj_norm_w: ?*c.ggml_tensor = null,
    qf_proj_norm_b: ?*c.ggml_tensor = null,
    qf_proj_linear_w: ?*c.ggml_tensor = null,
    qf_proj_linear_b: ?*c.ggml_tensor = null,
    qf_proj_post_norm_w: ?*c.ggml_tensor = null,
    qf_proj_post_norm_b: ?*c.ggml_tensor = null,
    qf_proj_img_pos: ?*c.ggml_tensor = null, // Vision only
    qf_proj_layers: std.ArrayList(Layer) = .empty,
};

/// Ports `clip_seanet` (clip-model.h:493 @cd959bae9): the pocket-tts SEANet
/// stack, used in both directions.
pub const Seanet = struct {
    /// Ports `stage` (clip-model.h:495 @cd959bae9), a member of `clip_seanet`.
    pub const Stage = struct {
        res_conv1_w: ?*c.ggml_tensor = null,
        res_conv1_b: ?*c.ggml_tensor = null,
        res_conv2_w: ?*c.ggml_tensor = null,
        res_conv2_b: ?*c.ggml_tensor = null,
        scale_conv_w: ?*c.ggml_tensor = null, // strided conv (encoder) or convtr (decoder)
        scale_conv_b: ?*c.ggml_tensor = null,
    };

    conv_in_w: ?*c.ggml_tensor = null,
    conv_in_b: ?*c.ggml_tensor = null,
    conv_out_w: ?*c.ggml_tensor = null,
    conv_out_b: ?*c.ggml_tensor = null,
    stages: std.ArrayList(Stage) = .empty,
};

/// Ports `clip_flow_net` (clip-model.h:512 @cd959bae9): the pocket-tts
/// flow-matching decoder (SimpleMLPAdaLN).
pub const FlowNet = struct {
    /// Ports `block` (clip-model.h:514 @cd959bae9), a member of `clip_flow_net`.
    pub const Block = struct {
        norm_w: ?*c.ggml_tensor = null,
        norm_b: ?*c.ggml_tensor = null,
        up_w: ?*c.ggml_tensor = null,
        up_b: ?*c.ggml_tensor = null,
        down_w: ?*c.ggml_tensor = null,
        down_b: ?*c.ggml_tensor = null,
        ada_w: ?*c.ggml_tensor = null, // -> shift, scale, gate
        ada_b: ?*c.ggml_tensor = null,
    };

    /// Ports `time_embd` (clip-model.h:526 @cd959bae9), a member of `clip_flow_net`.
    pub const TimeEmbd = struct {
        freqs: ?*c.ggml_tensor = null,
        up_w: ?*c.ggml_tensor = null,
        up_b: ?*c.ggml_tensor = null,
        down_w: ?*c.ggml_tensor = null,
        down_b: ?*c.ggml_tensor = null,
        norm: ?*c.ggml_tensor = null, // RMSNorm alpha
    };

    input_proj_w: ?*c.ggml_tensor = null,
    input_proj_b: ?*c.ggml_tensor = null,
    cond_embd_w: ?*c.ggml_tensor = null,
    cond_embd_b: ?*c.ggml_tensor = null,
    final_ada_w: ?*c.ggml_tensor = null, // -> shift, scale
    final_ada_b: ?*c.ggml_tensor = null,
    final_proj_w: ?*c.ggml_tensor = null,
    final_proj_b: ?*c.ggml_tensor = null,
    time: std.ArrayList(TimeEmbd) = .empty,
    blocks: std.ArrayList(Block) = .empty,
};

/// Ports `clip_code2wav` (clip-model.h:548 @cd959bae9): qwen3tts RVQ codes
/// to raw PCM.
pub const Code2Wav = struct {
    /// Ports `upsample_block` (clip-model.h:550 @cd959bae9), a member of `clip_code2wav`.
    pub const UpsampleBlock = struct {
        conv_w: ?*c.ggml_tensor = null, // causal ConvTranspose1d, 2x
        conv_b: ?*c.ggml_tensor = null,
        dwconv_w: ?*c.ggml_tensor = null, // depthwise causal conv, k=7
        dwconv_b: ?*c.ggml_tensor = null,
        norm_w: ?*c.ggml_tensor = null, // LayerNorm
        norm_b: ?*c.ggml_tensor = null,
        pw1_w: ?*c.ggml_tensor = null, // pointwise expand
        pw1_b: ?*c.ggml_tensor = null,
        pw2_w: ?*c.ggml_tensor = null, // pointwise project
        pw2_b: ?*c.ggml_tensor = null,
        gamma: ?*c.ggml_tensor = null, // layer scale
    };

    /// Ports `dac_res` (clip-model.h:565 @cd959bae9), a member of `clip_code2wav`.
    pub const DacRes = struct {
        act1_alpha: ?*c.ggml_tensor = null,
        act1_beta: ?*c.ggml_tensor = null,
        conv1_w: ?*c.ggml_tensor = null,
        conv1_b: ?*c.ggml_tensor = null,
        act2_alpha: ?*c.ggml_tensor = null,
        act2_beta: ?*c.ggml_tensor = null,
        conv2_w: ?*c.ggml_tensor = null,
        conv2_b: ?*c.ggml_tensor = null,
    };

    /// Ports `dac_block` (clip-model.h:577 @cd959bae9), a member of `clip_code2wav`.
    pub const DacBlock = struct {
        snake_alpha: ?*c.ggml_tensor = null,
        snake_beta: ?*c.ggml_tensor = null,
        conv_w: ?*c.ggml_tensor = null, // causal ConvTranspose1d
        conv_b: ?*c.ggml_tensor = null,
        res: std.ArrayList(DacRes) = .empty,
    };

    // quantizer: RVQ codebook decode
    quant_first_in_w: ?*c.ggml_tensor = null, // semantic RVQ, in_proj (1x1 conv, loaded as 2D)
    quant_first_out_w: ?*c.ggml_tensor = null,
    quant_first_out_b: ?*c.ggml_tensor = null,
    quant_first_cb_w: ?*c.ggml_tensor = null, // codebook (1 layer)
    quant_rest_in_w: ?*c.ggml_tensor = null, // acoustic RVQ
    quant_rest_out_w: ?*c.ggml_tensor = null,
    quant_rest_cb_w: ?*c.ggml_tensor = null, // codebooks, merged 3D [15, vocab, dim]

    pre_conv_w: ?*c.ggml_tensor = null,
    pre_conv_b: ?*c.ggml_tensor = null,

    tfm_in_proj_w: ?*c.ggml_tensor = null,
    tfm_in_proj_b: ?*c.ggml_tensor = null,
    tfm_out_proj_w: ?*c.ggml_tensor = null,
    tfm_out_proj_b: ?*c.ggml_tensor = null,
    tfm_output_norm_w: ?*c.ggml_tensor = null,
    tfm_layers: std.ArrayList(Layer) = .empty, // reuses the generic block fields

    upsample: std.ArrayList(UpsampleBlock) = .empty,

    dac_entry_w: ?*c.ggml_tensor = null,
    dac_entry_b: ?*c.ggml_tensor = null,
    dac: std.ArrayList(DacBlock) = .empty,
    dac_post_snake_alpha: ?*c.ggml_tensor = null,
    dac_post_snake_beta: ?*c.ggml_tensor = null,
    dac_post_conv_w: ?*c.ggml_tensor = null,
    dac_post_conv_b: ?*c.ggml_tensor = null,
};

/// Ports `clip_chatterbox` (clip-model.h:615 @cd959bae9): chatterbox's
/// flow encoder, CFM estimator, HiFT vocoder, S3 tokenizer and CAMPPlus
/// speaker encoder.
pub const Chatterbox = struct {
    /// Ports `enc_layer` (clip-model.h:617 @cd959bae9), a member of `clip_chatterbox`:
    /// an espnet rel-pos conformer layer of the flow encoder.
    pub const EncLayer = struct {
        norm_mha_w: ?*c.ggml_tensor = null,
        norm_mha_b: ?*c.ggml_tensor = null,
        attn_q_w: ?*c.ggml_tensor = null,
        attn_q_b: ?*c.ggml_tensor = null,
        attn_k_w: ?*c.ggml_tensor = null,
        attn_k_b: ?*c.ggml_tensor = null,
        attn_v_w: ?*c.ggml_tensor = null,
        attn_v_b: ?*c.ggml_tensor = null,
        attn_pos_w: ?*c.ggml_tensor = null, // linear_pos, no bias
        attn_pos_bias_u: ?*c.ggml_tensor = null,
        attn_pos_bias_v: ?*c.ggml_tensor = null,
        attn_out_w: ?*c.ggml_tensor = null,
        attn_out_b: ?*c.ggml_tensor = null,
        norm_ff_w: ?*c.ggml_tensor = null,
        norm_ff_b: ?*c.ggml_tensor = null,
        ffn_1_w: ?*c.ggml_tensor = null,
        ffn_1_b: ?*c.ggml_tensor = null,
        ffn_2_w: ?*c.ggml_tensor = null,
        ffn_2_b: ?*c.ggml_tensor = null,
    };

    /// Ports `causal_block` (clip-model.h:640 @cd959bae9), a member of `clip_chatterbox`:
    /// conv k3 left padded + layer norm.
    pub const CausalBlock = struct {
        conv_w: ?*c.ggml_tensor = null,
        conv_b: ?*c.ggml_tensor = null,
        norm_w: ?*c.ggml_tensor = null,
        norm_b: ?*c.ggml_tensor = null,
    };

    /// Ports `resnet` (clip-model.h:648 @cd959bae9), a member of `clip_chatterbox`:
    /// a time conditioned resnet of the estimator.
    pub const Resnet = struct {
        block1: CausalBlock = .{},
        block2: CausalBlock = .{},
        mlp_w: ?*c.ggml_tensor = null, // time projection (mlp.1)
        mlp_b: ?*c.ggml_tensor = null,
        res_conv_w: ?*c.ggml_tensor = null,
        res_conv_b: ?*c.ggml_tensor = null,
    };

    /// Ports `tfm_block` (clip-model.h:658 @cd959bae9), a member of `clip_chatterbox`:
    /// a diffusers style transformer block of the estimator.
    pub const TfmBlock = struct {
        norm1_w: ?*c.ggml_tensor = null,
        norm1_b: ?*c.ggml_tensor = null,
        attn_q_w: ?*c.ggml_tensor = null, // no bias on qkv
        attn_k_w: ?*c.ggml_tensor = null,
        attn_v_w: ?*c.ggml_tensor = null,
        attn_out_w: ?*c.ggml_tensor = null, // to_out.0
        attn_out_b: ?*c.ggml_tensor = null,
        norm3_w: ?*c.ggml_tensor = null,
        norm3_b: ?*c.ggml_tensor = null,
        ff_in_w: ?*c.ggml_tensor = null, // ff.net.0.proj
        ff_in_b: ?*c.ggml_tensor = null,
        ff_out_w: ?*c.ggml_tensor = null, // ff.net.2
        ff_out_b: ?*c.ggml_tensor = null,
    };

    /// Ports `est_stage` (clip-model.h:675 @cd959bae9), a member of `clip_chatterbox`:
    /// resnet, transformer stack, boundary causal conv (null on mid stages).
    pub const EstStage = struct {
        res: Resnet = .{},
        tfm: std.ArrayList(TfmBlock) = .empty,
        conv_w: ?*c.ggml_tensor = null,
        conv_b: ?*c.ggml_tensor = null,
    };

    /// Ports `hift_res_unit` (clip-model.h:683 @cd959bae9), a member of `clip_chatterbox`:
    /// a snake resblock unit of the hift vocoder.
    pub const HiftResUnit = struct {
        act1_alpha: ?*c.ggml_tensor = null,
        act2_alpha: ?*c.ggml_tensor = null,
        conv1_w: ?*c.ggml_tensor = null,
        conv1_b: ?*c.ggml_tensor = null,
        conv2_w: ?*c.ggml_tensor = null,
        conv2_b: ?*c.ggml_tensor = null,
    };

    /// Ports `hift_res` (clip-model.h:691 @cd959bae9), a member of `clip_chatterbox`.
    pub const HiftRes = struct {
        units: std.ArrayList(HiftResUnit) = .empty, // dilations 1/3/5 on conv1
    };

    /// Ports `hift_up` (clip-model.h:696 @cd959bae9), a member of `clip_chatterbox`:
    /// conv transpose, source injection, 3 resblocks.
    pub const HiftUp = struct {
        up_w: ?*c.ggml_tensor = null,
        up_b: ?*c.ggml_tensor = null,
        source_down_w: ?*c.ggml_tensor = null,
        source_down_b: ?*c.ggml_tensor = null,
        source_res: HiftRes = .{},
        res_0: HiftRes = .{},
        res_1: HiftRes = .{},
        res_2: HiftRes = .{},
    };

    /// Ports `f0_conv` (clip-model.h:708 @cd959bae9), a member of `clip_chatterbox`.
    pub const F0Conv = struct {
        w: ?*c.ggml_tensor = null,
        b: ?*c.ggml_tensor = null,
    };

    /// Ports `s3tok_block` (clip-model.h:714 @cd959bae9), a member of `clip_chatterbox`:
    /// neox rope on q/k, fsmn memory on v.
    pub const S3tokBlock = struct {
        attn_ln_w: ?*c.ggml_tensor = null,
        attn_ln_b: ?*c.ggml_tensor = null,
        attn_q_w: ?*c.ggml_tensor = null,
        attn_q_b: ?*c.ggml_tensor = null,
        attn_k_w: ?*c.ggml_tensor = null, // no bias
        attn_v_w: ?*c.ggml_tensor = null,
        attn_v_b: ?*c.ggml_tensor = null,
        fsmn_w: ?*c.ggml_tensor = null,
        attn_out_w: ?*c.ggml_tensor = null,
        attn_out_b: ?*c.ggml_tensor = null,
        mlp_ln_w: ?*c.ggml_tensor = null,
        mlp_ln_b: ?*c.ggml_tensor = null,
        mlp_in_w: ?*c.ggml_tensor = null,
        mlp_in_b: ?*c.ggml_tensor = null,
        mlp_out_w: ?*c.ggml_tensor = null,
        mlp_out_b: ?*c.ggml_tensor = null,
    };

    /// Ports `bn` (clip-model.h:734 @cd959bae9), a member of `clip_chatterbox`: a folded
    /// batchnorm (w/b stay null on the affine free variant).
    pub const Bn = struct {
        mean: ?*c.ggml_tensor = null,
        @"var": ?*c.ggml_tensor = null,
        w: ?*c.ggml_tensor = null,
        b: ?*c.ggml_tensor = null,
    };

    /// Ports `spk_res2d` (clip-model.h:742 @cd959bae9), a member of `clip_chatterbox`: the
    /// shortcut is null on stride 1 blocks.
    pub const SpkRes2d = struct {
        conv1_w: ?*c.ggml_tensor = null,
        bn1: Bn = .{},
        conv2_w: ?*c.ggml_tensor = null,
        bn2: Bn = .{},
        shortcut_w: ?*c.ggml_tensor = null,
        shortcut_bn: Bn = .{},
    };

    /// Ports `spk_cam_layer` (clip-model.h:752 @cd959bae9), a member of `clip_chatterbox`.
    pub const SpkCamLayer = struct {
        nl1_bn: Bn = .{},
        linear1_w: ?*c.ggml_tensor = null,
        nl2_bn: Bn = .{},
        local_w: ?*c.ggml_tensor = null, // cam_layer.linear_local
        ctx1_w: ?*c.ggml_tensor = null, // cam_layer.linear1
        ctx1_b: ?*c.ggml_tensor = null,
        ctx2_w: ?*c.ggml_tensor = null, // cam_layer.linear2
        ctx2_b: ?*c.ggml_tensor = null,
    };

    /// Ports `spk_cam_block` (clip-model.h:762 @cd959bae9), a member of `clip_chatterbox`.
    pub const SpkCamBlock = struct {
        layers: std.ArrayList(SpkCamLayer) = .empty,
        transit_bn: Bn = .{},
        transit_w: ?*c.ggml_tensor = null,
    };

    // flow encoder
    input_embedding_w: ?*c.ggml_tensor = null, // flow.input_embedding
    embed_linear_w: ?*c.ggml_tensor = null, // fenc.embed.out.0
    embed_linear_b: ?*c.ggml_tensor = null,
    embed_norm_w: ?*c.ggml_tensor = null, // fenc.embed.out.1
    embed_norm_b: ?*c.ggml_tensor = null,
    pre_conv1_w: ?*c.ggml_tensor = null, // pre_lookahead_layer
    pre_conv1_b: ?*c.ggml_tensor = null,
    pre_conv2_w: ?*c.ggml_tensor = null,
    pre_conv2_b: ?*c.ggml_tensor = null,
    enc: std.ArrayList(EncLayer) = .empty,
    up_conv_w: ?*c.ggml_tensor = null, // fenc.up_layer.conv
    up_conv_b: ?*c.ggml_tensor = null,
    up_embed_linear_w: ?*c.ggml_tensor = null, // fenc.up_embed.out.0
    up_embed_linear_b: ?*c.ggml_tensor = null,
    up_embed_norm_w: ?*c.ggml_tensor = null, // fenc.up_embed.out.1
    up_embed_norm_b: ?*c.ggml_tensor = null,
    up_enc: std.ArrayList(EncLayer) = .empty,
    after_norm_w: ?*c.ggml_tensor = null,
    after_norm_b: ?*c.ggml_tensor = null,
    encoder_proj_w: ?*c.ggml_tensor = null, // flow.encoder_proj
    encoder_proj_b: ?*c.ggml_tensor = null,

    // cfm estimator
    time_mlp_1_w: ?*c.ggml_tensor = null,
    time_mlp_1_b: ?*c.ggml_tensor = null,
    time_mlp_2_w: ?*c.ggml_tensor = null,
    time_mlp_2_b: ?*c.ggml_tensor = null,
    time_embed_mixer_w: ?*c.ggml_tensor = null, // meanflow variant only
    est_down: EstStage = .{},
    est_mid: std.ArrayList(EstStage) = .empty,
    est_up: EstStage = .{},
    est_final_block: CausalBlock = .{},
    est_final_proj_w: ?*c.ggml_tensor = null,
    est_final_proj_b: ?*c.ggml_tensor = null,

    // hift vocoder
    hift_pre_w: ?*c.ggml_tensor = null,
    hift_pre_b: ?*c.ggml_tensor = null,
    hift_ups: std.ArrayList(HiftUp) = .empty,
    hift_post_w: ?*c.ggml_tensor = null,
    hift_post_b: ?*c.ggml_tensor = null,
    f0_condnet: std.ArrayList(F0Conv) = .empty,
    f0_classifier_w: ?*c.ggml_tensor = null,
    f0_classifier_b: ?*c.ggml_tensor = null,

    // s3 tokenizer
    s3tok_conv1_w: ?*c.ggml_tensor = null,
    s3tok_conv1_b: ?*c.ggml_tensor = null,
    s3tok_conv2_w: ?*c.ggml_tensor = null,
    s3tok_conv2_b: ?*c.ggml_tensor = null,
    s3tok_blocks: std.ArrayList(S3tokBlock) = .empty,
    s3tok_down_w: ?*c.ggml_tensor = null, // quantizer._codebook.project_down
    s3tok_down_b: ?*c.ggml_tensor = null,

    // CAMPPlus speaker encoder
    spk_conv1_w: ?*c.ggml_tensor = null,
    spk_bn1: Bn = .{},
    spk_layer1_0: SpkRes2d = .{},
    spk_layer1_1: SpkRes2d = .{},
    spk_layer2_0: SpkRes2d = .{},
    spk_layer2_1: SpkRes2d = .{},
    spk_conv2_w: ?*c.ggml_tensor = null,
    spk_bn2: Bn = .{},
    spk_tdnn_w: ?*c.ggml_tensor = null,
    spk_tdnn_bn: Bn = .{},
    spk_block1: SpkCamBlock = .{},
    spk_block2: SpkCamBlock = .{},
    spk_block3: SpkCamBlock = .{},
    spk_out_bn: Bn = .{},
    spk_dense_w: ?*c.ggml_tensor = null,
    spk_dense_bn: Bn = .{},
    spk_affine_w: ?*c.ggml_tensor = null,
    spk_affine_b: ?*c.ggml_tensor = null,
};

/// Ports `clip_vocos` (clip-model.h:844 @cd959bae9): soprano's Vocos decoder.
pub const Vocos = struct {
    input_w: ?*c.ggml_tensor = null,
    input_b: ?*c.ggml_tensor = null,
    norm_w: ?*c.ggml_tensor = null,
    norm_b: ?*c.ggml_tensor = null,
    output_norm_w: ?*c.ggml_tensor = null,
    output_norm_b: ?*c.ggml_tensor = null,
    output_w: ?*c.ggml_tensor = null,
    output_b: ?*c.ggml_tensor = null,
    blocks: std.ArrayList(Code2Wav.UpsampleBlock) = .empty,
};

/// Ports `clamp_info` (clip-model.h:1208 @cd959bae9), a member of `clip_model`: the clamp
/// bounds of a Gemma4ClippableLinear.
pub const ClampInfo = struct {
    inp_max: f32,
    inp_min: f32,
    out_max: f32,
    out_min: f32,
};

/// Ports `audio_patch` (clip-model.h:884 @cd959bae9), a member of `clip_model`:
/// one patch embedder per audio pyramid stage (muse-spark).
pub const AudioPatch = struct {
    proj_w: ?*c.ggml_tensor = null,
    proj_b: ?*c.ggml_tensor = null,
    norm_w: ?*c.ggml_tensor = null,
};

/// Ports `clip_model` (clip-model.h:856 @cd959bae9).
pub const Model = struct {
    modality: impl.Modality = .vision,
    proj_type: names.ProjectorType = .mlp,
    hparams: Hparams = .{},

    layers: std.ArrayList(Layer) = .empty,

    // one patch embedder per audio pyramid stage (muse-spark)
    a_patches: std.ArrayList(AudioPatch) = .empty,

    n_deepstack_layers: i32 = 0, // used by Qwen3-VL, calculated from clip_layer

    mobilenet_blocks: std.ArrayList(MobilenetV5Block) = .empty,
    mobilenet_stage_ends: std.ArrayList(c_int) = .empty,
    yasa_stages: std.ArrayList(Yasa2Stage) = .empty,
    mm_a_local_layers: std.ArrayList(Layer) = .empty,

    // qwen3tts code2wav: RVQ codes -> raw PCM
    c2w: Code2Wav = .{},
    // pocket-tts: SEANet stack, shared by the encoder (speaker path) and the decoder (gen path)
    seanet: Seanet = .{},
    // pocket-tts: flow-matching decoder, backbone hidden state -> next latent
    flow: FlowNet = .{},
    gen_tfm_layers: std.ArrayList(Layer) = .empty, // mimi decoder_transformer

    // chatterbox flow encoder, cfm estimator, hift vocoder, s3 tokenizer and CAMPPlus speaker encoder
    cbx: Chatterbox = .{},

    // chatterbox tensors read on the host, see clip_cbx_read_tensor(); keys owned
    cbx_tensors: std.StringHashMapUnmanaged(?*c.ggml_tensor) = .empty,
    // predefined voices, no rows if the voice is named in the prompt (lfm3a)
    gen_voices: impl.GenVoiceMap = .{},
    // soprano: Vocos decoder, hidden states -> STFT coefficients
    vocos: Vocos = .{},

    n_sam_layers: i32 = 12, // used by deepseek-ocr sam encoder
    sam_layers: std.ArrayList(Layer) = .empty,

    // nemotron transcribe: speaker expert, Sortformer head, fusion
    spk_layers: std.ArrayList(Layer) = .empty,
    spk_inp_proj_w: ?*c.ggml_tensor = null,
    spk_pre_ln_w: ?*c.ggml_tensor = null,
    spk_pre_ln_b: ?*c.ggml_tensor = null,
    spk_post_ln_w: ?*c.ggml_tensor = null,
    spk_post_ln_b: ?*c.ggml_tensor = null,
    spk_head_proj_w: ?*c.ggml_tensor = null,
    spk_head_proj_b: ?*c.ggml_tensor = null,
    spk_head_up_w: ?*c.ggml_tensor = null,
    spk_head_up_b: ?*c.ggml_tensor = null,
    spk_head_fc1_w: ?*c.ggml_tensor = null,
    spk_head_fc1_b: ?*c.ggml_tensor = null,
    spk_head_fc2_w: ?*c.ggml_tensor = null,
    spk_head_fc2_b: ?*c.ggml_tensor = null,
    fusion_asr_norm_w: ?*c.ggml_tensor = null,
    fusion_asr_norm_b: ?*c.ggml_tensor = null,
    fusion_spk_norm_w: ?*c.ggml_tensor = null,
    fusion_spk_norm_b: ?*c.ggml_tensor = null,
    fusion_spk_kernel: ?*c.ggml_tensor = null,
    spk_sil_emb: std.ArrayList(f32) = .empty,

    // lfm2 audio
    pre_encode_conv_X_w: [7]?*c.ggml_tensor = @splat(null),
    pre_encode_conv_X_b: [7]?*c.ggml_tensor = @splat(null),

    // gemma4 Gemma4ClippableLinear, by tensor name
    clamp_info_map: std.StringHashMapUnmanaged(ClampInfo) = .empty,

    // gemma4 audio conformer
    sscp_conv_w: [2]?*c.ggml_tensor = @splat(null),
    sscp_conv_b: [2]?*c.ggml_tensor = @splat(null),
    sscp_norm_w: [2]?*c.ggml_tensor = @splat(null),

    // qformer projector(s)
    qf_proj_blocks: std.ArrayList(QfBlock) = .empty,

    // embeddings
    class_embedding: ?*c.ggml_tensor = null,
    patch_embeddings_0: ?*c.ggml_tensor = null,
    patch_embeddings_1: ?*c.ggml_tensor = null, // additional patch kernel; interpretation is model-specific (Qwen2VL, Nemotron Omni)
    patch_bias: ?*c.ggml_tensor = null,
    position_embeddings: ?*c.ggml_tensor = null,
    norm_embd_w: ?*c.ggml_tensor = null,
    norm_embd_b: ?*c.ggml_tensor = null,
    // "indexed" patch embedding norms
    patch_norm_1_w: ?*c.ggml_tensor = null,
    patch_norm_1_b: ?*c.ggml_tensor = null,
    patch_norm_2_w: ?*c.ggml_tensor = null,
    patch_norm_2_b: ?*c.ggml_tensor = null,
    patch_norm_3_w: ?*c.ggml_tensor = null,
    patch_norm_3_b: ?*c.ggml_tensor = null,
    pre_ln_w: ?*c.ggml_tensor = null,
    pre_ln_b: ?*c.ggml_tensor = null,
    post_ln_w: ?*c.ggml_tensor = null,
    post_ln_b: ?*c.ggml_tensor = null,
    mm_fc_w: ?*c.ggml_tensor = null,
    mm_fc_b: ?*c.ggml_tensor = null,
    mm_ffn_up_w: ?*c.ggml_tensor = null,
    mm_ffn_up_b: ?*c.ggml_tensor = null,
    mm_ffn_gate_w: ?*c.ggml_tensor = null,
    mm_ffn_gate_b: ?*c.ggml_tensor = null,
    mm_ffn_down_w: ?*c.ggml_tensor = null,
    mm_ffn_down_b: ?*c.ggml_tensor = null,
    mm_post_norm_w: ?*c.ggml_tensor = null,
    mm_post_norm_b: ?*c.ggml_tensor = null,
    // LLaVA projection
    mm_input_norm_w: ?*c.ggml_tensor = null,
    mm_input_norm_b: ?*c.ggml_tensor = null,
    mm_0_w: ?*c.ggml_tensor = null,
    mm_0_b: ?*c.ggml_tensor = null,
    mm_2_w: ?*c.ggml_tensor = null,
    mm_2_b: ?*c.ggml_tensor = null,
    mm_merger_fc1_w: ?*c.ggml_tensor = null, // minimax-m3
    mm_merger_fc1_b: ?*c.ggml_tensor = null,
    mm_merger_fc2_w: ?*c.ggml_tensor = null,
    mm_merger_fc2_b: ?*c.ggml_tensor = null,
    image_newline: ?*c.ggml_tensor = null,
    view_seperator: ?*c.ggml_tensor = null,
    tok_embd_sep: ?*c.ggml_tensor = null, // [n_embd_text, n_sep] rows of the text model tok_embd (MiniCPM-V 4.7)
    // Yi type models with mlp+normalization projection
    mm_1_w: ?*c.ggml_tensor = null, // Yi type models have 0, 1, 3, 4
    mm_1_b: ?*c.ggml_tensor = null,
    mm_3_w: ?*c.ggml_tensor = null,
    mm_3_b: ?*c.ggml_tensor = null,
    mm_4_w: ?*c.ggml_tensor = null,
    mm_4_b: ?*c.ggml_tensor = null,
    mm_5_w: ?*c.ggml_tensor = null,
    mm_5_b: ?*c.ggml_tensor = null,
    mm_6_w: ?*c.ggml_tensor = null,
    mm_6_b: ?*c.ggml_tensor = null,
    // GLMV-Edge projection
    mm_model_adapter_conv_w: ?*c.ggml_tensor = null,
    mm_model_adapter_conv_b: ?*c.ggml_tensor = null,
    // MobileVLM projection
    mm_model_mlp_1_w: ?*c.ggml_tensor = null,
    mm_model_mlp_1_b: ?*c.ggml_tensor = null,
    mm_model_mlp_3_w: ?*c.ggml_tensor = null,
    mm_model_mlp_3_b: ?*c.ggml_tensor = null,
    mm_model_block_1_block_0_0_w: ?*c.ggml_tensor = null,
    mm_model_block_1_block_0_1_w: ?*c.ggml_tensor = null,
    mm_model_block_1_block_0_1_b: ?*c.ggml_tensor = null,
    mm_model_block_1_block_1_fc1_w: ?*c.ggml_tensor = null,
    mm_model_block_1_block_1_fc1_b: ?*c.ggml_tensor = null,
    mm_model_block_1_block_1_fc2_w: ?*c.ggml_tensor = null,
    mm_model_block_1_block_1_fc2_b: ?*c.ggml_tensor = null,
    mm_model_block_1_block_2_0_w: ?*c.ggml_tensor = null,
    mm_model_block_1_block_2_1_w: ?*c.ggml_tensor = null,
    mm_model_block_1_block_2_1_b: ?*c.ggml_tensor = null,
    mm_model_block_2_block_0_0_w: ?*c.ggml_tensor = null,
    mm_model_block_2_block_0_1_w: ?*c.ggml_tensor = null,
    mm_model_block_2_block_0_1_b: ?*c.ggml_tensor = null,
    mm_model_block_2_block_1_fc1_w: ?*c.ggml_tensor = null,
    mm_model_block_2_block_1_fc1_b: ?*c.ggml_tensor = null,
    mm_model_block_2_block_1_fc2_w: ?*c.ggml_tensor = null,
    mm_model_block_2_block_1_fc2_b: ?*c.ggml_tensor = null,
    mm_model_block_2_block_2_0_w: ?*c.ggml_tensor = null,
    mm_model_block_2_block_2_1_w: ?*c.ggml_tensor = null,
    mm_model_block_2_block_2_1_b: ?*c.ggml_tensor = null,
    // MobileVLM_V2 projection
    mm_model_mlp_0_w: ?*c.ggml_tensor = null,
    mm_model_mlp_0_b: ?*c.ggml_tensor = null,
    mm_model_mlp_2_w: ?*c.ggml_tensor = null,
    mm_model_mlp_2_b: ?*c.ggml_tensor = null,
    mm_model_peg_0_w: ?*c.ggml_tensor = null,
    mm_model_peg_0_b: ?*c.ggml_tensor = null,
    // MINICPMV projection
    mm_model_pos_embed_k: ?*c.ggml_tensor = null,
    mm_model_query: ?*c.ggml_tensor = null,
    mm_model_proj: ?*c.ggml_tensor = null,
    mm_model_proj_b: ?*c.ggml_tensor = null,
    mm_model_kv_proj: ?*c.ggml_tensor = null,
    mm_model_attn_q_w: ?*c.ggml_tensor = null,
    mm_model_attn_q_b: ?*c.ggml_tensor = null,
    mm_model_attn_k_w: ?*c.ggml_tensor = null,
    mm_model_attn_k_b: ?*c.ggml_tensor = null,
    mm_model_attn_v_w: ?*c.ggml_tensor = null,
    mm_model_attn_v_b: ?*c.ggml_tensor = null,
    mm_model_attn_o_w: ?*c.ggml_tensor = null,
    mm_model_attn_o_b: ?*c.ggml_tensor = null,
    mm_model_ln_q_w: ?*c.ggml_tensor = null,
    mm_model_ln_q_b: ?*c.ggml_tensor = null,
    mm_model_ln_kv_w: ?*c.ggml_tensor = null,
    mm_model_ln_kv_b: ?*c.ggml_tensor = null,
    mm_model_ln_post_w: ?*c.ggml_tensor = null,
    mm_model_ln_post_b: ?*c.ggml_tensor = null,
    // MiniCPM-V 4.6 ViT merger (window self-attention + ViT MLP downsample)
    vit_merger_ln1_w: ?*c.ggml_tensor = null,
    vit_merger_ln1_b: ?*c.ggml_tensor = null,
    vit_merger_attn_q_w: ?*c.ggml_tensor = null,
    vit_merger_attn_q_b: ?*c.ggml_tensor = null,
    vit_merger_attn_k_w: ?*c.ggml_tensor = null,
    vit_merger_attn_k_b: ?*c.ggml_tensor = null,
    vit_merger_attn_v_w: ?*c.ggml_tensor = null,
    vit_merger_attn_v_b: ?*c.ggml_tensor = null,
    vit_merger_attn_o_w: ?*c.ggml_tensor = null,
    vit_merger_attn_o_b: ?*c.ggml_tensor = null,
    vit_merger_ds_ln_w: ?*c.ggml_tensor = null,
    vit_merger_ds_ln_b: ?*c.ggml_tensor = null,
    vit_merger_ds_up_w: ?*c.ggml_tensor = null,
    vit_merger_ds_up_b: ?*c.ggml_tensor = null,
    vit_merger_ds_down_w: ?*c.ggml_tensor = null,
    vit_merger_ds_down_b: ?*c.ggml_tensor = null,
    // gemma3
    mm_input_proj_w: ?*c.ggml_tensor = null,
    mm_soft_emb_norm_w: ?*c.ggml_tensor = null,
    // mobilenetv5 for gemma3n
    mobilenet_stem_conv_w: ?*c.ggml_tensor = null,
    mobilenet_stem_conv_b: ?*c.ggml_tensor = null,
    mobilenet_stem_norm_w: ?*c.ggml_tensor = null,
    mm_post_proj_norm_w: ?*c.ggml_tensor = null,
    // Multi-Scale Fusion Adapter (MSFA) components
    msfa_concat_conv_w: ?*c.ggml_tensor = null,
    msfa_concat_norm_w: ?*c.ggml_tensor = null,
    msfa_ffn_expand_w: ?*c.ggml_tensor = null,
    msfa_ffn_project_w: ?*c.ggml_tensor = null,
    msfa_ffn_expand_bn: ?*c.ggml_tensor = null,
    msfa_ffn_project_bn: ?*c.ggml_tensor = null,
    // yasa2
    yasa_patch_w: ?*c.ggml_tensor = null,
    yasa_patch_b: ?*c.ggml_tensor = null,
    yasa_patch_ln_w: ?*c.ggml_tensor = null,
    yasa_patch_ln_b: ?*c.ggml_tensor = null,
    yasa_backbone_ln_w: ?*c.ggml_tensor = null,
    yasa_backbone_ln_b: ?*c.ggml_tensor = null,
    yasa_vision_pos_embed: ?*c.ggml_tensor = null,
    // lfm3vl
    lfm3vl_embedding_norm_w: ?*c.ggml_tensor = null,
    // pixtral, glm4v
    token_embd_img_break: ?*c.ggml_tensor = null,

    // deepseek4v sentinel embeddings (image_newline is reused for IMAGE_NEW_LINE)
    token_embd_img_start: ?*c.ggml_tensor = null,
    token_embd_img_end: ?*c.ggml_tensor = null,
    token_embd_img_pad: ?*c.ggml_tensor = null,
    mm_patch_merger_w: ?*c.ggml_tensor = null,
    mm_patch_merger_b: ?*c.ggml_tensor = null,
    // ultravox / whisper encoder
    conv1d_1_w: ?*c.ggml_tensor = null,
    conv1d_1_b: ?*c.ggml_tensor = null,
    conv1d_2_w: ?*c.ggml_tensor = null,
    conv1d_2_b: ?*c.ggml_tensor = null,
    conv_out_w: ?*c.ggml_tensor = null,
    conv_out_b: ?*c.ggml_tensor = null,
    mm_norm_pre_w: ?*c.ggml_tensor = null,
    mm_norm_pre_b: ?*c.ggml_tensor = null,
    mm_norm_mid_w: ?*c.ggml_tensor = null,
    // mimo-audio-tokenizer: post-transformer downsample + RVQ codebook
    downsample_conv_w: ?*c.ggml_tensor = null, // no bias
    downsample_norm_w: ?*c.ggml_tensor = null,
    downsample_norm_b: ?*c.ggml_tensor = null,
    rvq_codebook: ?*c.ggml_tensor = null, // merged 3D [n_q, max_bins, dim]
    // mimo-v2.5: text-side RVQ code embedding ("text codebook")
    mm_a_code_embd: ?*c.ggml_tensor = null, // merged 3D [n_channels, vocab, dim]
    // mimo-v2.5: LLM-side connector (input_local_transformer, separate from the
    // audio_tokenizer's own encoder `layers`)
    mm_a_local_norm_w: ?*c.ggml_tensor = null,
    // qwen3a
    conv2d_1_w: ?*c.ggml_tensor = null,
    conv2d_1_b: ?*c.ggml_tensor = null,
    conv2d_2_w: ?*c.ggml_tensor = null,
    conv2d_2_b: ?*c.ggml_tensor = null,
    conv2d_3_w: ?*c.ggml_tensor = null,
    conv2d_3_b: ?*c.ggml_tensor = null,
    // qwen3tts speaker encoder (ECAPA-TDNN)
    // reused tensors: stem conv is conv1d_1_w/b, feature aggregation is conv_out_w/b, output proj is mm_fc_w/b
    spk_asp_attn_w: ?*c.ggml_tensor = null,
    spk_asp_attn_b: ?*c.ggml_tensor = null,
    spk_asp_tdnn_w: ?*c.ggml_tensor = null,
    spk_asp_tdnn_b: ?*c.ggml_tensor = null,
    // qwen3tts code_predictor
    gen_code_proj_in_w: ?*c.ggml_tensor = null, // small_to_mtp_projection
    gen_code_proj_in_b: ?*c.ggml_tensor = null,
    gen_code_embd_w: ?*c.ggml_tensor = null, // per-codebook embedding, merged 3D
    gen_code_head_w: ?*c.ggml_tensor = null, // per-codebook output head, merged 3D
    gen_code_out_embd_w: ?*c.ggml_tensor = null, // codebook-0 embedding, fed back into the talker
    gen_code_norm_w: ?*c.ggml_tensor = null, // final norm
    // lfm3a depthformer (MaskGIT)
    gen_code_head_norm_w: ?*c.ggml_tensor = null, // per-codebook norm before the tied head, merged 2D
    gen_code_mask_embd: ?*c.ggml_tensor = null,
    // qwen3tts code2wav: RVQ codes -> raw PCM
    // pocket-tts: SEANet stack, shared by the encoder (speaker path) and the decoder (gen path)
    // pocket-tts: voice latent -> backbone embd (speaker path)
    spk_proj_w: ?*c.ggml_tensor = null,
    downsample_w: ?*c.ggml_tensor = null,
    // pocket-tts: flow-matching decoder, backbone hidden state -> next latent
    gen_out_eos_w: ?*c.ggml_tensor = null,
    gen_out_eos_b: ?*c.ggml_tensor = null,
    gen_input_lin_w: ?*c.ggml_tensor = null, // latent -> backbone embd
    gen_emb_mean: ?*c.ggml_tensor = null,
    gen_emb_std: ?*c.ggml_tensor = null,
    gen_quant_out_w: ?*c.ggml_tensor = null, // latent -> decoder dim
    gen_upsample_w: ?*c.ggml_tensor = null, // depthwise convtr, frame rate -> encoder frame rate
    // cogvlm
    mm_post_fc_norm_w: ?*c.ggml_tensor = null,
    mm_post_fc_norm_b: ?*c.ggml_tensor = null,
    mm_h_to_4h_w: ?*c.ggml_tensor = null,
    mm_gate_w: ?*c.ggml_tensor = null,
    mm_4h_to_h_w: ?*c.ggml_tensor = null,
    mm_boi: ?*c.ggml_tensor = null,
    mm_eoi: ?*c.ggml_tensor = null,
    // hunyuanvl perceiver
    mm_pre_norm_w: ?*c.ggml_tensor = null,
    mm_img_begin: ?*c.ggml_tensor = null,
    mm_img_end: ?*c.ggml_tensor = null,
    // deepseek ocr sam
    patch_embed_proj_w: ?*c.ggml_tensor = null,
    patch_embed_proj_b: ?*c.ggml_tensor = null,
    pos_embed: ?*c.ggml_tensor = null,
    neck_0_w: ?*c.ggml_tensor = null,
    neck_1_w: ?*c.ggml_tensor = null,
    neck_1_b: ?*c.ggml_tensor = null,
    neck_2_w: ?*c.ggml_tensor = null,
    neck_3_w: ?*c.ggml_tensor = null,
    neck_3_b: ?*c.ggml_tensor = null,
    net_2: ?*c.ggml_tensor = null,
    net_3: ?*c.ggml_tensor = null,
    // deepseek-ocr-2
    resample_query_768: ?*c.ggml_tensor = null,
    resample_query_1024: ?*c.ggml_tensor = null,
    // lfm2 audio
    pre_encode_out_w: ?*c.ggml_tensor = null,
    pre_encode_out_b: ?*c.ggml_tensor = null,
    // gemma4
    std_bias: ?*c.ggml_tensor = null,
    std_scale: ?*c.ggml_tensor = null,
    // Gemma4ClippableLinear
    // gemma4 audio conformer
    sscp_inp_proj_w: ?*c.ggml_tensor = null,
    sscp_inp_proj_b: ?*c.ggml_tensor = null,
    audio_out_proj_w: ?*c.ggml_tensor = null,
    audio_out_proj_b: ?*c.ggml_tensor = null,
    // granite_speech encoder
    inp_proj_w: ?*c.ggml_tensor = null,
    inp_proj_b: ?*c.ggml_tensor = null,
    ctc_out_w: ?*c.ggml_tensor = null,
    ctc_out_b: ?*c.ggml_tensor = null,
    ctc_out_mid_w: ?*c.ggml_tensor = null,
    ctc_out_mid_b: ?*c.ggml_tensor = null,
    // qformer projector(s)

    /// Ports `audio_has_avgpool` (clip-model.h:1235 @cd959bae9), a member of `clip_model`.
    pub fn audioHasAvgpool(self: *const Model) bool {
        return self.proj_type == .qwen2a or self.proj_type == .voxtral or self.proj_type == .music_flamingo;
    }

    /// Ports `audio_has_stack_frames` (clip-model.h:1241 @cd959bae9), a member of `clip_model`.
    pub fn audioHasStackFrames(self: *const Model) bool {
        return self.proj_type == .ultravox or self.proj_type == .voxtral or self.proj_type == .meralion;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    impl.refAllDeep(@This());
}
