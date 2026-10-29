//! `common_params`: what the tools' command lines set.
//!
//! # Provenance
//!
//! Ports `common_cpu_params`, `common_params_sampling`, `common_params_model`
//! and `common_params` from `llama.cpp/common/common.h` (v0.4.1,
//! `b29c606e2`) -- the members the tools' options and the tools themselves
//! touch, with upstream's defaults. The server's, the CLI's and the
//! speculative decoder's members are left out: no option those tools take
//! reaches them.
//!
//! # Ownership
//!
//! Every string and list is the arena's that `Params.init` takes: a tool
//! parses once, runs, and exits.

const std = @import("std");
const c = @import("tools_h");

/// Ports `llama_example` (common/common.h:84 @b29c606e2).
pub const Example = enum(c_int) {
    batched,
    debug,
    common,
    speculative,
    completion,
    cli,
    embedding,
    perplexity,
    retrieval,
    passkey,
    imatrix,
    bench,
    server,
    cvector_generator,
    export_lora,
    mtmd,
    lookup,
    parallel,
    tts,
    diffusion,
    finetune,
    fit_params,
    results,
    export_graph_ops,
    download,
    tokenize,
};

/// Ports `common_sampler_type` (common/common.h:115 @b29c606e2).
pub const SamplerType = enum(c_int) {
    none = 0,
    dry = 1,
    top_k = 2,
    top_p = 3,
    min_p = 4,
    typical_p = 6,
    temperature = 7,
    xtc = 8,
    infill = 9,
    penalties = 10,
    top_n_sigma = 11,
    adaptive_p = 12,

    /// Ports `common_sampler_type_to_chr` (sampling.cpp:795 @b29c606e2).
    pub fn toChr(self: SamplerType) u8 {
        return switch (self) {
            .dry => 'd',
            .top_k => 'k',
            .typical_p => 'y',
            .top_p => 'p',
            .top_n_sigma => 's',
            .min_p => 'm',
            .temperature => 't',
            .xtc => 'x',
            .infill => 'i',
            .penalties => 'e',
            .adaptive_p => 'a',
            else => '?',
        };
    }

    /// Ports `common_sampler_type_to_str` (sampling.cpp:812 @b29c606e2).
    pub fn toStr(self: SamplerType) []const u8 {
        return switch (self) {
            .dry => "dry",
            .top_k => "top_k",
            .typical_p => "typ_p",
            .top_p => "top_p",
            .top_n_sigma => "top_n_sigma",
            .min_p => "min_p",
            .temperature => "temperature",
            .xtc => "xtc",
            .infill => "infill",
            .penalties => "penalties",
            .adaptive_p => "adaptive_p",
            else => "",
        };
    }
};

/// Ports `common_params_sampling_config` (common/common.h:156 @b29c606e2).
pub const SamplingConfig = struct {
    pub const samplers: u64 = 1 << 0;
    pub const top_k: u64 = 1 << 1;
    pub const top_p: u64 = 1 << 2;
    pub const min_p: u64 = 1 << 3;
    pub const xtc_probability: u64 = 1 << 4;
    pub const xtc_threshold: u64 = 1 << 5;
    pub const temp: u64 = 1 << 6;
    pub const penalty_last_n: u64 = 1 << 7;
    pub const penalty_repeat: u64 = 1 << 8;
    pub const mirostat: u64 = 1 << 9;
    pub const mirostat_tau: u64 = 1 << 10;
    pub const mirostat_eta: u64 = 1 << 11;
};

/// Ports `dimre_method` (common/common.h:132 @b29c606e2).
pub const DimreMethod = enum { pca, mean };

/// Ports `common_grammar_type` (common/common.h:187 @b29c606e2).
pub const GrammarType = enum { none, user, output_format, tool_calls };

/// Ports `common_cpu_params` (common/common.h:68 @b29c606e2).
pub const CpuParams = struct {
    n_threads: i32 = -1,
    cpumask: [c.GGML_MAX_N_THREADS]bool = @splat(false), // CPU affinity mask.
    mask_valid: bool = false, // Default: any CPU
    priority: c.ggml_sched_priority = c.GGML_SCHED_PRIO_NORMAL,
    strict_cpu: bool = false, // Use strict CPU placement
    poll: u32 = 50, // Polling (busywait) level (0 - no polling, 100 - mostly polling)
};

/// Ports `common_adapter_lora_info` (common/common.h:50 @b29c606e2).
pub const LoraInfo = struct {
    path: [:0]const u8,
    scale: f32,
    task_name: []const u8 = "",
    prompt_prefix: []const u8 = "",
    ptr: ?*c.llama_adapter_lora = null,
};

/// Ports `common_control_vector_load_info` (common/common.h:1105 @b29c606e2).
pub const ControlVectorLoadInfo = struct {
    strength: f32,
    fname: [:0]const u8,
};

/// Ports `common_params_sampling` (common/common.h:224 @b29c606e2), the members an
/// option sets.
pub const Sampling = struct {
    seed: u32 = c.LLAMA_DEFAULT_SEED, // the seed used to initialize llama_sampler
    n_prev: i32 = 64, // number of previous tokens to remember
    n_probs: i32 = 0, // if greater than 0, output the probabilities of top n_probs tokens.
    min_keep: i32 = 0, // 0 = disabled, otherwise samplers should return at least min_keep tokens
    top_k: i32 = 40, // <= 0 to use vocab size
    top_p: f32 = 0.95, // 1.0 = disabled
    min_p: f32 = 0.05, // 0.0 = disabled
    xtc_probability: f32 = 0.00, // 0.0 = disabled
    xtc_threshold: f32 = 0.10, // > 0.5 disables XTC
    typ_p: f32 = 1.00, // typical_p, 1.0 = disabled
    temp: f32 = 0.80, // <= 0.0 to sample greedily, 0.0 to not output probabilities
    dynatemp_range: f32 = 0.00, // 0.0 = disabled
    dynatemp_exponent: f32 = 1.00, // controls how entropy maps to temperature in dynamic temperature sampler
    penalty_last_n: i32 = 64, // last n tokens to penalize (0 = disable penalty)
    penalty_repeat: f32 = 1.00, // 1.0 = disabled
    penalty_freq: f32 = 0.00, // 0.0 = disabled
    penalty_present: f32 = 0.00, // 0.0 = disabled
    dry_multiplier: f32 = 0.0, // 0.0 = disabled
    dry_base: f32 = 1.75, // 0.0 = disabled
    dry_allowed_length: i32 = 2, // tokens extending repetitions beyond this receive penalty
    dry_penalty_last_n: i32 = 64, // how many tokens to scan for repetitions (0 = disable penalty)
    adaptive_target: f32 = -1.0, // select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
    adaptive_decay: f32 = 0.90, // EMA decay for adaptation
    mirostat: i32 = 0, // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    top_n_sigma: f32 = -1.00, // -1.0 = disabled
    mirostat_tau: f32 = 5.00, // target entropy
    mirostat_eta: f32 = 0.10, // learning rate
    ignore_eos: bool = false,
    no_perf: bool = false, // disable performance metrics
    user_sampling_config: u64 = 0, // bitfield to track user-specified samplers
    dry_sequence_breakers: std.ArrayList([]const u8) = .empty,
    samplers: std.ArrayList(SamplerType) = .empty,
    grammar_type: GrammarType = .none,
    grammar: []const u8 = "",
    grammar_lazy: bool = false,
    logit_bias: std.ArrayList(c.llama_logit_bias) = .empty,
    backend_sampling: bool = false,
};

/// Ports `common_params_model` (common/common.h:302 @b29c606e2).
pub const Model = struct {
    path: []const u8 = "", // model local path
    url: []const u8 = "", // model url to download
    hf_repo: []const u8 = "", // HF repo
    hf_file: []const u8 = "", // HF file
    docker_repo: []const u8 = "", // Docker repo
};

/// Ports `common_params` (common/common.h:448 @b29c606e2), the members the tools'
/// options and the tools themselves reach. `default_template_kwargs` and
/// `preserve_reasoning_specified` are left out: no option the tools take sets
/// them, and the one reader, `common_chat_format_example`, ignores them on the
/// legacy path, the only one `chat.zig` ports.
pub const Params = struct {
    arena: std.mem.Allocator,

    n_predict: i32 = -1, // max. number of new tokens to predict, -1 == no limit
    n_ctx: i32 = 0, // context size, 0 == context the model was trained with
    n_batch: i32 = 2048, // logical batch size for prompt processing
    n_ubatch: i32 = 512, // physical batch size for prompt processing
    n_keep: i32 = 0, // number of tokens to keep from initial prompt
    n_chunks: i32 = -1, // max number of chunks to process (-1 = unlimited)
    n_parallel: i32 = 1, // number of parallel sequences to decode
    n_sequences: i32 = 1, // number of sequences to decode
    n_outputs_max: i32 = 0, // max outputs in a batch (0 = n_batch)
    n_outputs_max_per_seq: i32 = 1, // max outputs per sequence
    rope_freq_base: f32 = 0.0, // RoPE base frequency
    rope_freq_scale: f32 = 0.0, // RoPE frequency scaling factor
    yarn_ext_factor: f32 = -1.0, // YaRN extrapolation mix factor
    yarn_attn_factor: f32 = -1.0, // YaRN magnitude scaling factor
    yarn_beta_fast: f32 = -1.0, // YaRN low correction dim
    yarn_beta_slow: f32 = -1.0, // YaRN high correction dim
    yarn_orig_ctx: i32 = 0, // YaRN original context length

    devices: std.ArrayList(c.ggml_backend_dev_t) = .empty, // devices to use for offloading

    n_gpu_layers: i32 = -1, // number of layers to store in VRAM, -1 is auto, <= -2 is all
    main_gpu: i32 = 0, // the GPU that is used for scratch and small tensors
    tensor_split: [128]f32 = @splat(0), // how split tensors should be distributed across GPUs
    fit_params: bool = true, // whether to fit unset model/context parameters to free device memory
    fit_params_print: bool = false, // print the estimated required memory to run the model
    fit_params_min_ctx: i32 = 4096, // minimum context size to set when trying to reduce memory use
    fit_params_target: std.ArrayList(usize) = .empty, // margin per device, set by `init`

    split_mode: c.llama_split_mode = c.LLAMA_SPLIT_MODE_LAYER, // how to split the model across GPUs
    load_mode: c.llama_load_mode = c.LLAMA_LOAD_MODE_AUTO, // how to load the model

    lazy_mode: c.llama_lazy_mode = c.LLAMA_LAZY_MODE_AUTO, // on-demand reading of tensors marked by the arch

    cpuparams: CpuParams = .{},
    cpuparams_batch: CpuParams = .{},

    cb_eval: c.ggml_backend_sched_eval_callback = null,
    cb_eval_user_data: ?*anyopaque = null,

    numa: c.ggml_numa_strategy = c.GGML_NUMA_STRATEGY_DISABLED,

    rope_scaling_type: c.llama_rope_scaling_type = c.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
    pooling_type: c.enum_llama_pooling_type = c.LLAMA_POOLING_TYPE_UNSPECIFIED, // pooling type for embeddings
    attention_type: c.llama_attention_type = c.LLAMA_ATTENTION_TYPE_UNSPECIFIED, // attention type for embeddings
    flash_attn_type: c.llama_flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_AUTO, // whether to use Flash Attention

    sampling: Sampling = .{},
    /// `speculative.draft.cache_type_k` and `_v`.
    spec_draft_cache_type_k: c.ggml_type = c.GGML_TYPE_F16,
    spec_draft_cache_type_v: c.ggml_type = c.GGML_TYPE_F16,

    model: Model = .{},

    hf_token: []const u8 = "", // HF token (aka bearer token)
    prompt: []const u8 = "",
    prompt_file: []const u8 = "", // store the external prompt file name
    logits_file: []const u8 = "", // file for saving *all* logits

    in_files: std.ArrayList([]const u8) = .empty, // all input files
    kv_overrides: std.ArrayList(c.llama_model_kv_override) = .empty,
    tensor_buft_overrides: std.ArrayList(c.llama_model_tensor_buft_override) = .empty,

    lora_init_without_apply: bool = false, // only load lora to memory, but do not apply it to ctx
    lora_adapters: std.ArrayList(LoraInfo) = .empty, // lora adapter path with user defined scale

    control_vectors: std.ArrayList(ControlVectorLoadInfo) = .empty, // control vector with user defined scale

    verbosity: i32 = 3, // LOG_LEVEL_INFO
    control_vector_layer_start: i32 = -1, // layer range for control vector
    control_vector_layer_end: i32 = -1, // layer range for control vector
    offline: bool = false,

    ppl_stride: i32 = 0, // stride for perplexity calculations
    ppl_output_type: i32 = 0, // = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line
    hellaswag: bool = false, // compute HellaSwag score over random tasks from datafile supplied in prompt
    hellaswag_tasks: usize = 400, // number of tasks to use when computing the HellaSwag score

    winogrande: bool = false, // compute Winogrande score over random tasks from datafile supplied in prompt
    winogrande_tasks: usize = 0, // number of tasks to use when computing the Winogrande score

    multiple_choice: bool = false, // compute TruthfulQA score over random tasks from datafile supplied in prompt
    multiple_choice_tasks: usize = 0, // number of tasks to use when computing the TruthfulQA score

    kl_divergence: bool = false, // compute KL divergence

    check: bool = false, // check rather than generate results for llama-results

    usage: bool = false, // print usage
    completion: bool = false, // print source-able completion script
    use_color: bool = false, // use color to distinguish generations and inputs

    escape: bool = true, // escape "\n", "\r", "\t", "\'", "\"", and "\\"
    no_perf: bool = false, // disable performance metrics
    ctx_shift: bool = false, // context shift on infinite text generation
    swa_full: bool = false, // use full-size SWA cache
    kv_unified: bool = false, // enable unified KV cache

    no_kv_offload: bool = false, // disable KV offloading
    warmup: bool = true, // warmup run
    check_tensors: bool = false, // validate tensor data
    no_op_offload: bool = false, // globally disable offload host tensor operations to device
    no_extra_bufts: bool = false, // disable extra buffer types (used for weight repacking)
    no_host: bool = false, // bypass host buffer allowing extra buffers to be used

    cache_type_k: c.ggml_type = c.GGML_TYPE_F16, // KV cache data type for the K
    cache_type_v: c.ggml_type = c.GGML_TYPE_F16, // KV cache data type for the V

    mmproj: Model = .{},
    mmproj_use_gpu: bool = true, // use GPU for multimodal model
    mmproj_device: c.ggml_backend_dev_t = null, // GPU device to use for multimodal model
    image_min_tokens: c_int = -1,
    image_max_tokens: c_int = -1,
    no_mmproj: bool = false, // explicitly disable multimodal model
    image: std.ArrayList([]const u8) = .empty, // path to image file(s)

    // for video input
    video_fps: f32 = 4.0,
    video_timestamp_interval_ms: i64 = 5000,
    video_ffmpeg_bin_dir: []const u8 = "",

    system_prompt: []const u8 = "",
    simple_io: bool = false, // improves compatibility with subprocesses and limited consoles
    chat_template: []const u8 = "",
    use_jinja: bool = true,

    tts_lang: []const u8 = "",
    tts_speaker_file: []const u8 = "",

    embedding: bool = false, // get only sentence embedding

    is_pp_shared: bool = false,
    is_tg_separate: bool = false,

    n_pp: std.ArrayList(i32) = .empty,
    n_tg: std.ArrayList(i32) = .empty,
    n_pl: std.ArrayList(i32) = .empty,

    n_out_freq: i32 = 10, // output the imatrix every n_out_freq iterations
    n_save_freq: i32 = 0, // save the imatrix every n_save_freq iterations
    i_chunk: i32 = 0, // start processing from this chunk
    imat_dat: i8 = 0, // whether the legacy imatrix.dat format should be output (gguf <= 0 < dat)

    process_output: bool = false, // collect data for the output tensor
    compute_ppl: bool = true, // whether to compute perplexity
    show_statistics: bool = false, // show imatrix statistics per tensor
    parse_special: bool = false, // whether to parse special tokens during imatrix tokenization

    n_pca_batch: c_int = 100,
    n_pca_iterations: c_int = 1000,
    cvector_dimre_method: DimreMethod = .pca,
    cvector_positive_file: []const u8 = "tools/cvector-generator/positive.txt",
    cvector_negative_file: []const u8 = "tools/cvector-generator/negative.txt",

    batched_bench_output_jsonl: bool = false,

    tokenize_ids: bool = false, // if true, only print the token IDs
    tokenize_stdin: bool = false, // if true, read the prompt from stdin
    tokenize_no_bos: bool = false, // if true, do not add the BOS token
    tokenize_show_count: bool = false, // if true, print the total token count

    out_file: []const u8 = "", // output filename for all example programs
    no_alloc: bool = false, // Don't allocate model buffers

    /// The C++ default member initialisers that are not constants: the
    /// sampler chain, the DRY breakers, and one fit margin per device.
    ///
    /// Return: the defaults, owning `arena`'s lists.
    pub fn init(arena: std.mem.Allocator) !Params {
        var p: Params = .{ .arena = arena };
        try p.sampling.dry_sequence_breakers.appendSlice(arena, &.{ "\n", ":", "\"", "*" });
        try p.sampling.samplers.appendSlice(arena, &.{
            .penalties, .dry,   .top_n_sigma, .top_k,       .typical_p,
            .top_p,     .min_p, .xtc,         .temperature,
        });
        try p.fit_params_target.appendNTimes(arena, 1024 * 1024 * 1024, c.llama_max_devices());
        return p;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
