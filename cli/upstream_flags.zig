//! Upstream's complete flag inventory, for diagnostics only.
//!
//! # Provenance
//!
//! **Not a port.** Extracted from `llama.cpp/common/arg.cpp` (v0.3.0,
//! `c1d0e7a00`) by listing every string in every `common_arg` constructor:
//!
//! ```sh
//! grep -oE '\{"[-a-zA-Z0-9_]+"(, *"[^"]+")*\}' llama.cpp/common/arg.cpp \
//!     | tr -d '{}' | tr ',' '\n' | tr -d ' "' | grep '^-' | sort -u
//! ```
//!
//! # Why this exists
//!
//! Decision 22 says an unsupported flag is rejected with a clear error. "Clear"
//! means distinguishing two cases a user cannot otherwise tell apart:
//!
//! - `--typo` -- not a flag at all, probably a mistake.
//! - `--mirostat` -- a real `llama-cli` flag that this port has not reached yet.
//!
//! Both are refused and both exit non-zero. Only the message differs, and that
//! difference is the whole point: the second tells the user their command line
//! is correct and the port is behind, which is true and useful.
//!
//! This list covers every upstream tool, not just `llama-cli`, so a
//! `llama-server` flag also gets the "not supported here" message rather than
//! "unknown". That is the honest answer: it is a real flag, and we do not take
//! it.
//!
//! It is a *diagnostic*, never a parser input. Nothing here is accepted;
//! `args.zig` owns what is supported, and a flag appearing in both lists would
//! be a bug in `args.zig`, which its tests check for.

/// Every flag string upstream's argument parser recognises, sorted.
pub const all = [_][]const u8{
    "--adaptive-decay",              "--adaptive-target",             "--agent",                      "--alias",
    "--api-key",                     "--api-key-file",                "--api-prefix",                 "--attention",
    "--audio",                       "--backend-sampling",            "--batch-size",                 "--binary-file",
    "--cache-idle-slots",            "--cache-list",                  "--cache-prompt",               "--cache-ram",
    "--cache-reuse",                 "--cache-type-k",                "--cache-type-k-draft",         "--cache-type-v",
    "--cache-type-v-draft",          "--chat-template",               "--chat-template-file",         "--chat-template-kwargs",
    "--check",                       "--check-tensors",               "--checkpoint-min-step",        "--chunk",
    "--chunk-separator",             "--chunk-size",                  "--chunks",                     "--cls-separator",
    "--color",                       "--completion-bash",             "--cont-batching",              "--context-file",
    "--context-shift",               "--control-vector",              "--control-vector-layer-range", "--control-vector-scaled",
    "--conversation",                "--cors-credentials",            "--cors-headers",               "--cors-methods",
    "--cors-origins",                "--cpu-mask",                    "--cpu-mask-batch",             "--cpu-mask-batch-draft",
    "--cpu-mask-draft",              "--cpu-moe",                     "--cpu-moe-draft",              "--cpu-range",
    "--cpu-range-batch",             "--cpu-range-batch-draft",       "--cpu-range-draft",            "--cpu-strict",
    "--cpu-strict-batch",            "--cpu-strict-batch-draft",      "--cpu-strict-draft",           "--ctx-checkpoints",
    "--ctx-size",                    "--defrag-thold",                "--device",                     "--device-draft",
    "--dflash",                      "--diffusion-add-gumbel-noise",  "--diffusion-alg-temp",         "--diffusion-algorithm",
    "--diffusion-block-length",      "--diffusion-cfg-scale",         "--diffusion-eps",              "--diffusion-steps",
    "--diffusion-visual",            "--direct-io",                   "--display-prompt",             "--draft",
    "--draft-max",                   "--draft-min",                   "--draft-n",                    "--draft-n-min",
    "--draft-p-min",                 "--draft-p-split",               "--dry-allowed-length",         "--dry-base",
    "--dry-multiplier",              "--dry-penalty-last-n",          "--dry-sequence-breaker",       "--dynatemp-exp",
    "--dynatemp-range",              "--eagle3",                      "--embd-gemma-default",         "--embd-normalize",
    "--embd-output-format",          "--embd-separator",              "--embedding",                  "--embeddings",
    "--epochs",                      "--escape",                      "--file",                       "--fim-qwen-14b-spec",
    "--fim-qwen-30b-default",        "--fim-qwen-3b-default",         "--fim-qwen-7b-default",        "--fim-qwen-7b-spec",
    "--frequency-penalty",           "--from-chunk",                  "--gpt-oss-120b-default",       "--gpt-oss-20b-default",
    "--gpu-layers",                  "--gpu-layers-draft",            "--grammar",                    "--grammar-file",
    "--grp-attn-n",                  "--grp-attn-w",                  "--hellaswag",                  "--hellaswag-tasks",
    "--help",                        "--hf-file",                     "--hf-repo",                    "--hf-repo-draft",
    "--hf-token",                    "--host",                        "--ids",                        "--ignore-eos",
    "--image",                       "--image-max-tokens",            "--image-min-tokens",           "--in-file",
    "--in-prefix",                   "--in-prefix-bos",               "--in-suffix",                  "--interactive",
    "--interactive-first",           "--jinja",                       "--json-schema",                "--json-schema-file",
    "--junk",                        "--keep",                        "--kl-divergence",              "--kl-divergence-base",
    "--kv-offload",                  "--kv-unified",                  "--learning-rate-decay-epochs", "--list-devices",
    "--load-mode",                   "--log-colors",                  "--log-disable",                "--log-file",
    "--log-prefix",                  "--log-prompts-dir",             "--log-timestamps",             "--log-verbose",
    "--log-verbosity",               "--logit-bias",                  "--logits-output-dir",          "--lookup-cache-dynamic",
    "--lookup-cache-static",         "--lora",                        "--lora-init-without-apply",    "--lora-scaled",
    "--main-gpu",                    "--mcp-servers-config",          "--mcp-servers-json",           "--media-path",
    "--method",                      "--metrics",                     "--min-p",                      "--mirostat",
    "--mirostat-ent",                "--mirostat-lr",                 "--mlock",                      "--mmap",
    "--mmproj",                      "--mmproj-auto",                 "--mmproj-device",              "--mmproj-offload",
    "--mmproj-url",                  "--model",                       "--model-draft",                "--model-url",
    "--models-autoload",             "--models-dir",                  "--models-max",                 "--models-preset",
    "--mtmd-batch-max-tokens",       "--mtp",                         "--multiline-input",            "--multiple-choice",
    "--multiple-choice-tasks",       "--n-cpu-moe",                   "--n-cpu-moe-draft",            "--n-gpu-layers",
    "--n-gpu-layers-draft",          "--n-predict",                   "--negative-file",              "--no-agent",
    "--no-bos",                      "--no-cache-idle-slots",         "--no-cache-prompt",            "--no-cont-batching",
    "--no-context-shift",            "--no-conversation",             "--no-cors-credentials",        "--no-direct-io",
    "--no-display-prompt",           "--no-escape",                   "--no-host",                    "--no-jinja",
    "--no-kv-offload",               "--no-kv-unified",               "--no-log-prefix",              "--no-log-timestamps",
    "--no-mmap",                     "--no-mmproj",                   "--no-mmproj-auto",             "--no-mmproj-offload",
    "--no-models-autoload",          "--no-op-offload",               "--no-parse-special",           "--no-perf",
    "--no-ppl",                      "--no-prefill-assistant",        "--no-reasoning-preserve",      "--no-repack",
    "--no-show-timings",             "--no-skip-chat-parsing",        "--no-slots",                   "--no-spec-draft-backend-sampling",
    "--no-ui",                       "--no-ui-mcp-proxy",             "--no-warmup",                  "--no-webui",
    "--no-webui-mcp-proxy",          "--numa",                        "--offline",                    "--op-offload",
    "--optimizer",                   "--output",                      "--output-file",                "--output-format",
    "--output-frequency",            "--override-kv",                 "--override-tensor",            "--override-tensor-draft",
    "--parallel",                    "--parse-special",               "--path",                       "--pca-batch",
    "--pca-iter",                    "--perf",                        "--poll",                       "--poll-batch",
    "--poll-batch-draft",            "--poll-draft",                  "--pooling",                    "--port",
    "--pos",                         "--positive-file",               "--ppl",                        "--ppl-output-type",
    "--ppl-stride",                  "--predict",                     "--prefill-assistant",          "--presence-penalty",
    "--print-token-count",           "--prio",                        "--prio-batch",                 "--prio-batch-draft",
    "--prio-draft",                  "--process-output",              "--prompt",                     "--prompt-cache",
    "--prompt-cache-all",            "--prompt-cache-ro",             "--props",                      "--reasoning",
    "--reasoning-budget",            "--reasoning-budget-message",    "--reasoning-effort",           "--reasoning-format",
    "--reasoning-preserve",          "--repack",                      "--repeat-last-n",              "--repeat-penalty",
    "--rerank",                      "--reranking",                   "--reuse-port",                 "--reverse-prompt",
    "--rope-freq-base",              "--rope-freq-scale",             "--rope-scale",                 "--rope-scaling",
    "--rpc",                         "--sampler-seq",                 "--samplers",                   "--sampling-seq",
    "--save-all-logits",             "--save-frequency",              "--save-logits",                "--seed",
    "--sequences",                   "--server-base",                 "--show-count",                 "--show-statistics",
    "--show-timings",                "--simple-io",                   "--single-turn",                "--skip-chat-parsing",
    "--sleep-idle-seconds",          "--slot-prompt-similarity",      "--slot-save-path",             "--slots",
    "--spec-default",                "--spec-draft-backend-sampling", "--spec-draft-cpu-mask",        "--spec-draft-cpu-mask-batch",
    "--spec-draft-cpu-moe",          "--spec-draft-cpu-range",        "--spec-draft-cpu-range-batch", "--spec-draft-cpu-strict",
    "--spec-draft-cpu-strict-batch", "--spec-draft-device",           "--spec-draft-hf",              "--spec-draft-model",
    "--spec-draft-n-cpu-moe",        "--spec-draft-n-max",            "--spec-draft-n-min",           "--spec-draft-ncmoe",
    "--spec-draft-ngl",              "--spec-draft-override-tensor",  "--spec-draft-p-min",           "--spec-draft-p-split",
    "--spec-draft-poll",             "--spec-draft-poll-batch",       "--spec-draft-prio",            "--spec-draft-prio-batch",
    "--spec-draft-threads",          "--spec-draft-threads-batch",    "--spec-draft-type-k",          "--spec-draft-type-v",
    "--spec-ngram-map-k-min-hits",   "--spec-ngram-map-k-size-m",     "--spec-ngram-map-k-size-n",    "--spec-ngram-map-k4v-min-hits",
    "--spec-ngram-map-k4v-size-m",   "--spec-ngram-map-k4v-size-n",   "--spec-ngram-min-hits",        "--spec-ngram-mod-n-match",
    "--spec-ngram-mod-n-max",        "--spec-ngram-mod-n-min",        "--spec-ngram-simple-min-hits", "--spec-ngram-simple-size-m",
    "--spec-ngram-simple-size-n",    "--spec-ngram-size-m",           "--spec-ngram-size-n",          "--spec-type",
    "--special",                     "--split-mode",                  "--spm-infill",                 "--sse-ping-interval",
    "--ssl-cert-file",               "--ssl-key-file",                "--stdin",                      "--swa-checkpoints",
    "--swa-full",                    "--system-prompt",               "--system-prompt-file",         "--tags",
    "--temp",                        "--temperature",                 "--tensor-filter",              "--tensor-split",
    "--threads",                     "--threads-batch",               "--threads-batch-draft",        "--threads-draft",
    "--threads-http",                "--timeout",                     "--tools",                      "--tools-runtime",
    "--top-k",                       "--top-n-sigma",                 "--top-nsigma",                 "--top-p",
    "--tts-lang",                    "--tts-speaker-file",            "--typical",                    "--typical-p",
    "--ubatch-size",                 "--ui",                          "--ui-config",                  "--ui-config-file",
    "--ui-mcp-proxy",                "--usage",                       "--val-split",                  "--verbose",
    "--verbose-prompt",              "--verbosity",                   "--version",                    "--video",
    "--vision-gemma-12b-default",    "--vision-gemma-4b-default",     "--warmup",                     "--webui",
    "--webui-config",                "--webui-config-file",           "--webui-mcp-proxy",            "--weight-decay",
    "--winogrande",                  "--winogrande-tasks",            "--xtc-probability",            "--xtc-threshold",
    "--yarn-attn-factor",            "--yarn-beta-fast",              "--yarn-beta-slow",             "--yarn-ext-factor",
    "--yarn-orig-ctx",               "-C",                            "-Cb",                          "-Cbd",
    "-Cd",                           "-Cr",                           "-Crb",                         "-Crbd",
    "-Crd",                          "-a",                            "-ag",                          "-b",
    "-bf",                           "-bs",                           "-c",                           "-cb",
    "-cl",                           "-cmoe",                         "-cmoed",                       "-cms",
    "-cnv",                          "-co",                           "-cram",                        "-ctk",
    "-ctkd",                         "-ctv",                          "-ctvd",                        "-ctxcp",
    "-decay-epochs",                 "-dev",                          "-devd",                        "-dio",
    "-dt",                           "-e",                            "-epochs",                      "-f",
    "-gan",                          "-gaw",                          "-h",                           "-hf",
    "-hfd",                          "-hff",                          "-hfr",                         "-hfrd",
    "-hft",                          "-i",                            "-if",                          "-j",
    "-jf",                           "-kvo",                          "-kvu",                         "-l",
    "-lcd",                          "-lcs",                          "-lm",                          "-lv",
    "-m",                            "-md",                           "-mg",                          "-mli",
    "-mm",                           "-mmdev",                        "-mmu",                         "-mu",
    "-n",                            "-ncmoe",                        "-ncmoed",                      "-ndio",
    "-ngl",                          "-ngld",                         "-nkvo",                        "-no-ag",
    "-no-cnv",                       "-no-kvu",                       "-nocb",                        "-np",
    "-npl",                          "-npp",                          "-nr",                          "-ns",
    "-ntg",                          "-o",                            "-ofreq",                       "-opt",
    "-ot",                           "-otd",                          "-p",                           "-pps",
    "-ptc",                          "-r",                            "-rea",                         "-s",
    "-sm",                           "-sp",                           "-sps",                         "-st",
    "-sys",                          "-sysf",                         "-t",                           "-tb",
    "-tbd",                          "-td",                           "-tgs",                         "-to",
    "-ts",                           "-ub",                           "-v",                           "-val-split",
    "-wd",
};

/// Whether `flag` is a flag upstream recognises.
///
/// Parameters:
/// - `flag`: the argument as written, leading dashes included.
///
/// Return: true when upstream would accept it somewhere.
pub fn contains(flag: []const u8) bool {
    const std = @import("std");
    for (all) |f| {
        if (std.mem.eql(u8, f, flag)) return true;
    }
    return false;
}
