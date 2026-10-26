# SPEC — llamazig

> **What this document is.** The specification of *what* llamazig is and what it
> does: its artifacts, interfaces, observable behaviour, and limits. It is the
> contract, not the reasoning and not the method.
>
> | File | Question | Contents |
> |---|---|---|
> | `README.md` | **why** | Motivation, context, current status |
> | `SPEC.md` | **what** | This file: surface, behaviour, guarantees |
> | `PLAN.md` | **how** | Decisions, scope measurements, what comes next |
> | `NOTES.md` | **why** | The record of each completed step and what each gate caught |
>
> **Rules for this file.** Describe observable behaviour, not implementation.
> State facts, not rationale. Prefer tables to paragraphs. A sentence that
> answers "why did we…" belongs in `NOTES.md`; one that answers "how is it
> done" belongs in `PLAN.md` or in the code. Everything here must be true of
> the current tree — a spec that describes an intention is a plan.
>
> Anything not yet built is listed under [§9 Not in this
> deliverable](#9-not-in-this-deliverable) rather than described as if it
> exists.

---

## 1. Deliverables

| Artifact | Path | What it is |
|---|---|---|
| `llama-cli` | `zig-out/bin/llama-cli` | Command-line completion and conversation binary |
| `libggml.a` | `zig-out/lib/libggml.a` | ggml, ported; the two Metal Objective-C files and the embedded kernels beside it |
| `libllama.a` | `zig-out/lib/libllama.a` | llama.cpp's model layer, ported: exports `llama.h` |
| `libllama-ref.a` | `zig-out/lib/libllama-ref.a` | Upstream's libllama, the pinned C++ unmodified — a test oracle, not a deliverable |
| `libllamazig.a` | `zig-out/lib/libllamazig.a` | This project's own library scaffold |
| docs | `zig-out/docs/` | Generated API documentation |

Our `llama-cli` is a distinct binary that shares upstream's name, not upstream's `llama-cli` relinked.

## 2. Scope

**In.** ggml (tensor core, allocator, quantization, the CPU backend's ARM
and x86 kernels), the Metal, CUDA and Vulkan backends' host layers,
libllama, and a `llama-cli`-compatible completion and conversation binary.

**Out.** `common/` beyond what the tools below use, `tools/server/`,
upstream's `llama-cli`, training, and model conversion. The library reads
GGUF and `llama-cli` never writes one.

**Also in, as separate binaries (`PLAN.md`, Decision 61).** Upstream's
`gguf-split`, `quantize`, `tokenize`, `fit-params`, `export-lora`,
`batched-bench`, `cvector-generator`, `imatrix`, `perplexity` and
`llama-bench`, each a Zig port named as upstream names it, with upstream's
command line and output. Five of them write GGUF.

## 3. Library: ggml

### 3.1 What is Zig

Ported translation units keep the C ABI of the file they replace: same exported
symbol names, same signatures. Callers link against them without change.

| Reference source | Lines | Exported symbols | Replaced by |
|---|---:|---:|---|
| `ggml/src/ggml-alloc.c` | 1,249 | — | `src/ggml/alloc.zig` |
| `ggml/src/ggml.c` | 8,067 | 381 | `src/ggml/{impl,types,context,runtime,ops,graph,quantize}.zig` |
| `ggml/src/ggml-quants.c` | 5,667 | 79 | `src/ggml/quants/` |
| `ggml/src/ggml-cpu/ggml-cpu.c` | 3,900 | 65 | `src/ggml/cpu/` |
| `ggml/src/ggml-cpu/quants.c` | 1,339 | 45 | `src/ggml/cpu/quants/` |
| `ggml/src/ggml-cpu/arch/arm/quants.c` | 4,319 | 28 | `src/ggml/cpu/quants/arm/` |
| `ggml/src/ggml-cpu/arch/x86/quants.c` | 4,108 | 27 | `src/ggml/cpu/quants/x86/` (AVX2 arms) |

And every C++ translation unit:

| Reference source | Lines | Exported symbols | Replaced by |
|---|---:|---:|---|
| `ggml/src/ggml-threading.cpp` | 12 | 3 | `src/ggml/threading.zig` |
| `ggml/src/ggml-backend-reg.cpp` | 593 | 16 | `src/ggml/backend_reg.zig` |
| `ggml/src/gguf.cpp` | 1,706 | 61 | `src/ggml/gguf.zig` |
| `ggml/src/ggml-backend.cpp` | 2,443 | 102 | `src/ggml/{backend,backend_sched}.zig` |
| `ggml/src/ggml-cpu/binary-ops.cpp` | 154 | 4 | `src/ggml/cpu/binary_ops.zig` |
| `ggml/src/ggml-cpu/unary-ops.cpp` | 337 | 23 | `src/ggml/cpu/unary_ops.zig` |
| `ggml/src/ggml-cpu/vec.cpp` | 613 | 10 | `src/ggml/cpu/vec.zig` |
| `ggml/src/ggml-cpu/ops.cpp` | 12,021 | 88 | `src/ggml/cpu/ops/` |
| `ggml/src/ggml-cpu/llamafile/sgemm.cpp` | 4,164 | 1 | `src/ggml/cpu/ops/sgemm.zig` |
| `ggml/src/ggml-cpu/traits.cpp` | 36 | 2 | `src/ggml/cpu/extra.zig` |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | 712 | 7 | `src/ggml/cpu/cpu_backend.zig` |
| `ggml/src/ggml-cpu/repack.cpp` | 4,836 | 36 | `src/ggml/cpu/repack/` |
| `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 5,156 | 28 | `src/ggml/cpu/repack/arm/` |
| `ggml/src/ggml-cpu/arch/x86/repack.cpp` | 6,407 | 12 | `src/ggml/cpu/repack/x86/` (AVX2 arms) |
| `ggml/src/ggml-metal/ggml-metal-common.cpp` | 457 | 6 | `src/ggml/metal/common.zig` |
| `ggml/src/ggml-metal/ggml-metal-tuning.cpp` | 1,087 | 7 (mangled) | `src/ggml/metal/{tuning,tuning_table}.zig` |
| `ggml/src/ggml-metal/ggml-metal.cpp` | 998 | 6 | `src/ggml/metal/backend.zig` |
| `ggml/src/ggml-metal/ggml-metal-device.cpp` | 2,267 | 73 | `src/ggml/metal/library.zig` |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | 5,369 | 66 | `src/ggml/metal/ops.zig` |
| `ggml/src/ggml-backend-meta.cpp` | 2,495 | 8 | `src/ggml/backend_meta.zig` |
| `ggml/src/ggml-opt.cpp` | 1,094 | 37 | `src/ggml/opt.zig` |
| `ggml/src/ggml-cuda/*.cu` host code | ~43,900 (with kernels) | — | `src/ggml/cuda/` |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | 19,787 | 9 | `src/ggml/vulkan/` |
| `src/llama*.cpp`, `src/unicode*.cpp` | 47,215 | 240 (all of `llama.h`) | `src/llama/`, exported by `src/llama/exports.zig` |
| `src/models/qwen35.cpp` | 645 | — | `src/llama/models/{qwen35,delta_net}.zig` |

`ggml/src/ggml.cpp` is dropped rather than ported: it only installed a
`std::terminate` handler, and no C++ is left to throw.

`scripts/port-coverage` derives each symbol contract by compiling the source
and reports the count. For a C++ file the contract is the *unmangled*
exports — with one documented exception: `repack.cpp` also has a C++-linkage
dispatch layer that no symbol count can see, and `scripts/cluster-check`
covers that half.

The last four move as one unit: `repack.cpp` derives from the two abstract
bases `traits.cpp` declares, and `ggml-cpu.cpp` registers the buffer type
`repack.cpp` builds.

### 3.2 What is not Zig

No C or C++ compiles into `libggml.a` or `libllama.a` apart from three
objects that are empty on this target. `llama-cli` links no C++ runtime.

| Component | Language | Note |
|---|---|---|
| `ggml-cpu/hbm.cpp`, `amx/amx.cpp`, `amx/mmq.cpp` | C++ | All three empty on this target: `GGML_USE_CPU_HBM` and `__AMX_INT8__` are off |
| `ggml-metal-device.m`, `ggml-metal-context.m` | Objective-C | Metal object interface; stays Objective-C permanently |
| `ggml-metal/kernels/*.metal` | MSL | Compiled by the GPU driver at load; never ported |
| `ggml-cuda/*.cu` kernels | CUDA | Compiled unmodified to fatbins by `nvcc`, embedded, loaded through the driver API |
| `src/ggml/cuda/cub_shim.cu` | CUDA C++ | CUB's device-wide algorithms behind a C ABI; the only host code the port writes in C++ |
| `ggml-vulkan/vulkan-shaders/*` | GLSL | The reference build's generated SPIR-V, embedded; never regenerated |

Every architecture in upstream's `llama_model_mapping` is ported. Qwen3.5 is
the canonical model of the conformance gates; the others are gated on
upstream's per-architecture fixtures (`make arch-model-diff`).

### 3.3 Backends

| Backend | State |
|---|---|
| Metal | Enabled. All 20 shader libraries embedded in the binary; no external `.metallib`. |
| CPU | Enabled. Accelerate on, llamafile SGEMM on, BLAS and OpenMP off. |
| CUDA | Enabled on Linux when `nvcc` is found (`-Dcuda`). Host layer in Zig; upstream's `.cu` kernels compiled unmodified to fatbins and embedded. |
| Vulkan | Enabled with `-Dvulkan` (Linux, pinned toolchain in `.tools/vulkan/`). Host layer in Zig; the reference generator's SPIR-V embedded. |
| BLAS | Apple only, enabled with `-Dblas`: the backend in Zig, over the Accelerate the CPU backend already links. Off by default, as in the reference configuration. No other BLAS: each is an external library (PLAN.md, Decision 60). |
| zDNN, ZenDNN | Not built: each exists to call a vendor library (Decision 60). |
| SYCL, HIP | Not built. |

### 3.4 Data types

`enum ggml_type` has 44 members. The CPU type table carries an entry for the
31 the reference does, matching it exactly:

| | Count | Types |
|---|---:|---|
| Dot-product kernel | 28 | Every listed type except `Q8_1`, `Q8_K` and `I32` |
| Quantize from `f32` | 24 | All but `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ1_S`, `IQ1_M` |

Types the reference leaves out of the table — `I8`, `I16`, `I64`, `F64` and the
rest — have neither.

## 4. Executable: `llama-cli`

### 4.1 Invocation

```
llamazig -m PATH [options]
```

`-m` is required. The binary reads a GGUF model, tokenizes a prompt, generates,
and exits; under `-cnv` it holds a conversation on stdin instead (§5.3).

### 4.2 Options

Flag names, aliases, argument forms and defaults match upstream `llama-cli`.
A command line written for upstream runs unchanged for the options below.

| Flag | Value | Default | Meaning |
|---|---|---|---|
| `-h`, `--help`, `--usage` | — | — | Print help, exit 0 |
| `--version` | — | — | Print version, exit 0 |
| `-m`, `--model` | `PATH` | *(required)* | Model file |
| `-p`, `--prompt` | `TEXT` | `""` | Prompt to complete |
| `-n`, `--predict`, `--n-predict` | `N` | `-1` | Tokens to generate; `-1` = until stop |
| `-c`, `--ctx-size` | `N` | `0` | Context size; `0` = the model's own, which `--fit` may shrink — unless `-c 0` is given explicitly |
| `-b`, `--batch-size` | `N` | `2048` | Logical batch size |
| `-ngl`, `--gpu-layers`, `--n-gpu-layers` | `N` | `-1` | Layers to offload; `-1` = auto |
| `-fit`, `--fit` | `on`\|`off` | `on` | Fit the parameters left unset to free memory before loading (§4.5) |
| `-fitt`, `--fit-target` | `MiB0,MiB1,...` | `1024` | Memory `--fit` leaves free per device; one value applies to every device |
| `-fitc`, `--fit-ctx` | `N` | `4096` | The least context `--fit` shrinks to |
| `-t`, `--threads` | `N` | `0` | Threads; `0` = chosen by libllama |
| `-s`, `--seed` | `N` | `0xFFFFFFFF` | RNG seed; default requests a random one |
| `--temp`, `--temperature` | `F` | `0.80` | Temperature; `<= 0` samples greedily |
| `--top-k` | `N` | `40` | Top-k; `<= 0` uses the vocabulary size |
| `--top-p` | `F` | `0.95` | Top-p; `1.0` disables |
| `--min-p` | `F` | `0.05` | Min-p; `0.0` disables |
| `--repeat-last-n` | `N` | `64` | Tokens penalized; `0` disables, `-1` = context size |
| `--repeat-penalty` | `F` | `1.00` | Repeat penalty; `1.0` disables |
| `--ignore-eos` | — | off | Never stop at end-of-generation |
| `-e`, `--escape` | — | **on** | Interpret `\n`, `\t`, `\\` in the prompt |
| `--no-escape` | — | — | Take the prompt literally |
| `--no-warmup` | — | — | Skip the warmup decode |
| `--no-display-prompt` | — | — | Do not echo the prompt before the completion |
| `--show-timings` | — | **on** | Print prompt and generation tokens/second |
| `--no-show-timings` | — | — | Suppress the timings line |
| `--jinja` | — | off | Render the prompt through a chat template |
| `--no-jinja` | — | — | Raw completion; never apply a chat template |
| `--chat-template` | `TEXT` | — | Jinja template to use instead of the model's |
| `--chat-template-file` | `FNAME` | — | The same, read from a file |
| `-sys`, `--system-prompt` | `TEXT` | `""` | System message prepended to the conversation |
| `-sysf`, `--system-prompt-file` | `FNAME` | — | The same, read from a file |
| `-cnv`, `--conversation`, `-i`, `--interactive` | — | off | Multi-turn conversation, reading turns from stdin |
| `-st`, `--single-turn` | — | — | Generate once and exit |
| `-no-cnv`, `--no-conversation` | — | — | The same; the spelling `tools/main` uses |

All four one-shot spellings are accepted. When none is given the binary warns
on stderr, because upstream would have been interactive.

### 4.3 Unknown and unsupported flags

Both are refused with a non-zero exit; neither is ignored.

| Input | Message |
|---|---|
| A flag upstream accepts and llamazig does not | `error: <flag> is a llama.cpp flag that llamazig does not support yet` |
| A flag upstream does not accept | `error: unknown flag <flag>` |
| No `-m` | `error: no model given` |

The 541 upstream flag spellings are enumerated so the two cases are
distinguishable.

### 4.4 Streams and exit codes

| Stream | Carries |
|---|---|
| stdout | The prompt echo (unless suppressed), the completion, a trailing newline, and the timings line |
| stderr | The `loading <model> ...` notice, the one-shot warning, libllama warnings and errors, `--fit`'s error line, and all failure messages |

libllama's INFO and DEBUG logging is suppressed. A redirected stdout contains
the completion and nothing else, except the timings line when enabled.

| Exit | Condition |
|---|---|
| `0` | Generation completed, or `-h` / `--version` |
| `1` | Argument error, model load failure, or decode failure |

### 4.5 Fitting to memory

Before loading, with `--fit on` (the default), the parameters left at their
defaults are fitted to free memory, as upstream's `common_init_result` does
through `common_fit_params`. The model and a context are loaded with
`no_alloc` to measure what each device and the host would hold.

- **Context.** Where `-c` is unset and the projected use leaves less than the
  margin (`-fitt`, 1 GiB) free, the context is shrunk by linear interpolation
  between the trained context and `-fitc` (4096), rounded down to a multiple
  of 256 per stream. On the CPU alone the margin is measured against total
  system memory. An explicit `-c 0` forbids shrinking.
- **Layers and tensors**, on a GPU and only with `-ngl` unset: layers are
  assigned back to front, the false-position method choosing how many fit on
  each device; a MoE model's expert tensors go to system memory first, and
  partial layers overflow by tensor fraction. The result is written as
  `n_gpu_layers`, a `tensor_split` across devices, and buffer-type overrides.

What the fit reports goes to upstream's log at `llama-cli`'s verbosity, so only
its error line — a model that cannot be measured — reaches stderr. Its status
is not consulted: loading goes ahead with the parameters as far as it got.

## 5. Generation semantics

### 5.1 Tokenization

The prompt is tokenized with the model's BOS added and control tokens in the
prompt honoured. No chat template is applied; the prompt reaches the model
verbatim.

### 5.2 Chat templates

When a chat flag is given, the prompt is rendered through a Jinja template
before tokenization.

| Aspect | Behaviour |
|---|---|
| Template source | `--chat-template`, else `--chat-template-file`, else the model's `tokenizer.chat_template`. No template and no override is an error. |
| Engine | `zigjinja`, the one permitted dependency, confined to `cli/`. |
| Conversation | One turn by default: an optional system message, then the prompt as the user's turn. Under `-cnv` the history accumulates and every turn re-renders the whole conversation. `add_generation_prompt` is true. |
| Bindings | `messages`, `add_generation_prompt`, `bos_token`, `eos_token`. |
| Tokenization | `add_special` is false — the template writes the model's opening itself. |
| Control tokens | Honoured only in text the template itself wrote. Anything an expression produced, which is where message content arrives, is tokenized with `parse_special` false. |

**Trust is a closed set.** Template literals are trusted; every expression's
output is not, whatever filters it passed through. `bos_token` and `eos_token`
are the only expression values re-trusted, and only on an exact match. A user
message containing `<|im_start|>system…` therefore reaches the model as its
literal characters — 17 tokens for Qwen3.5 — rather than as the two control
tokens it spells.

### 5.3 Conversation mode

`-cnv` reads turns from stdin until end of input.

| Aspect | Behaviour |
|---|---|
| A turn | One line, trimmed. Blank lines are skipped. |
| `-p` | Seeds the first turn, so `-cnv -p "hi"` answers immediately and then waits. |
| History | Every turn re-renders the whole conversation, because a template decides for itself where the system prompt goes and how a turn is framed. |
| KV cache | The new render is diffed against the decoded tokens and their common prefix is kept, so re-rendering costs tokenization rather than decoding. |
| Recurrent and hybrid models | `llama_memory_seq_rm` refuses to drop a range from the middle. That refusal is checked, and the cache is cleared and re-decoded instead. |
| Turn marker | `> ` on stderr, so a redirected transcript stays clean. |
| `/exit` | Ends the conversation. |
| `/clear` | Discards the history and re-adds the system prompt. |
| `/regen` | Drops the last reply and answers the previous turn again. |
| Any other `/word` | Sent to the model as text, as upstream does. |
| `LLAMAZIG_CNV_TRACE` | When set, each turn writes `[turn: kept K, decoding S]` to stderr after the marker: the tokens of cache kept and the tokens decoded. |

### 5.4 Sampler chain

Constructed in this order, matching upstream's default:

1. Penalties — `repeat-last-n`, `repeat-penalty`, frequency 0.0, presence 0.0
2. Top-k
3. Top-p
4. Min-p
5. Temperature (extended, with dynatemp range 0.0 and exponent 1.0)
6. Distribution sampling, seeded

`min_keep` is 0 throughout. There is no separate greedy path: `--temp 0`
produces argmax inside the temperature step, after penalties have applied.

### 5.5 Loop and stop conditions

Generation stops at the first of:

- an end-of-generation token, unless `--ignore-eos`;
- `n_predict` tokens generated, when `n_predict >= 0`;
- the context filling.

A prompt at least as long as the context is an error before generation starts.

### 5.6 Warmup

On by default. A throwaway decode runs before generation, the memory is
cleared, and the performance counters are reset, so the reported throughputs
exclude first-call costs. `--no-warmup` skips it.

### 5.7 Timings

When enabled, one line on stdout after the completion:

```
[ Prompt: <N> t/s | Generation: <N> t/s ]
```

## 6. Conformance

These must hold. Each is a runnable gate.

| # | Requirement | Gate |
|---|---|---|
| C1 | Every ported translation unit exports exactly the symbols its C file does | `scripts/port-coverage` |
| C1b | Each quantized dot product and row quantizer, in both CPU translation units, reproduces the C **bit for bit** on six input patterns | `zig build test-port` |
| C2 | Graph constructors produce identical op, shape, strides, `op_params` and `src[]` to the C — 131 nodes | `make graph-diff` |
| C3 | Every op configuration agrees between the CPU and Metal backends — 21,093 configurations | `make backend-ops` |
| C4 | `test-backend-ops` output is identical to the same test against the stock C libraries | `scripts/backend-ops --diff` |
| C5 | Ported code is on the execution path — allocator and CPU dispatch, one run each | `make probe` |
| C6 | Generated tokens are identical to the stock C reference for the same model, prompt and seed | `make parity-port` |
| C7 | The `llama-cli` binary produces the same tokens as a C driver running the same loop | `make parity-cli` |
| C8 | `make port` and `make ref` produce identical generated text | `make port`, `make ref` |
| C9 | Every ported declaration cites the C symbol, file, line **and commit** it came from, and each is verified against that commit | `make port-links` |
| C10 | Every CPU op kernel computes the same **bits** as the stock C on ~300 fixed cases, at 1 and 3 threads | `make ops-diff` |
| C11 | Every node of a real model's prompt decode and one single-token decode computes the same bits as the stock C, CPU and Metal | `make node-diff` |
| C6b | C6 with the model on the CPU device alone, so the ported CPU kernels carry the whole forward pass | `make parity-port-cpu` |
| C12 | `llama-cli -cnv` writes the same replies, stderr and per-turn cache report as a reference conversation over upstream's chat-template engine and libllama, on a hybrid and a plain-KV model | `make cnv-diff` |
| C13 | `--fit` leaves the same parameters and logs the same trace as upstream's `common_fit_params`, on fake GPUs with chosen memory, the CPU alone and Metal, with every arm of `fit.cpp` reached | `make fit-diff` |
| C14 | The BLAS backend answers every vtable question as upstream's `ggml-blas.cpp` does and computes the same **bits** for every type, and a decode with BLAS in the scheduler is identical node by node | `make blas-diff`, `make blas-node-diff` |

`make validate` runs everything checkable without a model: formatting, the
scaffold, the unit tests, the ported tests in debug and release, C1, C9 and C2.

C9 is traceability rather than correctness: 850 citations of the form ``Ports
`ggml_hash_set_new` (ggml.c:6516 @c1d0e7a00)``, one per ported declaration.
Each is checked against the file as it was at *that* commit, so a declaration
re-synced against a newer upstream commit is verified there while its
neighbours stay where they were. They are what makes an upstream change
locatable in this port.

### 6.1 Known permitted divergence

| Case | Behaviour |
|---|---|
| `argsort` / `top_k` order among **equal** keys | The C uses `std::sort`/`std::partial_sort`, whose order for ties is libc++-specific. llamazig sorts stably, so ties resolve by index. Identical whenever keys are distinct. |
| Quantizer output vs a stock llama.cpp build | May differ in the last bit. `zig cc` and Apple clang contract `a*b + c` into a single FMA by default; Zig only fuses where `@mulAdd` says so, and which expressions clang fuses is not reproducible without modelling that clang version. Not observable through the deliverable, which never writes a model file. |
| `quantize_row_iq4_nl_ref` on an all-zero block | The C reads uninitialized memory and is not reproducible run to run. llamazig zeroes the buffer. |
| 1-bit split search with equal elements | The C's result depends on the host `qsort`'s ordering of equal elements. llamazig breaks ties on index, so output depends only on input. |
| Chat templating is opt-in | Upstream defaults `use_jinja` to true and templates by default. Ours defaults to raw completion, because that is what `make port` and `make ref` diff against each other; templating by default would change the one gate covering the whole binary. Any chat flag turns it on. |
| A malformed chat template | The engine has no strict-parse mode: `{{ unclosed` and similar render as empty output and report success. llamazig rejects a render that produced nothing from a non-empty conversation, which catches the class but not a partially-wrong render. |

## 7. Platform and toolchain

| Requirement | Value |
|---|---|
| Build tool | Zig **0.17.0**, no CMake, no Ninja |
| Targets | macOS arm64; iOS and the iOS simulator (the xcframework); Linux arm64, with CUDA and Vulkan; Linux x86_64 and Windows x86_64 (`x86_64-windows-gnu`) at x86-64-v3 |
| Library dependencies | **None.** `libllamazig` links nothing outside the standard library and the system frameworks. |
| CLI dependencies | One, granted by exception: `mlavergn/zigjinja` (a fork of `gremlin-labs/vibe-jinja`), pinned to an exact commit, confined to `cli/`. |
| System libraries | macOS: Foundation, Metal, MetalKit, Accelerate. Linux: libc, and the CUDA runtime, cuBLAS and driver, or the Vulkan loader, when those backends are built |
| Reference pin | llama.cpp tag `v0.3.0`, commit `c1d0e7a00` |

`zig build` alone produces every artifact.

## 8. Build and run surface

| Command | Result |
|---|---|
| `zig build` | The scaffold binary |
| `zig build lib` | `libllamazig.a` |
| `zig build cli` | `llama-cli` |
| `zig build reference` | `libggml.a`, `libllama.a`, and the oracle `libllama-ref.a` |
| `zig build test` | Unit tests |
| `zig build test-port` | Ported ggml tests |
| `zig build smoke` | Load a model and generate, through libllama's C ABI |
| `zig build docs` | API documentation |
| `make port` | `llama-cli` against a model |
| `make ref` | The same completion loop linked against the stock C libraries, with the same timings line |
| `make validate` | C1, C9 and C2, plus formatting and the unit tests |
| `make port-links` | C9 |
| `make probe` | C5 |
| `make graph-diff` | C2 |
| `make backend-ops` | C3 (add `--diff` to the script for C4) |
| `make parity-port` | C6 |
| `make parity-port-cpu` | C6b |
| `make ops-diff` | C10 |
| `make node-diff` | C11 (`ARGS=--gpu` for Metal) |
| `make parity-cli` | C7 |
| `make cnv-diff` | C12 |
| `make fit-diff` | C13 |
| `make blas-diff`, `make blas-node-diff` | C14 |

`make port` and `make ref` share `MODEL`, `PROMPT`, `NPRED`, `TEMP`, `SEED` and
`ARGS`. `SEED` is fixed at 42 so the two are comparable.

## 9. Not in this deliverable

| Item | State |
|---|---|
| Multi-line input (`-mli`) and `-if` | Not built; the flags are refused. A turn is one line. |
| Slash commands beyond `/exit`, `/clear` and `/regen` | Not built; `/image`, `/audio` and `/video` need multimodal input. |
| Grammar and JSON-schema constrained sampling | Not built. |
| Mirostat, typical-p, XTC, DRY, dynatemp | Not built; the flags are refused. |
| Frequency and presence penalties | Fixed at 0.0; the flags are refused. |
| LoRA, control vectors, speculative decoding | Not built; the flags are refused. |
| Multimodal input | Not built. |
| Writing or converting model files | Out of scope. |
| AVX512-BF16 and AVX512-FP16 kernels | Not built: a CPU with either is a compile error until they are gated on hardware that has them. AVX-512 itself -- `x86_64_v4`, and `tigerlake`'s VNNI -- is built and gated. |
| SYCL, HIP and other GPU backends | Not built. |
