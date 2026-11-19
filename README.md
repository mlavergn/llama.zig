# llamazig

A port of [llama.cpp](https://github.com/ggml-org/llama.cpp) to Zig, kept in sync with upstream one release at a time. Pure Zig apart from the GPU kernels and Metal's two Objective-C files.

- Synced to upstream **v1.0.0** (`b1b770d90`); `build/upstream.zig` holds the pin.
- `SYNC.md` -- how it follows upstream. `HISTORY.md` -- what each release changed. `CLAUDE.md` -- guidance for agents.

## What it builds

| Artifact | What |
|---|---|
| `llama-cli` | Completion and conversation, flag-compatible with upstream's for what it supports |
| `llama-server` | Upstream's server, ported (`tools/server/`) |
| `llama-tts`, `llama-mtmd-cli` | Text-to-speech and multimodal CLIs over libmtmd |
| `gguf-split`, `quantize`, `tokenize`, `fit-params`, `export-lora`, `batched-bench`, `cvector-generator`, `imatrix`, `perplexity`, `llama-bench` | Upstream's tools, same command lines and output |
| `libggml.a` | ggml: CPU, Metal, CUDA, Vulkan, BLAS (Apple) |
| `libllama.a` | libllama, exporting `llama.h` (237 symbols, equal to upstream's); every architecture upstream maps (171) |
| libmtmd | `mtmd.h`/`mtmd-helper.h` (115 functions), every projector type |
| `llama.xcframework` | Drop-in for upstream's `build-xcframework.sh` output ([`xcframework/README.md`](xcframework/README.md)) |
| `libllama-ref.a` | Upstream's libllama, unmodified: a test oracle, not a deliverable |

What is not Zig: `ggml-metal-device.m` and `ggml-metal-context.m` (Objective-C), the Metal kernels (MSL, embedded), the CUDA kernels (unmodified `.cu` compiled to embedded fatbins, plus `src/ggml/cuda/cub_shim.cu`), the Vulkan shaders (the reference build's SPIR-V, embedded), and three C++ objects empty on every target (`hbm`, `amx`).

## Platforms

| Target | Backends |
|---|---|
| macOS arm64 | Metal, CPU (Accelerate, llamafile SGEMM), optional BLAS (`-Dblas`) |
| iOS, tvOS, visionOS (devices and simulators) | Through `llama.xcframework`, minimum OS 26.0 |
| Linux arm64 | CPU; CUDA (`-Dcuda`, when `nvcc` is found); Vulkan (`-Dvulkan`) |
| Linux x86_64, Windows x86_64 (`x86_64-windows-gnu`) | CPU at `x86_64_v3` (AVX2) or `x86_64_v4`/`tigerlake` (AVX-512). No `llama-server` on Windows |

- **Zig 0.17.0**; `zig build` alone builds everything. No CMake.
- **No library dependencies.** `llama-cli` and `llama-server` use one: [`zigjinja`](https://github.com/mlavergn/zigjinja), for chat templates.
- System libraries: macOS Foundation, Metal, MetalKit, Accelerate; Linux libc, plus the CUDA runtime, cuBLAS and driver, or the Vulkan loader, when those backends are built.

## Build and run

```sh
zig build cli --release=fast        # -> zig-out/bin/llama-cli
zig build reference                 # -> libggml.a, libllama.a, libllama-ref.a
zig build test                      # unit tests
make qwen35                         # fetch Qwen3.5-2B Q4_K_M
make port PROMPT="def fibonacci(n):" TEMP=0
```

`make port` and `make ref` run the same completion loop over the port and over stock upstream; their output is identical. Both take `MODEL`, `PROMPT`, `NPRED`, `TEMP`, `SEED` (fixed at 42) and `ARGS`.

## `llama-cli`

```
llama-cli -m PATH [options]
```

Generates from `-p` and exits, or under `-cnv` holds a conversation on stdin. Flag names, aliases, arguments and defaults match upstream's, so an upstream command line runs unchanged for these:

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help`, `--usage` / `--version` | | Print and exit 0 |
| `-m`, `--model PATH` | required | Model file |
| `-p`, `--prompt TEXT` | `""` | Prompt |
| `-n`, `--predict`, `--n-predict N` | `-1` | Tokens to generate; `-1` until stop |
| `-c`, `--ctx-size N` | `0` | Context; `0` = the model's, which `--fit` may shrink (not when `-c 0` is explicit) |
| `-b`, `--batch-size N` | `2048` | Logical batch |
| `-ngl`, `--gpu-layers`, `--n-gpu-layers N` | `-1` | Layers offloaded; `-1` auto |
| `-fit`, `--fit on\|off` | `on` | Fit unset parameters to free memory |
| `-fitt`, `--fit-target MiB,...` | `1024` | Memory left free per device |
| `-fitc`, `--fit-ctx N` | `4096` | Least context `--fit` shrinks to |
| `-t`, `--threads N` | `0` | `0` = libllama's choice |
| `-s`, `--seed N` | random | RNG seed |
| `--temp`, `--temperature F` | `0.80` | `<= 0` is greedy |
| `--top-k N` / `--top-p F` / `--min-p F` | `40` / `0.95` / `0.05` | |
| `--repeat-last-n N` / `--repeat-penalty F` | `64` / `1.00` | |
| `--ignore-eos` | off | Never stop at end-of-generation |
| `-e`, `--escape` / `--no-escape` | on | `\n`, `\t`, `\\` in the prompt |
| `--no-warmup`, `--no-display-prompt` | | |
| `--show-timings` / `--no-show-timings` | on | Throughput line on stdout |
| `--jinja` / `--no-jinja` | off | Render through the chat template |
| `--chat-template TEXT`, `--chat-template-file F` | | Template instead of the model's |
| `-sys`, `--system-prompt TEXT`, `-sysf F` | | System message |
| `-cnv`, `--conversation`, `-i`, `--interactive` | off | Conversation on stdin |
| `-st`, `--single-turn`, `-no-cnv`, `--no-conversation` | | Generate once (the default; a warning says so) |
| `--grammar G`, `--grammar-file F`, `-j`, `--json-schema S`, `-jf F` | | Constrain generation; the last given wins |
| `--reasoning-format FMT` | `deepseek` | `none` leaves thinking in the answer |
| `-rea`, `--reasoning on\|off\|auto` | `auto` | Thinking; `on`/`off` also set `enable_thinking` |
| `--reasoning-budget N` / `--reasoning-budget-message M` | `-1` / `""` | Thinking token budget |
| `--reasoning-effort LEVEL` | `default` | The `reasoning_effort` kwarg |
| `--reasoning-preserve` / `--no-reasoning-preserve` | on | The `preserve_reasoning` kwarg |
| `--skip-chat-parsing` | off | Read replies as plain answer |
| `--chat-template-kwargs JSON` | | Extra template variables |
| `-co`, `--color on\|off\|auto` | `auto` | Grey reasoning on a terminal |

- **Chat is opt-in** (upstream templates by default): any chat flag turns it on; `--no-jinja` vetoes.
- **Refused, never ignored**, exit 1:
  - an upstream flag not supported: `error: <flag> is a llama.cpp flag that llamazig does not support yet`
  - anything else: `error: unknown flag <flag>`
  - no model: `error: no model given`
  - a bad value: `error while handling argument "<flag>": <upstream's message>`; a grammar that does not parse: `error: failed to parse grammar`
- **stdout:** the prompt echo, the completion, the timings line; in a chat each reply with `[Start thinking]`/`[End thinking]`. A redirected stdout holds nothing else.
- **stderr:** `loading <model> ...`, warnings, errors, `> ` turn markers. libllama's INFO/DEBUG log is silenced.
- **Exit:** `0` done; `1` argument, load or decode failure.
- **`--fit`** (as upstream's `common_fit_params`): with `-c` unset, shrinks the context toward `-fitc` until the margin is free; with `-ngl` unset on a GPU, places layers back to front, MoE experts in system memory first.

Not supported: `-mli`, `-if`; slash commands beyond `/exit`, `/clear`, `/regen`; mirostat, typical-p, XTC, DRY, dynatemp; frequency and presence penalties; LoRA, control vectors, speculative decoding; loading flags (`-lzm`, `-ot`, `--cpu-moe`, ...); `-hf` downloads; media input (use `llama-mtmd-cli`).

## Correctness

Every component is checked against stock upstream built by the same Zig toolchain (`llama.cpp.zmake`), **on bits**: every graph node of real decodes, every CPU op and kernel, every op encoder, the sampler, tokenizer, loader, chat stack, server and tools. Each gate is negative-tested by injecting a fault. `CLAUDE.md` lists them.

Permitted differences:

- Chat templating is opt-in.
- Message text never becomes control tokens: input parts tokenize with `parse_special` off, where upstream's server tokenizes the whole render with it on.
- Quantizer output may differ from a stock build in the last bit (FMA contraction). The CLI never writes a model.
- `quantize_row_iq4_nl_ref` on an all-zero block: upstream reads uninitialized memory; ours zeroes.
- 1-bit split search: ties broken by index, not by the host `qsort`.

## Known gaps

| Item | Waiting for |
|---|---|
| AVX512-BF16 / AVX512-FP16 kernels | A Sapphire Rapids or newer CPU (a compile error until then) |
| Linux x86_64's OS arms (threading, affinity, NUMA) | Linux x86 hardware; the kernels are gated on Windows |
| `set_numa_thread_affinity` | A multi-socket machine |
| The xcframework's x86_64 slices | An Intel Mac or Rosetta; built and checked, never run |
| The xcframework on tvOS and visionOS devices | Running the smoke app there |
| Swapping in [zigggml](../zigggml) | A zigggml whose A/B gates pass |

## Licence

MIT; see [`LICENSE`](LICENSE). A derivative work of llama.cpp (MIT, the ggml authors), whose copyright `LICENSE` carries. Every ported file names the upstream file and commit it translates.
