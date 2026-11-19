# HISTORY.md

Releases and major changes, newest first. One entry per upstream sync (`SYNC.md`, Step 7):

- heading: port version, date, upstream tag and commit, port commit
- **Changed:** major changes, one line each
- **Found:** bugs and gate holes it turned up, one line each, with the gate
- **Open:** anything left for a later sync

The detail behind any line is in that commit's diff and message.

---

## 1.0.0 -- 2026-11-17 -- upstream v1.0.0 (`b1b770d90`, `b12359`) -- `e5205e4`

Driven from the GB10: every Linux gate ran before the commit; the Mac's run as its pickup.

- **Changed:**
  - 2,016 touched citations, 966 unmapped hunks, 72 uncited files; 9,902 citation sites repinned, 707 line literals moved
  - the prune: ggml 102 -> 99 ops, `llama.h` 255 -> 237 exports, 13 `llama-ext.h` functions moved into it
  - the mtmd refactor: `clip.cpp` split into `mtmd-model`/`-subcontext`/`-graph`; 445 citations followed by `harness/upstream_moved.zig`
  - 3 architectures (`kimi-k31`, `ricursive`, `xing4-0`: 171), 2 projector types (`magevl`, `drama3-gen`)
  - F32 `src1` on the CPU (`ggml_vec_dot_f16_f32`), YaRN `truncate`, per-model RoPE
  - CUDA OCP FP8 and few-row MMQ; Intel's Vulkan FA prefill
  - MTP hidden states from a device buffer; the MoE cache's ClassLRU with CPU misses; external PLE files
  - TTS text normalizer (`tts_norm.zig`), `/v1/audio/speech`, the server's SSRF and CORS fixes, cpp-httplib 0.66
  - zigjinja `6efb5e4` (v3.1.11)
- **Open:** Mac and DESKTOP-BR7J5S9 pickups (`SYNC.md`); Metal ported blind.

## 0.9.0 -- 2026-11-10 -- upstream v0.9.0 (`cd959bae9`, `b12110`) -- `ff4ce0c`

- **Changed:**
  - 584 touched citations, 278 unmapped hunks, 43 uncited files; 11,020 citation sites repinned
  - 4 architectures (`clover_moe`, `limite`, `metis16`, `mistral_large4`: 168), 3 projector types (`nemotron_transcribe`, `chatterbox`, `chatterbox_spkenc`)
  - `GGML_TYPE_F8_E4M3`, `ggml_flash_attn_ext_rows`, batched `ggml_backend_tensor_{set,get,copy}_n`
  - Vulkan all-reduce (`ggml-vulkan-comm.cpp`), CUDA `batch-get.cu`
  - cross-layer KV sharing; layer-input device buffer (`llama-ext.h` 28 -> 31); voice API (`mtmd_*` 107 -> 112)
  - miniaudio 0.11.26, cpp-httplib 0.64
  - **Apple minimum OS 26.0 on every platform**
- **Found:**
  - a libmtmd write-after-free
  - NEON `vmaxq_f32` is `FMAX` (propagates NaN), Zig's `@max` is `FMAXNM`: a q8_0 block with a NaN got a different scale (GB10, a T5 fixture)
  - DeepSeek V4 / Qwen4-Exp embeddings: reported upstream (#32036)
- **Pickups:** GB10 clear (`32ca8b6`), DESKTOP-BR7J5S9 clear (`f08a6a0`).

## 0.8.0 -- 2026-11-09 -- upstream v0.8.0 (`8642541b2`, `b11960`) -- `cfd9292`

- **Changed:**
  - 793 touched citations, 784 unmapped hunks (`upstream-map` now covers `tools/server` and cpp-httplib); 10,066 sites repinned; `upstream-literals` added
  - 7 architectures (`lfm3`, `beam`, `flint`, `kolibri1`, `muse-spark`, `sensenova-u1`, `deepseek41`: 164), 9 projector types
  - `GGML_OP_TOP_K_MASK`, `-1` expert ids, the MoE cache's segmented LRU (`moe-slru-diff`), `no_alloc` on the memories
  - streaming audio API (`mtmd_*` 84 -> 107), 4 pre-tokenizers
  - `/v1/realtime`, `/tts`, the router's memory scheduler and `--connect`, cpp-httplib 0.62 with WebSockets
  - `harmony_xml` and `muse_spark` chat parsers
  - `metal-ops-diff`: Metal op encoders on bits against the reference
- **Found:**
  - a libllama throw is a Zig error where upstream catches it: the server aborted on a bad `/detokenize` id
  - five harness faults on the GB10; `cuda-ops-diff`/`vk-ops-diff` extras made to follow `-o` lists
- **Pickups:** GB10 clear (`t5-dense` CUDA-offload `kq-1` left unattributed); DESKTOP-BR7J5S9 clear.

## 0.7.0 -- 2026-11-04 -- upstream v0.7.0 (`620d2225c`, `b11669`) -- `0acdb2c`

- **Changed:**
  - 770 touched citations, 336 unmapped hunks, 27 uncited files; 9,450 sites repinned
  - 4 architectures (157), 5 projector types (68)
  - the MoE cache, `prism.hadamard`, the scheduler's copy and realloc callbacks
  - PLaMo-3 tokenizer, Unigram NFC
  - the chat API refactor (`common_chat_prompt` parts), two chat parsers; `bos_token`/`eos_token` became plain template text
  - cpp-httplib 0.61.0, slot checkpoints and draft state, five new decision-API types
  - paths with a space fixed in the `Makefile`s and `xcframework/`
- **Found:** `decision_confidence_score`'s vectorization group (GB10).
- **Pickups:** GB10 clear; DESKTOP-BR7J5S9 clear, `llama-cli.exe` and the tools building for Windows again (`675d18a`). JWHLPTOP excluded from here on.

## 0.6.0 -- 2026-11-03 -- upstream v0.6.0 (`d81235049`, `b11429`) -- `38978d0`

- **Changed:**
  - 760 touched citations, 413 unmapped hunks, 42 uncited files; 9,584 sites repinned
  - 2 architectures (`clef`, `glm5-next`), 2 projector types (`ling3vl`, `glm5v`)
  - 14 `llama.h` functions (`llama_batch_ext`, `llama_process`, `llama_get_causal_attn`)
  - `ggml_backend_buft_alloc_buffer_n`/`get_alloc_size_n`; `ggml-cpu/tiled` replacing `iqp`
  - Metal MMA mat-vec and FA tensor path; CUDA `CONV_3D` and shared-expert fusion; Vulkan coopmat1 integer-dot mmq
  - `/v1/systemone` decision API, `llm-jp-harmony` chat parser
- **Found:** one CUDA and one Linux build fix (GB10); the tiled kernels' first gate (Windows).

## 0.5.0 -- 2026-11-02 -- upstream v0.5.0 (`7fe450e19`, `b11146`) -- `e9c4b5b`

- **Changed:**
  - 762 touched citations (461 from Vulkan's file split), 246 unmapped hunks, 24 uncited files; 9,560 sites repinned
  - 1 architecture (`hrm_text`), 1 `llama.h` function, `ggml_dsv4_hc_pre_gated`
  - q1_0 repack; Metal tuning keyed by GPU family; four Metal fusions; the `ling3` chat parser
- **Found:** `impl_c.zig` missed five new `#define`s until `struct-layout` diffed the header's names.

## 0.4.1 -- 2026-10-29 -- upstream v0.4.1 (`b29c606e2`, `b10964`) -- `d4f0a16`

- **Changed:**
  - 197 touched citations, 168 unmapped hunks, 43 uncited files; 7,231 sites repinned
  - 3 architectures (`hy_v4`, `maple`, `spark2_5`)
  - the Metal fusion table (`ggml-metal-fusion.cpp` -> `metal/fusion.zig`), the Vulkan matmul refactor
- **Found:** `diag_f16` (GB10); three gate fixes.

### Between 0.4.1 and 0.5.0

- **Stage 9, chat parsing** (`1862287`, `216f26f`): `common_json`, JSON schema to grammar, the PEG parser, templates and the per-model parsers, all under `cli/`.
- **Stage 10, `llama-server`** (`216f26f`): the server, cpp-httplib, the web UI, router mode and the downloader, under `tools/server/`.
- Found: a use-after-free in zigjinja's `loop.previtem`, seen only in ReleaseSafe (`chat-parse-diff`); nothing in `tools/server/` compiled for Linux (`std.c.Stat`); every native Linux Zig binary took `compiler_rt`'s `expf` (fixed by linking `libgcc.a` first).

## 0.4.0 -- 2026-10-28 -- upstream v0.4.0 (`5266f24da`, `b10809`) -- `a67ea88`

The first sync, and the sync tooling's first real run.

- **Changed:** 456 touched citations, 217 unmapped hunks, 29 uncited files; 7,601 citations repinned, 529 line literals moved.
- **Found:** `gen-hparams` dropped every member after a mid-struct member function; `gen-model-tables` stopped on a conditional RoPE arm.

## 0.3.0 -- 2026-10-27 -- upstream v0.3.0 (`c1d0e7a00`, `b10621`) -- `bd498fc`

The port as built before syncing began, pinned at v0.3.0 throughout. Versions `0.0.2`-`0.0.44`.

- **Stages 1-3** (2026-09-01 to 09-02): `build.zig` replaces CMake; every C translation unit in ggml ported to Zig.
- **Chat and conversation** (09-08): chat templates through zigjinja, `-cnv`.
- **Stage 4, ggml's C++** (09-30 to 10-06): `gguf`, the backend and scheduler, `ops.cpp`, sgemm, the repack vtable cluster, all of `ggml-metal/` but the two `.m` files.
- **libllama** (10-07 to 10-16): sampler, grammar, unicode, vocabulary, then everything, swapped once at `llama.h`; every architecture in `llama_model_mapping`.
- **Zig 0.17** (10-13), with Zig's bundled libc++.
- **Stage 6, CUDA** (10-16): host layer in Zig over unmodified fatbins, on the GB10.
- **Stage 5, cross-platform** (10-13 to 10-25): Linux arm64 bit-exact at `generic`; i8mm; the SIMD layer; x86_64 at AVX2 and AVX-512; Windows x86_64.
- **Stage 7, Vulkan** (10-17 to 10-24): host layer in Zig, embedded SPIR-V; run on the UNO Q's Adreno 702.
- **`--fit`** (10-24), **BLAS over Accelerate** (10-25).
- **Stage 8** (10-26): libmtmd, `llama-tts`, `llama-mtmd-cli`; upstream's ten smaller tools.
- **Apple** (10-13 to 10-26): `llama.xcframework`, a drop-in for upstream's; run on an iPhone 15.
- **Found:**
  - a real `ggml_permute` bug (`graph-diff`)
  - Zig miscompiles `p.*.arr[i]` through a `[*c]` pointer: scheduler copies took their source's `op` (`node-diff`)
  - Zig miscompiles small `extern struct`s received by value: `ggml_bf16_to_fp32` returned 0.0 for every input, past nine gates (`abi-check`)
  - the `q5_K`, `tq1_0`, `tq2_0` dot-product epilogues, which had passed their goldens (`node-diff`)
  - `GGML_USE_CPU_REPACK` was off, so `repack` was dead code; four faults in its kernels (`repack-diff`)
  - typical's unrolled entropy loop and libc++'s arithmetic heap order (`sampler-diff`)
  - `output_resolve_row`'s `uint32_t` wrap; `tensor_split` left pointing into a dead stack frame (`context-diff`)
  - a leaf-tensor narrowing in `supports_op` that only a safe build saw (`parity-cli`)
  - `compiler_rt` shadowing glibc's `expf` through a `@panic`; `char` unsigned on aarch64 Linux (GB10)
  - the portable `dotq_s32` cost the CPU path most of its speed (0.21x the C); `sdot` restored it (`perf-cpu`)
  - chat templates had marked the complement of the input, letting control tokens through as text (`jinja-diff`)
  - `port-links` had silently skipped wrapped citations and `.cu` files
  - the vocabulary's storage grew ~7 MB per model reload (iPhone)
