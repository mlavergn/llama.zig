# UPSTREAM.md

Where the port stands in upstream's history, and how it got there. The plan
behind it is `MAINTENANCE.md`; the procedure a sync follows is `SYNC.md`.

## Basis

```
basis: v0.9.0 cd959bae99ff4bed22bef2604f45241c09bbb8c8 b12110
```

| | |
|---|---|
| Upstream | `https://github.com/ggml-org/llama.cpp` (the submodule's `origin`, and `.gitmodules`) |
| Tag | `v0.9.0`, released 2026-11-04 |
| Commit | `cd959bae99ff4bed22bef2604f45241c09bbb8c8`; citations carry `@cd959bae9` |
| Build tag | `b12110` |
| ggml version | `0.29.0` (ggml/CMakeLists.txt), which `ggml_version()` returns |
| Recorded in | `build/upstream.zig`, `llama.cpp.zmake/build.zig`, `src/ggml/runtime.zig`, `src/llama/api/llama.zig`; `scripts/check-reference-pin` holds them to the submodule |
| zigjinja | `5b2472e` (v3.1.9; not yet on its remote, carried to the GB10 in a bundle), a path dependency (`../zigjinja`, zigbuildx's `.clone`); the pickups fast-forward their checkout to it |

The `basis:` line is the machine-readable one: tag, full commit, build tag.

## History

| From | To | Date | Port version | Port commit | Work | Pickups |
|---|---|---|---|---|---|---|
| — | `v0.3.0` (`c1d0e7a00`, `b10621`) | 2026-10-27 | `0.3.0` | `bd498fc` | the port as built through Stages 1–8 | — |
| `v0.3.0` | `v0.4.0` (`5266f24da`, `b10809`) | 2026-10-28 | `0.4.0` | `a67ea88` | 456 touched citations, 217 unmapped hunks, 29 uncited files; 7,601 citations repinned; 529 line literals moved; Mac gates all pass (`NOTES.md`, "Upstream sync v0.3.0 → v0.4.0") | GB10: clear (`7c9cbc1`; `lint` not run, no `zlint` there) · DESKTOP-BR7J5S9: clear (`7c9cbc1`; `quant-diff` has no Windows mode) · JWHLPTOP: **open**, unreachable 2026-10-28 |
| `v0.4.0` | `v0.4.1` (`b29c606e2`, `b10964`) | 2026-10-29 | `0.4.1` | `d4f0a16` | 197 touched citations, 168 unmapped hunks, 43 uncited files; 7,231 citation sites repinned; 316 line literals moved; three architectures (hy_v4, maple, spark2_5), the Metal fusion table, the Vulkan matmul refactor; Mac gates (`NOTES.md`, "Upstream sync v0.4.0 → v0.4.1") | GB10: clear (`0.4.1+1`; three gate fixes, `diag_f16` found) · DESKTOP-BR7J5S9: clear (`0.4.1+1`) · JWHLPTOP: **open** (with v0.4.0's AVX-512 items), unreachable 2026-10-29 |
| `v0.4.1` | `v0.5.0` (`7fe450e19`, `b11146`) | 2026-11-01 | `0.5.0` | `e9c4b5b` | 762 touched citations (461 of them Vulkan's file split), 246 unmapped hunks, 24 uncited files; 9,560 citation sites repinned; 388 line literals moved; one architecture (`hrm_text`), one `llama.h` function, one op form (`ggml_dsv4_hc_pre_gated`), the q1_0 repack, the Metal tuning rewrite and four fusions, the Vulkan split, the `ling3` chat parser; Mac gates (`NOTES.md`, "Upstream sync v0.4.1 → v0.5.0") | GB10: clear (`NOTES.md`, "The v0.5.0 pickup") · DESKTOP-BR7J5S9: clear, apart from `http-diff` and `tools-diff server`, which wait for the server's Windows port · JWHLPTOP: **open**, unreachable (Marc, 2026-11-01), carrying v0.4.0's and v0.4.1's AVX-512 items |
| `v0.5.0` | `v0.6.0` (`d81235049`, `b11429`) | 2026-11-03 | `0.6.0` | `38978d0` | 760 touched citations, 413 unmapped hunks, 42 uncited files (after the `upstream-map` extent fix; 757/416 before); 9,584 citation sites repinned; 599 line literals moved by `upstream-relines` and the rest by hand; two architectures (`clef`, `glm5-next`), two projector types (`ling3vl`, `glm5v`), 14 `llama.h` functions (`llama_batch_ext`, `llama_process`, `llama_get_causal_attn`), `ggml_backend_buft_alloc_buffer_n`/`get_alloc_size_n`, `ggml-cpu/tiled` replacing `iqp`, the Metal MMA mat-vec and FA tensor path, the CUDA `CONV_3D` path and shared-expert fusion, the Vulkan coopmat1 integer-dot mmq, the `/v1/systemone` decision API, the `llm-jp-harmony` chat parser; Mac gates (`NOTES.md`, "Upstream sync v0.5.0 → v0.6.0") | GB10: clear (`0.6.0+1`; one CUDA and one Linux build fix; open: `mtmd-diff`'s two `qwen3tts` CPU passes over upstream's uninitialised `out_code_cache`, and `server-diff decision`'s parallel block, upstream against itself) · DESKTOP-BR7J5S9: clear (`0.6.0+1`; the tiled kernels' first gate), apart from the server's Windows port · JWHLPTOP: **open**, no longer paired, carrying v0.4.0's through v0.6.0's AVX-512 items |
| `v0.6.0` | `v0.7.0` (`620d2225c`, `b11669`) | 2026-11-04 | `0.7.0` | `0acdb2c` | 770 touched citations, 336 unmapped hunks, 27 uncited files; 9,450 citation sites repinned; 772 line literals moved by `upstream-relines` and 156 more by hand (its 22, and 134 in classes it misses); four architectures (`k2-horizon`, `maion-coder`, `gemma-embedding2`, `vllm-decision2`: 157), five projector types (68), the MoE cache, `prism.hadamard`, the scheduler's copy and realloc callbacks, the PLaMo-3 tokenizer and Unigram NFC, the chat API refactor and two chat parsers, cpp-httplib 0.61.0, slot checkpoints and draft state, the decision API's five new types; `llama.h` 255 exports unchanged; paths with a space fixed in the `Makefile`s and `xcframework/`; Mac gates (`NOTES.md`, "Upstream sync v0.6.0 → v0.7.0") | GB10: clear (`0.7.0+1`; CUDA and Vulkan before the commit, the rest after; one port bug fixed, `decision_confidence_score`'s vectorization group; open: `mtmd-diff`'s two `qwen3tts` CPU passes, and `server-diff` `sleep`, `parallel` and `decision`'s parallel block, each of which differs upstream against itself there) · DESKTOP-BR7J5S9: clear (x86 gates on silicon, `win-host-diffs`, the 1M-deep `json-diff`; `llama-cli` and the tools build again, `llama-cli.exe` token-identical on 6 prompts) · JWHLPTOP: **excluded until further notice** (Marc, 2026-11-04: no access), carrying v0.4.0's through v0.7.0's AVX-512 items |
| `v0.7.0` | `v0.8.0` (`8642541b2`, `b11960`) | 2026-11-08 | `0.8.0` | `cfd9292` | 793 touched citations, 784 unmapped hunks (426 before `upstream-map` took in `tools/server` and cpp-httplib); 10,066 citation sites repinned; 691 line literals moved by `upstream-relines` and 44 more found by the new `upstream-literals`; seven architectures (`lfm3`, `beam`, `flint`, `kolibri1`, `muse-spark`, `sensenova-u1`, `deepseek41`: 164), nine projector types, `GGML_OP_TOP_K_MASK`, `-1` expert ids, the MoE cache's segmented LRU, `no_alloc` on the memories, the streaming audio API (`mtmd_*` 84 → 107), four pre-tokenizers, `/v1/realtime`, `/tts`, the router's memory scheduler and `--connect`, cpp-httplib 0.62 with WebSockets, the `harmony_xml` and `muse_spark` chat parsers; `llama.h` 255 exports unchanged; a libllama throw is a Zig error where upstream catches it (the server aborted on a bad `/detokenize` id); Mac gates (`NOTES.md`, "Upstream sync v0.7.0 → v0.8.0") | GB10: clear but for what does not reproduce against itself there (2026-11-09: CUDA, Vulkan on all four arms, the Linux CPU set, models, libmtmd, chat, tools and server; five harness faults fixed; `t5-dense`'s CUDA-offload `kq-1` open, unattributed; the 26 CUDA and Vulkan injections 23 caught, 2 inert, 1 unreachable on its driver, after `cuda-ops-diff`'s and `vk-ops-diff`'s extras were made to follow `-o` lists and given four new shapes) · DESKTOP-BR7J5S9: clear (`x86-gates` at `x86_64_v3`, `win-host-diffs`; `llama-cli.exe` and the tools build) · JWHLPTOP: **excluded**, carrying v0.4.0's through v0.8.0's AVX-512 items |
| `v0.8.0` | `v0.9.0` (`cd959bae9`, `b12110`) | 2026-11-10 | `0.9.0` | — | 584 touched citations, 278 unmapped hunks, 43 uncited files; 11,020 citation sites repinned; 594 line literals moved by `upstream-relines` and the rest by hand (ten in `kv_cache_dsv4.zig` through a line-level diff); four architectures (`clover_moe`, `limite`, `metis16`, `mistral_large4`: 168), three projector types (`nemotron_transcribe`, `chatterbox`, `chatterbox_spkenc`), `GGML_TYPE_F8_E4M3`, `ggml_flash_attn_ext_rows`, the batched `ggml_backend_tensor_{set,get,copy}_n` and their buffer slots, the Vulkan all-reduce (`ggml-vulkan-comm.cpp`), CUDA `batch-get.cu`, cross-layer KV sharing, the layer-input device buffer (llama-ext.h 28 → 31), the voice API (`mtmd_*` 107 → 112), miniaudio 0.11.26, cpp-httplib 0.64; `llama.h` 255 exports unchanged; Apple minimum OS 26.0 (Decision 66); a libmtmd write-after-free fixed; DeepSeek V4/Qwen4-Exp embeddings reported upstream (#32036); Mac gates (`NOTES.md`, "Upstream sync v0.8.0 → v0.9.0") | GB10: **open** · DESKTOP-BR7J5S9: **open** · JWHLPTOP: **excluded**, carrying v0.4.0's through v0.9.0's AVX-512 items |

## Deferred

Upstream changes inside the ported scope that a sync knowingly did not take,
each with its reason (MAINTENANCE.md, Decision 8).

| Since | Change | Reason |
|---|---|---|
| `v0.4.0` | `llama-cli`'s `-lzm/--lazy-mode` and `-ncffn/--n-cpu-ffn` | our `llama-cli` exposes no loading flags (`--load-mode`, `-ot`, `--cpu-moe` are absent too); `cli/upstream_flags.zig` reports them as real flags not yet supported. The library default, `LLAMA_LAZY_MODE_AUTO`, applies |
| `v0.4.1` | `llama-cli`'s `--log-jsonl`/`--no-log-jsonl` | ported for every tool that parses through `common_params_parse` (`tools/common/`); our `llama-cli` has no logging flags (`--log-file`, `--log-colors` are absent too) and silences libllama's log, the same reasoning as `-lzm` |
| `v0.6.0` | `llama-cli`'s `--spec-draft-sampling` and `--rpc` | our `llama-cli` has no speculative decoding and no RPC backend; `cli/upstream_flags.zig` reports both as real flags not yet supported. The tools that parse through `tools/common/` take `--spec-draft-sampling` |
| `v0.6.0` | zigjinja: `dict()`/`namespace()` positional arguments, and `map`/`selectattr`/`rejectattr`/`join`/`sort`/`min`/`max` over array attributes, digit-string and bool indexes | no model template reaches them: `jinja-diff`'s 78 templates agree without them. The two probes that show them are kept out of the gate |
| `v0.6.0` | the order `llama_model_saver` writes `general.tensor_extra.*` in | upstream iterates a pointer-keyed `std::unordered_map`, so its order depends on addresses and is not reproducible run to run; the port writes in `tensors_by_name` order. No gate has a file with those keys |
| `v0.7.0` | `llama-cli`'s `--moe-cache-mib` and `--spec-draft-n-ubatch` | the same reasoning as `-lzm` and `--spec-draft-sampling`: our `llama-cli` has no loading flags and no speculative decoding; `cli/upstream_flags.zig` reports both as real flags not yet supported. The tools that parse through `tools/common/` take both |
| `v0.7.0` | `common_perf_print`'s "graph reallocs" line (sampling.cpp:588) | no ported tool or server path calls `common_perf_print`; the counters it prints are ported (`llama_perf_context_data.n_reallocs_same`/`_diff`) and the server prints its own line |
| `v0.8.0` | `llama-cli`'s `-hf` (named now by upstream's "--model or -hf is required") | our `llama-cli` has no download path; `cli/upstream_flags.zig` reports it as a real flag not yet supported. `SPEC.md`'s `error: no model given` stands; `tools/common/arg.zig` carries upstream's new text |
| `v0.8.0` | zigjinja: a list appended to itself (`arr.append(arr)`), whose `tojson` raises upstream | zigjinja copies by value, so no cycle exists and ours renders `[[]]`. No template does this; `test-jinja`'s case is the only one |
| `v0.8.0` | zigjinja: `mark_used`'s depth limit, and `gather_string_parts`' depth inside `for`/`if` bodies | the statistics walk runs only in caps probes over values `from_json` bounds; upstream's results arrays nest per statement, so its limit comes sooner there and ours counts one level. Only a list 120-odd deep shows either |
| `v0.9.0` | the SVE arms `arch/arm/repack.cpp` added (`ggml_gemv_mxfp4_4x4`, `ggml_gemv_q8_0_4x8`, `ggml_gemm_q6_K_8x8`) | no target the port builds has SVE: no Apple core does, and the GB10's `generic` baseline does not; the citations note it at each kernel |
| `v0.9.0` | CUDA's `ggml_backend_cuda_comm_init_rank`, its `ggml_backend_comm_init_rank` proc entry, and the NCCL all-reduce's `n_ranks` | all under `#ifdef GGML_USE_NCCL`, which this build does not define (as before); `cuda-backend-diff` checks the proc entry is null on both sides |
| `v0.9.0` | `mtmd_helper_video_seq` (mtmd-helper.h:413) | C++-only test API, used by `tests/test-mtmd-impl.cpp`, with no C entry point; it is the ffmpeg video helper's logic, which this build compiles only as its `#else` arms (no `MTMD_VIDEO`, as before). `mtmd_video_config_init` and the qwen3vl template are ported |
| before `v0.3.0` (`0.0.23`) | CUDA's `allreduce.cu` pipeline (`ggml_cuda_ar_pipeline_init`/`_free`); v0.5.0's changes there are HIP enablement and added `CUDA_CHECK`s | the tensor-parallel all-reduce's `internal` arm reports failure and falls back to the meta backend's butterfly with the C's own warning, exactly as the C does when its pipeline fails to start; the GB10 has a single GPU, so no machine here could gate a ported pipeline. Recorded at v0.5.0, where it was first listed |

## Scope

Every upstream path falls in one of four classes.

- **Ported**: cited Zig, synced by hand through the citations and
  `scripts/upstream-map`. ggml core, `ggml-cpu/`, the Metal, CUDA, Vulkan and
  BLAS host layers, `src/` (libllama, every architecture), `tools/mtmd/`, the
  parts of `common/` the CLI and the ten tools use, and those tools.
- **Compiled unmodified**: the CUDA `.cu`/`.cuh` (fatbins), the Metal kernels
  (embedded), the Vulkan shaders (the reference build's SPIR-V, embedded),
  `ggml-metal-device.m` and `ggml-metal-context.m`.
- **Generated**: never edited by hand, regenerated at every sync, each
  stamping the pin from `build/upstream.zig` through `scripts/upstream_pin.py`.
  `gen-arch`, `gen-hparams`, `gen-model-tables`, `gen-layer`,
  `gen-unicode-data`, `gen-mtmd-names`, `gen-kargs`, `gen-tuning-table`,
  `gen-upstream-flags`, `gen-vk-push`, `gen-mmq-config`, `gen-fattn-config`,
  `iq-grids`; the goldens (`vecdot-golden`, `vec-golden`, `quants-golden`);
  the arch and mmproj fixtures.
- **Not ported**: the other backends (Hexagon, OpenCL, SYCL, OpenVINO, RPC,
  CANN, MUSA, HIP, zDNN, WebGPU, ET), KleidiAI, `examples/`, `tests/`,
  `tools/ui` (built by npm and embedded, not ported), and `common/` beyond
  what `cli/` and `tools/common/` mirror. `tools/server` has been ported
  since Stage 10 (`tools/server/`, with cpp-httplib in
  `tools/server/httplib/`).
