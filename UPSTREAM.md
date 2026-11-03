# UPSTREAM.md

Where the port stands in upstream's history, and how it got there. The plan
behind it is `MAINTENANCE.md`; the procedure a sync follows is `SYNC.md`.

## Basis

```
basis: v0.6.0 d81235049384534c167caea52b85a694f6103d14 b11429
```

| | |
|---|---|
| Upstream | `https://github.com/ggml-org/llama.cpp` (the submodule's `origin`, and `.gitmodules`) |
| Tag | `v0.6.0`, released 2026-10-05 |
| Commit | `d81235049384534c167caea52b85a694f6103d14`; citations carry `@d81235049` |
| Build tag | `b11429` |
| ggml version | `0.26.0` (ggml/CMakeLists.txt), which `ggml_version()` returns |
| Recorded in | `build/upstream.zig`, `llama.cpp.zmake/build.zig`, `src/ggml/runtime.zig`, `src/llama/api/llama.zig`; `scripts/check-reference-pin` holds them to the submodule |

The `basis:` line is the machine-readable one: tag, full commit, build tag.

## History

| From | To | Date | Port version | Port commit | Work | Pickups |
|---|---|---|---|---|---|---|
| — | `v0.3.0` (`c1d0e7a00`, `b10621`) | 2026-10-27 | `0.3.0` | `bd498fc` | the port as built through Stages 1–8 | — |
| `v0.3.0` | `v0.4.0` (`5266f24da`, `b10809`) | 2026-10-28 | `0.4.0` | `a67ea88` | 456 touched citations, 217 unmapped hunks, 29 uncited files; 7,601 citations repinned; 529 line literals moved; Mac gates all pass (`NOTES.md`, "Upstream sync v0.3.0 → v0.4.0") | GB10: clear (`7c9cbc1`; `lint` not run, no `zlint` there) · DESKTOP-BR7J5S9: clear (`7c9cbc1`; `quant-diff` has no Windows mode) · JWHLPTOP: **open**, unreachable 2026-10-28 |
| `v0.4.0` | `v0.4.1` (`b29c606e2`, `b10964`) | 2026-10-29 | `0.4.1` | `d4f0a16` | 197 touched citations, 168 unmapped hunks, 43 uncited files; 7,231 citation sites repinned; 316 line literals moved; three architectures (hy_v4, maple, spark2_5), the Metal fusion table, the Vulkan matmul refactor; Mac gates (`NOTES.md`, "Upstream sync v0.4.0 → v0.4.1") | GB10: clear (`0.4.1+1`; three gate fixes, `diag_f16` found) · DESKTOP-BR7J5S9: clear (`0.4.1+1`) · JWHLPTOP: **open** (with v0.4.0's AVX-512 items), unreachable 2026-10-29 |
| `v0.4.1` | `v0.5.0` (`7fe450e19`, `b11146`) | 2026-11-01 | `0.5.0` | `e9c4b5b` | 762 touched citations (461 of them Vulkan's file split), 246 unmapped hunks, 24 uncited files; 9,560 citation sites repinned; 388 line literals moved; one architecture (`hrm_text`), one `llama.h` function, one op form (`ggml_dsv4_hc_pre_gated`), the q1_0 repack, the Metal tuning rewrite and four fusions, the Vulkan split, the `ling3` chat parser; Mac gates (`NOTES.md`, "Upstream sync v0.4.1 → v0.5.0") | GB10: clear (`NOTES.md`, "The v0.5.0 pickup") · DESKTOP-BR7J5S9: clear, apart from `http-diff` and `tools-diff server`, which wait for the server's Windows port · JWHLPTOP: **open**, unreachable (Marc, 2026-11-01), carrying v0.4.0's and v0.4.1's AVX-512 items |
| `v0.5.0` | `v0.6.0` (`d81235049`, `b11429`) | 2026-11-03 | `0.6.0` | (this commit) | 760 touched citations, 413 unmapped hunks, 42 uncited files (after the `upstream-map` extent fix; 757/416 before); 9,584 citation sites repinned; 599 line literals moved by `upstream-relines` and the rest by hand; two architectures (`clef`, `glm5-next`), two projector types (`ling3vl`, `glm5v`), 14 `llama.h` functions (`llama_batch_ext`, `llama_process`, `llama_get_causal_attn`), `ggml_backend_buft_alloc_buffer_n`/`get_alloc_size_n`, `ggml-cpu/tiled` replacing `iqp`, the Metal MMA mat-vec and FA tensor path, the CUDA `CONV_3D` path and shared-expert fusion, the Vulkan coopmat1 integer-dot mmq, the `/v1/systemone` decision API, the `llm-jp-harmony` chat parser; Mac gates (`NOTES.md`, "Upstream sync v0.5.0 → v0.6.0") | GB10: **open** · DESKTOP-BR7J5S9: **open** · JWHLPTOP: **open**, carrying v0.4.0's, v0.4.1's and v0.5.0's AVX-512 items |

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
  CANN, MUSA, HIP, zDNN, WebGPU, ET), `examples/`, `tools/server`,
  `tools/ui`, and `common/` beyond what `cli/` and `tools/common/` mirror.
