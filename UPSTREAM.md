# UPSTREAM.md

Where the port stands in upstream's history, and how it got there. The plan
behind it is `MAINTENANCE.md`; the procedure a sync follows is `SYNC.md`.

## Basis

```
basis: v0.4.1 b29c606e28a01b1bc8c1351026a0fa6e616bf6c4 b10964
```

| | |
|---|---|
| Upstream | `https://github.com/ggml-org/llama.cpp` (the submodule's `origin`, and `.gitmodules`) |
| Tag | `v0.4.1`, released 2026-09-14 |
| Commit | `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`; citations carry `@b29c606e2` |
| Build tag | `b10964` |
| ggml version | `0.24.0` (ggml/CMakeLists.txt), which `ggml_version()` returns |
| Recorded in | `build/upstream.zig`, `llama.cpp.zmake/build.zig`, `src/ggml/runtime.zig`, `src/llama/api/llama.zig`; `scripts/check-reference-pin` holds them to the submodule |

The `basis:` line is the machine-readable one: tag, full commit, build tag.

## History

| From | To | Date | Port version | Port commit | Work | Pickups |
|---|---|---|---|---|---|---|
| — | `v0.3.0` (`c1d0e7a00`, `b10621`) | 2026-10-27 | `0.3.0` | `bd498fc` | the port as built through Stages 1–8 | — |
| `v0.3.0` | `v0.4.0` (`5266f24da`, `b10809`) | 2026-10-28 | `0.4.0` | `a67ea88` | 456 touched citations, 217 unmapped hunks, 29 uncited files; 7,601 citations repinned; 529 line literals moved; Mac gates all pass (`NOTES.md`, "Upstream sync v0.3.0 → v0.4.0") | GB10: clear (`7c9cbc1`; `lint` not run, no `zlint` there) · DESKTOP-BR7J5S9: clear (`7c9cbc1`; `quant-diff` has no Windows mode) · JWHLPTOP: **open**, unreachable 2026-10-28 |
| `v0.4.0` | `v0.4.1` (`b29c606e2`, `b10964`) | 2026-10-29 | `0.4.1` | `d4f0a16` | 197 touched citations, 168 unmapped hunks, 43 uncited files; 7,231 citation sites repinned; 316 line literals moved; three architectures (hy_v4, maple, spark2_5), the Metal fusion table, the Vulkan matmul refactor; Mac gates (`NOTES.md`, "Upstream sync v0.4.0 → v0.4.1") | GB10: clear (`0.4.1+1`; three gate fixes, `diag_f16` found) · DESKTOP-BR7J5S9: clear (`0.4.1+1`) · JWHLPTOP: **open** (with v0.4.0's AVX-512 items), unreachable 2026-10-29 |

## Deferred

Upstream changes inside the ported scope that a sync knowingly did not take,
each with its reason (MAINTENANCE.md, Decision 8).

| Since | Change | Reason |
|---|---|---|
| `v0.4.0` | `llama-cli`'s `-lzm/--lazy-mode` and `-ncffn/--n-cpu-ffn` | our `llama-cli` exposes no loading flags (`--load-mode`, `-ot`, `--cpu-moe` are absent too); `cli/upstream_flags.zig` reports them as real flags not yet supported. The library default, `LLAMA_LAZY_MODE_AUTO`, applies |
| `v0.4.1` | `llama-cli`'s `--log-jsonl`/`--no-log-jsonl` | ported for every tool that parses through `common_params_parse` (`tools/common/`); our `llama-cli` has no logging flags (`--log-file`, `--log-colors` are absent too) and silences libllama's log, the same reasoning as `-lzm` |

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
