# UPSTREAM.md

Where the port stands in upstream's history, and how it got there. The plan
behind it is `MAINTENANCE.md`; the procedure a sync follows is `SYNC.md`.

## Basis

```
basis: v0.4.0 5266f24da75dc449bd56cbed7addb9c8e4a6a73e b10809
```

| | |
|---|---|
| Upstream | `https://github.com/ggml-org/llama.cpp` (the submodule's `origin`, and `.gitmodules`) |
| Tag | `v0.4.0`, released 2026-09-04 |
| Commit | `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`; citations carry `@5266f24da` |
| Build tag | `b10809` |
| ggml version | `0.23.0` (ggml/CMakeLists.txt), which `ggml_version()` returns |
| Recorded in | `build/upstream.zig`, `llama.cpp.zmake/build.zig`, `src/ggml/runtime.zig`, `src/llama/api/llama.zig`; `scripts/check-reference-pin` holds them to the submodule |

The `basis:` line is the machine-readable one: tag, full commit, build tag.

## History

| From | To | Date | Port version | Port commit | Work | Pickups |
|---|---|---|---|---|---|---|
| — | `v0.3.0` (`c1d0e7a00`, `b10621`) | 2026-10-27 | `0.3.0` | `bd498fc` | the port as built through Stages 1–8 | — |
| `v0.3.0` | `v0.4.0` (`5266f24da`, `b10809`) | 2026-10-28 | `0.4.0` | `v0.4.0 sync upstream v0.4.0` (hash recorded by the next commit) | 456 touched citations, 217 unmapped hunks, 29 uncited files; 7,601 citations repinned; 529 line literals moved; Mac gates all pass (`NOTES.md`, "Upstream sync v0.3.0 → v0.4.0") | GB10: open · DESKTOP-BR7J5S9: open · JWHLPTOP: open |

## Deferred

Upstream changes inside the ported scope that a sync knowingly did not take,
each with its reason (MAINTENANCE.md, Decision 8).

| Since | Change | Reason |
|---|---|---|
| `v0.4.0` | `llama-cli`'s `-lzm/--lazy-mode` and `-ncffn/--n-cpu-ffn` | our `llama-cli` exposes no loading flags (`--load-mode`, `-ot`, `--cpu-moe` are absent too); `cli/upstream_flags.zig` reports them as real flags not yet supported. The library default, `LLAMA_LAZY_MODE_AUTO`, applies |

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
