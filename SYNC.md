# SYNC.md -- the upstream-sync prompt

Moves the port forward through upstream's releases, one tag per run. Say "run
SYNC.md". Re-runnable: it reads its state from "Current state" below.

## Settings

| Setting | Value | Meaning |
|---|---|---|
| Stop after | `latest` | A release tag (`v1.1.0`), or `latest` |
| Push | `yes` | `git push origin master` after each sync commit, so the other machines can pull it; `no` stops at each sync's pickup |

## Rules

- **One release at a time.** Never jump a tag, never move to `master`.
- **Upstream is `https://github.com/ggml-org/llama.cpp`**, fetched by URL, never by whatever remote the checkout names.
- **Every citation ends a sync at the new basis.** Unchanged ones are repinned mechanically; only entries in "Deferred" below may lag.
- **New features inside the ported scope are ported in the same sync.** Deferring one needs a reason in "Deferred".
- **Mac gates block the sync. Other machines' gates go on "Open pickups"** and must clear before the next sync starts.
- **Version:** `build.zig.zon` is the bare upstream tag after a sync (`1.0.0`); any commit between syncs adds `+N` (`1.0.0+1`).
- **Granted:**
  - moving `llama.cpp/` and `llama.cpp.zmake/llama.cpp` to the next tag, and `.gitmodules`;
  - editing `llama.cpp.zmake`'s `build.zig`, `build.zig.zon` and `Makefile` for that release, and committing there;
  - one commit per sync here, plus one per pickup fix; pushing `master` when **Push** is `yes`;
  - fixing `../zigjinja` on its current branch, committing and pinning, when `jinja-diff` needs it.
- **Forbidden:** editing anything under `llama.cpp/` or `llama.cpp.zmake/llama.cpp/`; adding a dependency; force-pushing; changing Settings.

## Current state

- **Pin:** `build/upstream.zig` -- `v1.0.0`, `b1b770d90`, build `b12359`. `testing/scripts/check-reference-pin` holds the submodule, zmake's checkout, its tarball URL and the version literals to it.
- **zigjinja:** `6efb5e4` (v3.1.11, on its `main`), a path dependency at `../zigjinja`.

### Open pickups (v1.0.0)

The v1.0.0 sync ran on the GB10, so the Mac is the machine that has not seen it. On each machine first move `llama.cpp/` and zmake's checkout to `b1b770d90`, zmake to `c2d6b81`, zigjinja to `6efb5e4`.

- **GB10 (`spark-dbff`): clear.** Unreachable on its hardware: CUDA cluster re-instantiation; Vulkan `write_n`/`write_2d` staging, the `!rebar` arm, GROUP_NORM's 2-D split, the Qualcomm/AMD driver arms; fit's skipped NextN layers; MTP device-row indexing on a trained model.
- **Mac: open.**
  1. Metal, never compiled or run (ported blind): the TOPK_MOE fusion rows and `checkTopkMoe`, `_topk_moe`, `_group_norm`, radix top-k without `smem_out`, `get_pipeline_topk_moe`/`group_norm`, `mul_mm` from K >= 32. Then `struct-layout`, `metal-diff`, `metal-ops-diff`, `tuning-diff`, `backend-ops --diff`, `node-diff ARGS=--gpu`.
  2. Goldens: `make buildmacos`, then `vecdot-golden`, `vec-golden` (add `ggml_vec_dot_f16_f32`), `quants-golden`, each with `GOLDEN_CPU=generic`.
  3. Apple CPU set: `ops-diff`, `repack-diff`, `node-diff`, `parity-port`, `parity-port-cpu` at default, `apple_m3`, `apple_a7`, `apple_a12`.
  4. macOS-only: `graph-diff`, `sched-diff`, `abi-check`, `types-golden`, `opt-diff`, `opt-llama-diff`, `adapter-diff`, `meta-diff`, `sampler-diff` (a Linux copy fails 13 typical-sampler records; check on the Mac before reading anything into it).
  5. Models and libmtmd on Metal: `context-diff --gpu`, `arch-model-diff`, `mtp-diff`, `gemma4-assistant-diff`, `mtmd-gates`, `mtmd-audio-diff`.
  6. The rest: `vocab-diff`, `loader-diff`, `arch-diff`, `quant-diff`, `unicode-diff`, `xcframework-check` (after `make -C xcframework upstream`), `blas-diff`, `blas-node-diff`, `cnv-diff`, `parity-cli`.
- **DESKTOP-BR7J5S9: open.** `make x86-gates` at `x86_64_v3`, `win-host-diffs`; new: the AVX/F16C `ggml_vec_dot_f16_f32` and F16×F32 tinyBLAS rows, the unbuffered `llama_file` arm via a `--direct-io` load. Check `llama-cli.exe` and the tools build.
- **GB10, since v1.0.0+14: open.** Compile CUDA (`-Dcuda=true`): ggml's 568 CUDA calls now go to the ported Zig functions rather than through `ggml_c`, never compiled. Vulkan against the real `libvulkan` (the host layer type-checks on the Mac against a stub).
- **zigggml: open.** Its A/B gates run this repository's harnesses by path, and those moved under `testing/` (`testing/harness/`, `testing/scripts/`, `testing/harness/port_links.zig`); harnesses import libllama as `llama` from `src/llama/module.zig` (no `harness_root.zig`). Update its references, then run its `make ab-*` groups.
- **JWHLPTOP: excluded** (no access). Carries every AVX-512 item since v0.4.0, now including `ggml_vec_dot_f16_f32`.

### Known not to reproduce against itself

Not port bugs; run `--self` before reading a failure here.

- GB10: `server-diff` `parallel`, `sleep`, and `decision`'s `!parallel` block (CUDA batches concurrent requests by timing; the 1 s idle timeout races startup).

---

## Steps

### 0. Preflight (every run)

1. `git status` clean here, in `llama.cpp.zmake` and in `../zigjinja`; no other agent or `zig build` running in this tree (`ps`). Otherwise stop and report.
2. Basis A = the pin above.
3. Next tag B:
   - `git -C llama.cpp fetch --tags https://github.com/ggml-org/llama.cpp.git`
   - `gh api 'repos/ggml-org/llama.cpp/releases?per_page=100'`: non-prerelease `v*` entries only (skip `bNNNNN`), `sort -V`, patch releases included; B is the first after A.
   - B must resolve to `gh api repos/ggml-org/llama.cpp/commits/<tag> --jq .sha`. Never use `target_commitish` (it is the branch head, later than the tag).
   - None: report the port is current and stop.
4. Any pickup open above: run it (Step 8) first.

### 1. Measure

- `git -C llama.cpp diff --stat A B`, `git log --oneline A B` over ported paths, plus the release notes.
- Scratchpad sizing note: files changed, lines ±, new/deleted files, new architectures, ops, flags, `llama.h` symbols.

### 2. Move the reference

1. Check out B in `llama.cpp/` and `llama.cpp.zmake/llama.cpp`; update `.gitmodules` (`branch = B`), zmake's tarball URL in `build.zig.zon`, and `build/upstream.zig` (tag, `LLAMA_VERSION`, `GGML_VERSION` from `ggml/CMakeLists.txt`, commit, build number from `git describe --tags --match 'b*'`). Set `build.zig.zon`'s `.version` to B now.
2. Sources added, moved or deleted: update `llama.cpp.zmake/build.zig` and `build/llamacpp.zig`.
3. Rebuild the zmake reference and `zig build reference`, both `--release=fast`. **Hash the artifacts and confirm they changed**: a failed build leaves A's library and every gate passes against it.

### 3. Work list

- `testing/scripts/upstream-map A B` > scratchpad. Three kinds: touched Zig functions; unmapped hunks (new definitions, uncited files, headers, structs, macros); deletions.
- Each item ends ported, deferred (with a reason, in "Deferred"), or out of scope (per "Scope").

### 4. Regenerate

Never hand-edit generated files. `build/upstream.zig` must already be at B.

- Scripts: `gen-arch`, `gen-kargs` (then `zig fmt`, `make struct-layout`), `gen-hparams`, `gen-model-tables`, `gen-layer`, `gen-unicode-data`, `gen-mtmd-names`, `gen-vk-push`, `gen-tuning-table` (then `make tuning-diff`), `gen-upstream-flags`, `iq-grids`.
- A generator that stops on new C++: extend the generator, never the output.
- Hand-kept lists they feed: `impl_c.zig`'s `count`, `testing/harness/struct_layout.cpp`'s constants.
- Goldens: `make buildmacos` first (they link upstream's CMake build), then `vecdot-golden`, `vec-golden`, `quants-golden`, each also with `GOLDEN_CPU=generic`.
  - Delete the untracked `src/ggml/cpu/quants/golden_generic.zig` it writes.
  - Restore `quants-golden`'s `IQ4_NL` `zeros` line (uninitialized read upstream; differs every capture).
- Fixtures: `arch-fixtures`, `mmproj-fixtures`. Compare the "no fixture" count with the last sync's.
- GB10 only (to the pickup): `gen-fattn-config`, `gen-mmq-config`, the CUDA fatbins and kernel table, the Vulkan SPIR-V.
- **Diff every regenerated file**: a new ordinal, field or argument has a consumer to port.

### 5. Port

Mechanical moves first:

- `testing/scripts/upstream-map --repin A B`: moves every citation whose definition is unchanged; refuses changed ones.
- `testing/scripts/upstream-relines --write A B`: moves the `"<file>.cpp", <LINE>` literals passed to `impl.assert`/`impl.abort`.
  - **Run once**; it is not idempotent.
  - `--keep <zig>:<file>:<N>` for literals already set by hand; `--skip <prefix>` for files another agent is editing.
  - Its "changed at B" list is hand work.
- After everything else: read `testing/scripts/upstream-literals` (catches literals behind a name). A report, not a gate: check each before fixing.

Then by hand, in dependency order: ggml core, CPU kernels, backends, libllama, libmtmd, `cli/`, `tools/`.

- Every touched definition re-read and cited at B: ``(file.c:N @<B-sha>)``. `CLAUDE.md`'s porting rules apply.
- Port CUDA and Vulkan host changes here too; the GB10 gates them.
- New in upstream:

| New | Do |
|---|---|
| Architecture | `src/llama/models/` file, registry entry, hparams arm, `arch-model-diff` fixture |
| Op | Constructor, CPU kernel, `metal/ops.zig` encoder, CUDA and Vulkan host, `supports_op` everywhere, `graph-diff` and `ops-diff` cases |
| Sampler | Port plus `sampler-diff` coverage |
| Flag | `cli/args.zig` or `tools/common/`, `upstream_flags.zig`, a `parity-cli`/`tools-diff` case |
| `llama.h`/`mtmd.h` symbol | `api/` plus export; `nm` count equal to the reference |
| Projector type | `arms_*.zig` arm, `mmproj-fixtures` type |
| Tool or backend | Out of scope: add it to "Not ported" |

- Sub-agents: disjoint files, own scratch dirs; nothing registered in a `module.zig` until `zig build test-llama` passes with it.
- Done when: `port-links` clean (bar "Deferred"); `git grep '@<A>' src cli tools` empty; `check-reference-pin` passes; every file header's "at vX (`sha`)" line moved.

### 6. Mac gates (block the sync)

1. `make validate`. With `../styleguide/Makefile.in`, `validate` runs `clean` and `zig fmt`: generate fixtures **after** it.
2. Bit-exact CPU set -- `ops-diff`, `repack-diff`, `node-diff`, `parity-port`, `parity-port-cpu` -- at default `GATE_CPU`, `apple_m3`, `generic`; plus `apple_a7`, `apple_a12` if `src/ggml/cpu/` changed.
3. `node-diff ARGS=--gpu`.
4. Every other Mac gate:
   - `graph-diff`, `backend-ops --diff`, `sched-diff`, `metal-diff`, `metal-ops-diff`, `tuning-diff`, `struct-layout`, `abi-check`
   - `sampler-diff`, `libcxx-diff`, `regex-diff`, `unicode-diff`
   - `vocab-diff`, `arch-diff`, `loader-diff`, `context-diff`
   - `arch-model-diff`, `mtp-diff`, `gemma4-assistant-diff`, `moe-slru-diff`
   - `jinja-diff`, `cnv-diff`, `fit-diff` (also `FIT_DIFF_O=ReleaseSafe`), `tools-diff` one tool at a time
   - `json-diff`, `schema-diff`, `peg-diff`, `chat-parse-diff`, `chat-msg-diff`, `spec-diff`
   - `server-task-diff`, `http-diff`, `server-diff` one case file at a time
   - `blas-diff`, `blas-node-diff`, `mtmd-gates`
   - `api-diff`, `adapter-diff`, `quant-diff`, `opt-diff`, `opt-llama-diff`
   - `meta-diff`, `batch-diff`, `chat-diff`, `host-arms-diff`
   - `parity-cli`; `xcframework-check` after `make -C xcframework upstream`
- A gate that did not build is not a pass.
- Every case added this sync: inject a fault, watch it fail, undo by reverse replacement.
- Fix the port, not the gate, unless the gate is wrong.

### 7. Record

- `HISTORY.md`: one entry for B (see its format).
- This file: "Current state" (pin, zigjinja, open pickups for B, all `open`), "Deferred".
- `CLAUDE.md`: every count that changed (exports, architectures, projector types, gate counts, pinned tag). A stale number is a bug.

### 8. Pickups

Must clear before the next sync. The commit (Step 9) reaches GitHub first -- or, to run them beside Step 6, carry the uncommitted tree as a commit object (`git commit-tree` from a scratch index, never on a branch) in a bundle to a local branch there, and bring every fix back before committing.

Reaching the machines: `../zigupdater`'s `zigupdater`.

- `tty <agent>` for a shell; it needs a terminal, so script it as `script -q /dev/null zigupdater tty <agent>`, one command line per session, long jobs under `nohup`.
- `push <agent> <exe> --run` for programs. Executables only: ship files as a tar inside a carrier program (rebuild it with a changing `-D`; `zig cc` does not track an `.incbin`'d file).
- One updater host on the Mac: concurrent pushes refuse each other (`testing/scripts/win-run` retries).

Per machine:

- **GB10** (`spark-dbff`, `~/src/mlavergn/llama.zig`; needs a `llama.cpp.zmake -> ../llama.cpp.zmake` symlink):
  - Step 4's GB10 regenerations, committed back from the Mac
  - `cuda-ops-diff`, `cuda-node-diff` (per node and fused), `cuda-backend-diff`, `cuda-parity-cli`
  - `vk-toolchain` if the pin moved; `vk-repro`, `vk-ops-diff`, `vk-node-diff`, `vk-backend-diff`, `vk-push-layout`, `vk-parity-cli` on all four arms. `vk-node-diff` and `vk-backend-diff` loop the arms themselves; the others take it from the environment: `GGML_VK_DISABLE_COOPMAT2=1`, then plus `_COOPMAT=1`, then plus `_INTEGER_DOT_PRODUCT=1`
  - Linux arm64 CPU set, `host-arms-diff`, every model, libmtmd, chat, tools and server gate. Not `sampler-diff` (macOS only)
- **DESKTOP-BR7J5S9:** `make x86-gates` at `x86_64_v3`, `win-host-diffs`, the 1M-deep `json-diff`; `llama-cli.exe` and the tools build. `[The Agent could not open a terminal]` usually means Application Control refused the binary; run it in a `tty` to see.
- **JWHLPTOP** (unavailable): `make x86-gates` with `X86_CPU=x86_64_v4`, then `tigerlake`; confirm 512-bit code by disassembly.
- Everywhere: find each named case in the gate's output; a total does not say which cases ran.

Fix failures on the Mac, commit (`… pickup fix for upstream B`), re-run, then mark the machine `clear` above. **An unreachable machine stops the loop.**

### 9. Commit

- Version: `B` without the `v` for a sync (set since Step 2); `+N` for a pickup fix.
- Message: `v<version> sync upstream B (<B-short-sha>)`, or `v<version> pickup fix for upstream B`; then the attribution line.
- One commit here; one in `llama.cpp.zmake` if it changed. Push if **Push** is `yes`.

### 10. Next

Report in one or two lines: B, version, items ported and deferred, pickup state. Back to Step 0. Stop only at **Stop after**, the latest release, or a blocker:

- a gate failure you cannot resolve;
- an unreachable pickup machine;
- a decision only a human can make: a new dependency, a change against a `CLAUDE.md` rule, a new backend or tool that may belong in scope.

A question that can wait goes in the `HISTORY.md` entry; keep going.

---

## Deferred

Upstream changes inside the ported scope deliberately not taken.

| Since | Change | Reason |
|---|---|---|
| v0.4.0 | `llama-cli` `-lzm/--lazy-mode`, `-ncffn/--n-cpu-ffn` | Our `llama-cli` has no loading flags; `upstream_flags.zig` reports them as unsupported. `LLAMA_LAZY_MODE_AUTO` applies |
| v0.4.1 | `llama-cli` `--log-jsonl`/`--no-log-jsonl` | No logging flags in our `llama-cli`; ported for `tools/common/` |
| v0.6.0 | `llama-cli` `--spec-draft-sampling`, `--rpc` | No speculative decoding or RPC in our `llama-cli`; `tools/common/` takes the former |
| v0.6.0 | zigjinja: `dict()`/`namespace()` positional args; `map`/`selectattr`/`rejectattr`/`join`/`sort`/`min`/`max` over array attributes; digit-string and bool indexes | No model template reaches them |
| v0.6.0 | `llama_model_saver`'s `general.tensor_extra.*` order | Upstream iterates a pointer-keyed map (not reproducible); ours writes `tensors_by_name` order |
| v0.7.0 | `llama-cli` `--moe-cache-mib`, `--spec-draft-n-ubatch` | As `-lzm`; `tools/common/` takes both |
| v0.7.0 | `common_perf_print`'s "graph reallocs" line | Nothing ported calls it; the counters are ported |
| v0.8.0 | `llama-cli` `-hf` | No download path; our `error: no model given` stands |
| v0.8.0 | zigjinja: a list appended to itself | zigjinja copies by value, renders `[[]]` where upstream raises; no template does this |
| v0.8.0 | zigjinja: `mark_used`'s depth limit, `gather_string_parts`' depth in `for`/`if` | Only a list ~120 deep shows either |
| v0.9.0 | SVE arms in `arch/arm/repack.cpp` | No target here has SVE |
| v0.9.0 | CUDA `ggml_backend_cuda_comm_init_rank`, its proc entry, NCCL `n_ranks` | `GGML_USE_NCCL` undefined; `cuda-backend-diff` checks the entry is null on both sides |
| v0.9.0 | `mtmd_helper_video_seq` | C++-only test API; no `MTMD_VIDEO` |
| v1.0.0 | `llama-cli` `--moe-cache-cpu-share`, `--ple/--shard` | As `-lzm`; `tools/common/` takes both |
| v1.0.0 | `MTMD_DEBUG_FUNCTIONS` image writers | Compiled out upstream; `mtmd_debug_*` unported |
| pre-v0.3.0 | CUDA `allreduce.cu` pipeline | Falls back to the meta backend's butterfly with the C's own warning, as the C does when its pipeline fails; single-GPU GB10 cannot gate a port |

## Scope

- **Ported** (cited Zig, synced through citations and `upstream-map`): ggml core, `ggml-cpu/`, the Metal, CUDA, Vulkan and BLAS host layers, `src/` (libllama, every architecture), `tools/mtmd/`, `tools/server/` (with cpp-httplib), the ten tools, and the parts of `common/` they and `cli/` use.
- **Compiled unmodified:** CUDA `.cu`/`.cuh` (fatbins), Metal kernels (embedded), Vulkan SPIR-V (the reference build's, embedded), `ggml-metal-device.m`, `ggml-metal-context.m`.
- **Generated** (Step 4): the `gen-*` scripts, `iq-grids`, the goldens, the arch and mmproj fixtures.
- **Not ported:** Hexagon, OpenCL, SYCL, OpenVINO, RPC, CANN, MUSA, HIP, zDNN, WebGPU, ET; KleidiAI; `examples/`, `tests/`; `tools/ui` (built by npm, embedded); `common/` beyond what `cli/` and `tools/common/` mirror.
