# metrics

Performance of the stock C against the port, side by side. Not a gate:
nothing here passes or fails on a number.

```sh
make perf                 # from the root: Metal, RUNS rounds, then the report
make perf-cpu             # the CPU device alone -- the ported CPU kernels
make -C metrics report    # summarise the newest results file again
make -C metrics RUNS=9 NPRED=1024 PROMPT="..."
make -C metrics SUDO=     # no root; CPU, DRAM and ANE power read `na`
make perf PORT_REV=f3d52b7  # the port at a revision, not the working tree
```

**Measure a revision, not the working tree.** This tree is shared with other
sessions, and without `PORT_REV` the port is whatever it builds at that
moment — measured: a CPU pass rebuilt the port mid-session from another
session's in-flight kernel edits. `PORT_REV` builds from `git archive` under
`build/port-<sha>/`, so a baseline can be re-measured later at the same code.

**A busy machine is refused.** `perf-run` samples the GPU's utilisation and
the load average first, and stops if the GPU is over 10% busy or the load is
over a third of the cores. Measured: with another process
holding the GPU at 98%, Metal generation read 33 tok/s on both sides against
233 on a quiet machine, and nothing else in the run looked wrong. `FORCE=1`
measures anyway and the results header says so. After the fact,
`perf-report` flags any run whose GPU drew more than 1 W during the idle
window, before anything of ours was loaded.

Defaults: `Qwen3.5-2B-Q4_K_M.gguf`, the prompt *How was the transistor
discovered?* through the model's chat template, 512 tokens, 5 rounds.

## What is compared

| side | libraries | built by |
|---|---|---|
| `ref` | `llama.cpp.zmake/zig-out/lib/lib{llama,ggml}.a` | the stock C, Zig toolchain, `--release=fast` |
| `port` | `zig-out/lib/lib{llama,ggml}.a` | this project, `zig build reference --release=fast` |

One driver, `perf_driver.c`, is linked against both, so the libraries are
the only variable — the pairing `scripts/parity-port` uses for correctness.
`scripts/check-reference-pin` runs first: a reference at a different
upstream commit would report the version gap as the port's speed.

What `port` contains today is the ported ggml plus the sampler and grammar.
The rest of libllama is still the C++ until it swaps at `llama.h` (`PLAN.md`,
Decision 42), so `port` against `ref` measures **ggml** above all.

This compares libraries, not binaries. Our `llama-cli` adds argument
parsing and chat templating on top; neither runs per token.

## The work is fixed

Decoding is greedy and continues past end-of-generation, so every run
decodes exactly `NPRED` tokens. `tok_hash` hashes the token ids, and
`perf-report` refuses to call the comparison valid unless every run on both
sides decoded the same tokens. Rounds alternate `ref,port` / `port,ref` so
thermal drift cancels, and one priming run per side is discarded.

## Metrics

| metric | source | scope |
|---|---|---|
| `ttft_ms` | wall clock, prompt string → first sampled token, after warmup; includes templating and tokenization | process |
| `pp_tps` | prompt tokens / TTFT | process |
| `tg_tps`, `tg_ms_per_tok` | wall clock over the `NPRED − 1` tokens after the first | process |
| `perf_tg_tps` | `llama_perf_context`'s own figure, as a cross-check | process |
| `load_ms`, `warmup_ms` | wall clock: backend init + weights + context; one throwaway decode | process |
| `footprint_*_mb` | `task_vm_info` `phys_footprint`: anonymous and Metal memory, not the mmapped weights | process |
| `rss_peak_mb` | `getrusage` max RSS: includes the mmapped weight pages | process |
| `cpu_*_s`, `cpu_gen_cores` | `getrusage` user + system; cores = CPU seconds / wall seconds | process |
| `gpu_gen_s`, `gpu_gen_util`, `gpu_ttft_ms` | IORegistry `AppUsage.accumulatedGPUTime` of this pid's Metal clients | process |
| `gpu_w_gen`, `cpu_w_gen`, `dram_w_gen`, `ane_w_gen` | IOReport `Energy Model` over the generation phase | **SoC-wide** |
| `soc_mj_per_tok` | GPU + CPU + DRAM energy / generated token, whichever channels moved | **SoC-wide** |
| `sys_w_idle`, `sys_j_run`, `sys_mj_per_tok` | SMC `PSTR`, integrated; run energy above idle | **whole machine** |

Things each source cannot do, measured rather than assumed:

- **Unprivileged, only the GPU's energy moves.** On macOS 27 the `Energy
  Model` channels for CPU, DRAM and ANE stay frozen for a non-root reader;
  `GPU Energy` (nJ) alone updates. Hence `SUDO=sudo` by default: the driver,
  and only the driver, runs as root. A channel that never moved across the
  run reports `na`, never 0.
- **`PSTR` updates about once a second.** A 0.55 s generation read 17 W
  against a 22 W idle. It is used only for the whole run: integrated from
  the prompt to `TAIL_MS` past the last token, above an idle window of
  `IDLE_MS` taken before loading. A smoothed reading lags but conserves
  area, so the integral still holds what the run added.
- **SoC and system power are not per-process.** Anything else running
  shows up. Close what you can, and read `spread` before any ratio.
- **The prompt phase is ~20 ms.** TTFT and GPU time resolve it; power does
  not, so no power figure is reported for it.

## Reading the report

`perf-report` prints, per metric, the median on each side, the spread
`(max − min) / median`, and `port / ref`. **A ratio inside the spread is not
a difference this run can claim.** The results file is `results/<stamp>.tsv`
with a header recording commit, upstream commit, chip, macOS, power source
and Low Power Mode; a battery run is not comparable to a mains run.
