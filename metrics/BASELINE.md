# Performance baseline

The gap between **llama.cpp.zig** — the stock C, built by `llama.cpp.zmake`
with the Zig toolchain — and **llamazig**, tracked release by release until
the port closes it and, the aim, passes it. `metrics/README.md` defines every
metric and its limits; this file records what they read.

## Status at 0.0.16

**Not yet measured.** No run so far meets the conditions below, so there is
no baseline row. The two sessions that tried did not measure the libraries:

- **The GPU was occupied** by another process (98% utilisation) and the
  load average sat between 6 and 10 from sibling sessions' builds. Which
  process was not measured at the time; the CPU list was read instead and
  pointed at a training job running `--preset cpu`, wrongly. Measured per
  process shortly after, `VTDecoderXPCService` — video playback — held 83%
  of GPU time. Metal generation read 33 tok/s on *both* sides against 233
  on a quiet machine. `perf-report` now marks such a run `CONTENDED` and
  `perf-run` refuses to start one.
- **The port was not 0.0.16.** `make -C metrics` rebuilt the port from the
  shared working tree, which carried other sessions' in-flight edits to
  `src/ggml/cpu/` and `src/llama/`. `PORT_REV` now builds from a commit.
- **The libraries were shared.** Both sides used to build into `zig-out/`
  and `llama.cpp.zmake/zig-out/`, which the gates use and which another
  session rebuilds at `-Dcpu=apple_m3` to gate the i8mm kernels. Both sides
  now install into `metrics/build/` with an explicit `-Dcpu`.

Both results files were discarded. What they and one earlier quiet run
suggest is under *Provisional*, and none of it is the baseline.

## Conditions

| | |
|---|---|
| llama.cpp.zig | `llama.cpp.zmake`, upstream `c1d0e7a00` (v0.3.0), `zig build lib --release=fast -Dcpu=baseline`, installed to `metrics/build/ref-baseline/` |
| llamazig 0.0.16 | commit `f3d52b7` (no tag) from `git archive`, `zig build reference --release=fast -Dcpu=baseline`, installed to `metrics/build/port-f3d52b765-baseline/` |
| machine | Apple M5 Max, 18 cores, 128 GiB, macOS 27.0.1 (26A434), AC power, `powermode 0` |
| model | `Qwen3.5-2B-Q4_K_M.gguf` |
| prompt | *How was the transistor discovered?*, through the model's chat template (14 tokens) |
| work | 512 tokens, greedy, past end-of-generation; 5 rounds, alternating, after one discarded priming run per side |
| power | `sudo`, so CPU, DRAM and ANE energy are readable |
| machine state | GPU ≤ 10% and load ≤ 6 before the run (`perf-run` checks); `gpu_w_idle` ≤ 1 W in the report |

Both sides compile the CPU kernels for the macOS baseline (`apple_m1`), so on
the CPU device they run the same NEON paths; the M5's i8mm kernels are
compiled into neither.

What "llamazig" contains at 0.0.16: all of ggml (CPU and Metal host code) and
the sampler and grammar are Zig; the rest of libllama is still the C++ and
swaps at once at `llama.h` (`PLAN.md`, Decision 42). So this comparison
measures **ggml** above all, and on Metal, mostly the host layer — the
kernels are the same MSL on both sides.

To take the baseline:

```sh
make perf     PORT_REV=f3d52b7   # Metal
make perf-cpu PORT_REV=f3d52b7   # CPU device alone: the ported CPU kernels
```

## Baseline — 0.0.16

To be filled from the two runs above. Medians of 5; *spread* is
`(max − min) / median`, and a ratio inside the spread is not a difference.

### Metal

| metric | llama.cpp.zig | llamazig 0.0.16 | llamazig / zig | better |
|---|---:|---:|---:|---|
| TTFT (ms) | | | | lower |
| prompt (tok/s) | | | | higher |
| generation (tok/s) | | | | higher |
| load (ms) | | | | lower |
| peak footprint (MiB) | | | | lower |
| peak RSS (MiB) | | | | lower |
| CPU during generation (cores) | | | | lower |
| GPU busy during generation (%) | | | | — |
| GPU power (W) | | | | lower |
| CPU power (W) | | | | lower |
| DRAM power (W) | | | | lower |
| SoC energy (mJ/token) | | | | lower |
| system energy above idle (mJ/token) | | | | lower |

### CPU device

| metric | llama.cpp.zig | llamazig 0.0.16 | llamazig / zig | better |
|---|---:|---:|---:|---|
| TTFT (ms) | | | | lower |
| prompt (tok/s) | | | | higher |
| generation (tok/s) | | | | higher |
| peak footprint (MiB) | | | | lower |
| CPU during generation (cores) | | | | lower |
| CPU power (W) | | | | lower |
| SoC energy (mJ/token) | | | | lower |

## Provisional

Indicative only, each for the reason given. Re-measure before quoting any of
it.

**Metal: at parity.** One quiet run while the apparatus was built — GPU idle
at 0.06 W, 2 rounds of 256 tokens, no `sudo`, the port built from the working
tree at about 0.0.16 (not pinned):

| metric | llama.cpp.zig | llamazig | ratio |
|---|---:|---:|---:|
| TTFT (ms) | 19.9 | 20.3 | 1.02 |
| generation (tok/s) | 233.9 | 232.8 | 0.995 |
| peak footprint (MiB) | 299 | 314 | 1.05 |
| CPU during generation (cores) | 0.18 | 0.20 | 1.12 |
| GPU busy (%) | 94.2 | 92.1 | 0.98 |
| GPU power (W) | 41.0 | 41.5 | 1.01 |
| SoC energy, GPU only (mJ/token) | 176 | 179 | 1.02 |

Generation and TTFT are inside their spread. The footprint (+15 MiB) and
the host CPU time (+12%) were outside it and are the two to watch when the
baseline is taken: on Metal they are the only costs that are the port's own.

**CPU device: possibly the real gap.** The contended run read llamazig at
**0.21×** llama.cpp.zig in generation (7.5 against 35.5 tok/s) and 0.09× in
prompt processing, with both sides on 3.9 cores. That run was contended *and*
built from other sessions' in-flight CPU-kernel edits, so it says nothing
reliable about 0.0.16. But contention slows both sides alike, and a 4.7×
difference is larger than anything contention did to the Metal pass, so it
is the first thing to confirm.

**One load stalled.** A single llamazig run took 12.3 s to load against
0.29 s of CPU — blocked, not computing. Six repeats loaded in 267–300 ms on
both sides. Probably external; the median absorbs it, and the spread column
will show it if it recurs.

## History

One row per release, from the medians above. A ratio is llamazig /
llama.cpp.zig, so above 1.0 is better for throughput and worse for time.

| llamazig | date | Metal tok/s | CPU tok/s | Metal TTFT | CPU TTFT | footprint | SoC mJ/token | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 0.0.16 | 2026-10-13 | — | — | — | — | — | — | not yet measured; see Status |
