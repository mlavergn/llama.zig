# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## The goal

**Port llama.cpp to Zig — pure Zig apart from the Metal Objective-C host layer.** The deliverable is a `llama-cli` reimplemented in Zig: not a recreation of the llama.cpp repo, and not a Zig wrapper around the C++ library. `SPEC.md` states what the deliverable is and does — artifacts, flag surface, generation semantics, and the conformance gates. `README.md` has the five stages; `PLAN.md` has the detailed plan, scope measurements, decisions, and open questions. Read `SPEC.md` first when the question is "what should this do"; `PLAN.md` when it is "why is it done this way".

Two decisions worth knowing before touching anything:

- **`ggml-metal-device.m` and `ggml-metal-context.m` stay Objective-C** (3,091 lines), compiled by `zig cc`. Zig has no Obj-C frontend, and reaching Metal from Zig would mean hand-writing every call against `objc_msgSend`. Everything else is ported.
- **Our `llama-cli` mimics upstream's argument set** for the features we support, so an existing command line runs unchanged against it. It is our own Zig binary, not upstream's linked against our library.

Where we are: **Stages 1, 2 and 3 complete.**

**No C compiles anywhere under `llama.cpp/ggml/src/` any more.** All six translation units are Zig, 598 exported symbols, every one swapped in and byte-verified. `ggml_base_sources` and `ggml_cpu_c_sources` in `build/llamacpp.zig` are both empty.

- `ggml-alloc.c` → `src/ggml/alloc.zig`.
- `ggml.c` → 381/381 symbols, split across `src/ggml/{impl,types,context,runtime,ops,graph,quantize}.zig`.
- `ggml-quants.c` → 79/79 symbols, split across `src/ggml/quants/`.
- `ggml-cpu/ggml-cpu.c` → 65/65 symbols, split across `src/ggml/cpu/`.
- `ggml-cpu/quants.c` → 45/45 symbols, split across `src/ggml/cpu/quants/`.
- `ggml-cpu/arch/arm/quants.c` → 28/28 symbols, split across `src/ggml/cpu/quants/arm/`.

What remains is C++ and Obj-C: the op kernels in `ggml-cpu/ops.cpp` and `vec.cpp`, the backend registry, the Metal host layer, and all of libllama. That is Stage 4.

`make validate` reports the exact counts.

**Porting a translation unit is treated as all-or-nothing.** A file is ported completely before it is swapped; the linker's missing-symbol errors are the completeness check. Develop a large port without wiring it in, so the build stays green until it is finished.

An incremental alternative was trialled and parked: compiling the C with `-Dggml_foo=ggml_foo_c` renames a symbol *and its internal callers* within that translation unit, leaving the external name free for a Zig version. It looked like it worked and did not — see `PLAN.md`, "The incremental-swap trial". The wholesale swap of `ggml.c` made it moot.

**Every ported file states its source as a full path.** The doc comment at the
top of a ported file names the reference file it replaces as a path from the
repository root, with extension — `llama.cpp/ggml/src/ggml-alloc.c`, not
`ggml-alloc.c` — plus the pinned tag and commit. A file drawing on more than one
source lists all of them. Within the file, each ported declaration names the C
function and the line it began at using the bare filename, since the header has
already established the path. Files that are not ports (`module.zig`,
`ported.zig`) say so explicitly rather than staying silent.

**Those citations are the diff map, and `make port-links` enforces them.** 850
of them, across 42 files. When the pin moves, they are what says which of our
functions a given upstream change lands in — so a stale line number is worse
than no line number, because it sends you to the wrong function and says
nothing. The rules:

- **A citation points at the first line of the definition, and names the commit
  it was read at** — ``Ports `ggml_hash_set_new` (ggml.c:6516 @c1d0e7a00)``.
  Where the name only appears on the closing line, as in `ggml-common.h`'s
  anonymous `typedef struct { … } block_q4_K;`, the opening line is cited and
  the checker follows the brace.
- **The commit goes on every citation, not once per file.** An upstream sync
  does not move a whole file at once: one function gets re-read against a newer
  commit while its forty neighbours stay put, which is exactly what a diff
  produces. A single per-file commit cannot express that and would be lying
  about the other forty the moment anyone did it. `port-links` reads each cited
  file **as it was at that citation's own commit**, via `git show
  <sha>:<path>`, so a file may cite as many commits as it needs to. The working
  tree is a fast path only when the citation is at the checkout's `HEAD`.
- **A citation may name several symbols**, `(ggml-impl.h:173, 179, 185, 191
  @c1d0e7a00)`; names and numbers pair up in order.
- **Where a symbol has several `#if` arms**, cite the arm this target compiles
  and say which one in the prose.
- **A citation may point at our own `harness/`** — `testing.zig` cites the C
  that captured its goldens. Those carry no upstream commit and are not asked
  for one.

The gate was negative-tested on all seven failure classes: a line number
drifted by two, a line number deleted, a renamed symbol, a missing commit, an
unknown commit, and — the two that prove the per-citation commit is real — a
citation at an older commit whose line is right *there* passing while the line
that is right at `HEAD` fails. Before it
existed the convention had rotted badly — **216 of the first 790 citations were
stale**, two pointing at the wrong file entirely, and every `Mirrors …` citation
in `blocks.zig` had never been checked at all because the first version of the
checker only looked for the word `Ports`.

**`ggml-impl.h` is not importable** — it includes `<arm_neon.h>`, whose `__mfp8` type translate-c cannot parse. `src/ggml/impl.zig` hand-writes what ported code needs from it: assertions, logging, `GGML_PAD`, the hash set, the float conversions, `struct ggml_cgraph`, and C-pointer narrowing helpers. Add to it rather than trying to widen the import.

**`make validate` is the fast check; parity is the real one.** `validate` runs formatting, the scaffold, the unit tests, the ported tests in both debug and release, `scripts/port-coverage`, `scripts/port-links`, and the graph diff. It proves the port is *consistent*, not *correct*.

Correctness has five gates, and none of them subsumes the others:

- **`make graph-diff`** builds 131 nodes through every constructor family and diffs op, shape, strides, `op_params`, and `src[]` against the C. Needs no model. **The sharpest gate for the port's actual failure mode** — a constructor writing the wrong thing — and the only one that catches a halved softmax scale. It found the real `ggml_permute` bug.
- **`make backend-ops`** runs upstream's `test-backend-ops` against our library: 21,093 op configurations, Metal against CPU, all passing. The broadest exercise, and the only one that *executes* the constructors at scale. **It cannot catch a consistently-wrong constructor** — both backends read the same `op_params` and agree. It catches crashes, assertion failures, and shapes a backend has no kernel for. `--diff` also compares output against the C reference (25,176 lines identical).
- **`make probe`** proves ported code is *on the execution path*. Green does not mean running: two definitions of a symbol in one static archive is not a link error, the linker just picks one silently. It takes **two runs**, `-Dprobe-ported` and `-Dprobe-cpu`, because the allocator fires while the graph is being planned and would mask the CPU dispatch every time.
- **`scripts/vecdot-prefix` localises a dot-product mismatch.** Not a gate — a diagnostic. It asks the shipped C kernel for every prefix of a row, so the first prefix that diverges isolates the loop iteration, and `nblk = 1` versus `nblk = 2` separates the scalar tail from the vector body. It is what turned an unguessable 1-ULP difference in `iq4_nl` into a solved equation.
- **The `_generic` dot products have no gate but their goldens.** `arch-fallback.h` renames nothing from `quants.c` on ARM and `arch/arm/quants.c` supplies every real entry point, so all 25 `ggml_vec_dot_*_generic` are exported and unreachable. `test-backend-ops` cannot see them and neither can token parity. `src/ggml/cpu/quants/golden.zig` holds the exact `f32` bits the C produces for **six** input patterns, regenerated by `scripts/vecdot-golden`. Compare on **bits**, never a tolerance.

  Two holes in that golden set were found by injection and closed. **Every `nvfp4` ARM golden was `0x00000000`** because the harness never called `ggml_cpu_init` and `GGML_CPU_UE4M3_TO_FP32` is a *table lookup* on NEON — a kernel returning zero would have passed all six. And **no pattern hit an exact tie**, so swapping round-half-to-even for round-half-away passed everything; the `ties` pattern exists to fix that. `testing.zig` now refuses a golden row that is entirely zero.
- **`make parity-cli`** runs the actual `llamazig` binary against a reference C driver, both greedy. Covers what `parity-port` structurally cannot, because it lives in the binary rather than the library: argument parsing, tokenizer flags, the sampler chain, the decode loop. Negative-tested — a `--temp` that parses but never reaches the sampler fails all six prompts.
- **`scripts/parity-port`** diffs generated tokens against the Apple-clang reference. End-to-end and **coarse** — measured, not guessed: doubling RoPE's `freq_base` passes, and so does halving the softmax scale. It catches structural faults (`ADD` as `SUB` fails all six prompts) and gross numeric ones (`freq_base = 1.0` fails all six). Never read a parity pass as "the numerics are right".

`scripts/port-links` is not on that list because it checks nothing about what the code *computes*. It keeps the port's map back to upstream honest, which is a maintenance gate rather than a correctness one.

`PLAN.md` has the full fault-injection table showing which gate catches what.

## Float contraction

**`zig cc` defaults to `-ffp-contract=on`, and so does Apple clang, but Zig
*language* code is strict.** The C fuses `a*b + c` into one FMA — one rounding
instead of two — and a literal Zig translation does not.

**The project handles this two different ways, and the difference is the
availability of an oracle.**

- **In the quantizers (`src/ggml/quants/`), the fusion is not named.** The
  sites are implicit in 5,000 lines of plain arithmetic, and guessing broke:
  fusing `makeQxQuants` fixed `q3_K` and simultaneously broke `q4_0`, the same
  function. `scripts/quants-golden` builds its reference with
  `-ffp-contract=off` and the ported code stays plain. Consequence: bytes our
  quantizers produce can differ in the last bit from a stock build — invisible
  to the deliverable, which reads model files and never writes one.
- **In the NEON kernels (`src/ggml/cpu/quants/arm/`), every fusion *is*
  named.** There the sites are explicit intrinsic calls — `vmlaq_n_f32`,
  `vfmaq_f32` — or a single `acc += a * b` statement per kernel, and
  `scripts/vecdot-prefix` pins the shape by asking the shipped kernel for every
  row prefix. Sixteen equations per kernel, not a guess. These are the live
  kernels, so their goldens are captured with contraction **on** and the port
  matches the shipped binary bit for bit.

The rule that separates the two: **name a fusion only when something can tell
you you got it wrong.** `scripts/vecdot-prefix` is that something.

## Threads in ported code

`ggml-cpu.c` is the first ported file that runs threads, and two decisions
there are worth knowing before touching `src/ggml/cpu/threading.zig`:

- **The mutex and condition variable are pthreads', not Zig's.** Zig 0.16's
  `std.Io.Mutex` and `std.Io.Condition` take an `Io` on every lock and wait,
  and a backend entered through a C ABI has none to thread through. The C's
  `ggml_mutex_lock` expands to `pthread_mutex_lock` on this platform anyway, so
  calling it directly is both the workable choice and the closer translation.
- **Zig 0.16 has no `@fence`.** `ggml_barrier` and the polling threads both
  want a standalone `atomic_thread_fence(seq_cst)`. The port uses the C's own
  thread-sanitizer fallback — a `fetchAdd` of zero with `seq_cst` — on the same
  field the C names. The memory orderings around it are the C's, verbatim;
  they are not incidental and should not be "tidied".

**Zig reserves `i1`, `i2`, `i3`, `i11`, `i12` and `i13` as integer type names,
and the C uses every one of them as a loop index.** `src/ggml/cpu/mulmat.zig`
renames them `j11`, `j12`, `j13` and so on, digit for digit, so the index
arithmetic can still be read against the C line by line. Do not renumber them.

## Where the C's own answer is unspecified

Two places, both measured rather than assumed, both excluded from the goldens
rather than matched. When the reference is unspecified, reproducing it is not
the goal:

- `quantize_row_iq4_nl_ref` **reads uninitialized memory** for an all-zero
  block. Two consecutive calls in one process return different bytes. Ours
  zeroes the buffer.
- The **1-bit split search depends on `qsort`'s ordering of equal elements**.
  macOS libc returns `31 1 2 ... 30 0` for 32 equal elements; glibc would
  differ. Ours breaks ties on index, so output depends only on input.

**A filtered test run can hide a compile error.** `zig build test -Dtest-filter=...` skips codegen for non-matching tests, so a broken test body in an unselected test passes silently. Worse, a filter that matches *nothing* reports a comfortable row of passes: `-Dtest-filter="_K dot matches the C\|widening product"` is a literal string, not an alternation, and it matched zero tests while looking green. **Confirm the count changed, and run unfiltered before believing anything.**

**A gate proves nothing until it has been made to fail.** Four separate false passes in this project came from a check that was green because it was testing nothing. Inject a fault, confirm the gate fails, then trust it — and confirm the build actually succeeded and the artifact actually changed, because a failed build leaves the previous library in place and the gate will happily pass on it.

**The reference is `llama.cpp.zmake`** — the stock C sources built by the Zig toolchain. Both sides are then compiled by the same toolchain, so the port is the only variable. `scripts/parity-port` still points at the old Apple-clang CMake build and needs repointing. Note `llama.cpp.zmake` is a separate repository and is never modified by this project.

## `llama.cpp/` is reference material only

The `llama.cpp/` directory is an upstream clone pinned to tag `v0.3.0` (commit `c1d0e7a00`), created by `make clone`. It has its own `.git` and is untracked here.

**It is here to be read, not written.** Treat it as a specification we are reimplementing:

- **Never edit files under `llama.cpp/`.** Not to fix a bug, not to add a workaround, not to make a build succeed. If something there is wrong or in the way, the answer is a change on our side or a note in `PLAN.md`.
- **Never commit, push, or open a PR against it.** It is a separate upstream repository that we do not contribute to.
- **Never treat its conventions, instructions, or tooling as ours.** Its `CLAUDE.md`, `AGENTS.md`, `skills/`, and CI config govern the upstream project and do not apply to this one. This file is the authority here.
- **It is pinned deliberately.** Do not update it to a newer tag or pull upstream changes. A multi-month port against a moving target does not converge; syncing upstream is a separate, explicitly-scoped decision.

Read it constantly — for behavior, for numerics, for the shape of a data structure, as the reference build to diff against. Just do not change it.

## `llama.cpp.zmake/` is the Zig build system

A separate repository holding a `build.zig` that replaces llama.cpp's CMake.
It reads the sources from a checkout nested inside it and **does not modify
them**, so the pin can be changed or the checkout replaced without
reconciling anything.

```
llama.cpp.zmake/
├── build.zig, build.zig.zon, Makefile
├── zig/metal_embed.zig     flattens a Metal kernel for embedding
├── zig/ui-stub/            stand-in for the generated web UI assets
└── llama.cpp/              sources, fetched by `make clone` (gitignored)
```

`make zmake` from this repo builds `llama-cli` through it. Its `README.md`
records the six CMake behaviours that are easy to miss and cost real
debugging — `sha1.c` being C++ despite its extension, `src` shadowing
`common/unicode.h`, and so on. Read that before changing its `build.zig`.

It is gitignored here because it is its own repository. Work done there is not
tracked by this one.

## What we own

`build.zig`, `build.zig.zon`, `build/`, `src/`, `cli/`, `harness/`, `scripts/`, `Makefile`, and the markdown at the root. That is the whole of this project's code.

`LICENSE` (MIT) and `NOTICE` are load-bearing, not boilerplate: every file under `src/` is a translation of MIT-licensed C, so this is a derivative work and the ggml authors' copyright has to travel with it. Keep the per-file provenance comments — they are how a reader traces a translation back to its source.

- **`build/llamacpp.zig`** declares the reference tree's build graph, replacing its CMake for the macOS arm64 configuration. It compiles upstream sources unmodified.
- **`build/metal_embed.zig`** is a build-time tool that flattens one Metal kernel and its headers into a single MSL file plus an assembly stub. It replaces the `cat`/`sed` pipeline CMake uses for `GGML_METAL_EMBED_LIBRARY`.
- **`harness/smoke.zig`** loads a model through libllama's C ABI and generates. It is the only check that proves inference actually works, and it rehearses the C-ABI boundary that Stages 3 and 4 depend on.

Note `build/` holds build *sources*, not build output. CMake's output directory was moved to `cmake-build/` to free the name.

- **`cli/`** builds `llama-cli`, our replacement for upstream's binary of that name — same name so an existing command line runs unchanged against it, but it is our Zig binary, not upstream's linked against our library. `args.zig` parses upstream's flag surface, `session.zig` loads a model and generates, `chat.zig` renders chat templates, `main.zig` stays thin. `c.zig` holds the single `@cImport` of `llama.h` — two of them produce two incompatible `*llama_model`, and the error says only that `*cimport.struct_llama_model` will not coerce to `*cimport.struct_llama_model`. `upstream_flags.zig` is a generated inventory of every flag upstream accepts, used only to tell "you typed a real flag we have not got to yet" apart from "you made a typo".

**Chat templates are Jinja, and trust runs the other way from upstream's.** `vibe-jinja` is the one permitted dependency, wired into `cli/` only. Three things about it are worth knowing before touching `cli/chat.zig`:

- **Template literals are trusted; everything an expression emits is not.** Upstream marks strings that came *from* input (`common/jinja/README.md`); we mark the complement, so trust is the closed set the template author wrote rather than the open set we remembered to taint. A filter chain, a `set`, a loop variable — all reach output through an expression and all come out untrusted with nothing tracking them. `Session.tokenizeSegments` then sets `parse_special` per run. Measured: the attack text tokenizes to 7 tokens *including the real `<|im_start|>`* with it on, and 17 harmless ones with it off.
- **The marking rides on `Environment.finalize`, and the bytecode VM ignores it.** `applyFinalize` has exactly one call site, `compiler.zig:642`, on the AST path. `jinja.compiler.compile` picks the bytecode VM whenever the template allows it, which silently drops the marking and returns a prompt that looks correct and is entirely trusted. `chat.zig` calls `Compiler.compile(template, false)` for that reason. Found by tests failing, not by reading.
- **Marking is idempotent on purpose.** Qwen3.5 does `{% set c = render_content(...) %}{{ c }}`, so a macro's already-marked output passes through `mark` again; wrapping twice nested the sentinels and swallowed the macro body's own literal text into an untrusted run.

**The engine has no strict-parse mode.** `{{ unclosed`, `{% bogusstatement %}`, a `for` with no `endfor` and `{{ 1 + }}` each render as **empty output and report success**. `chat.zig` rejects a render that produced nothing from a non-empty conversation; that is the only thing standing between a broken template and a silently empty prompt.

**`-cnv` holds a conversation, and re-renders the whole history every turn.** A template decides for itself where the system prompt goes and how a turn is framed, so appending to the previous render is not safe in general. `Session.generateTurn` diffs the new prompt against the decoded tokens and keeps their common prefix, so the re-render costs tokenization rather than decoding.

**`llama_memory_seq_rm` returns a bool, and on Qwen3.5 it returns false.** Recurrent and hybrid models cannot drop a range from the middle of the cache; the refusal is the only signal, and discarding it decodes the next turn on top of a stale cache at wrong positions. The symptom was the model emitting `<|im_end|>` as literal text mid-reply, which reads as a template bug and is not one. `generateTurn` checks the result and clears the cache instead. Found by a three-turn conversation, not by any unit test.

**Chat templating is opt-in, where upstream templates by default.** Raw completion is what `make port` and `make ref` diff against each other, so defaulting it on would change the one gate that covers the whole binary. Any chat flag turns it on; `--no-jinja` vetoes.

The greeting placeholders this repo started from (`src/base.zig`, `cli/client.zig`) are gone.

## Build state

All steps pass.

```sh
zig build            # the llamazig scaffold -> zig-out/bin/llama-cli (fast, ~0.2s)
zig build lib        # static library        -> zig-out/lib/libllamazig.a
zig build cli        # CLI                   -> zig-out/bin/llama-cli (links libllama)
zig build run        # run the CLI; args after `--`
zig build test       # unit tests; -Dtest-filter="..." narrows the run
zig build test-cli   # the CLI's tests only (~6s, vs ~1m for the library's)
zig build docs       # autodoc               -> zig-out/docs/

zig build reference  # llama.cpp -> zig-out/lib/{libggml.a, libllama.a}
zig build smoke      # load a model and generate; args after `--`
```

The stage-1 builds go through CMake and live in a separate tree:

```sh
make cmake           # fetch CMake into .tools/ -- not installed on this machine
make buildmacos      # control build, Apple clang -> cmake-build/apple/
make buildmacos-zig  # same sources via zig cc    -> cmake-build/zig/
make parity          # diff their token streams; the correctness gate
```

The reference build is deliberately **not** part of the default step — it is ~130k lines of C++ and takes ~16s in release, versus 0.2s for the scaffold.

```sh
zig build smoke --release=fast -- Qwen3.5-2B-Q4_K_M.gguf "The capital of France is" 24
```

Remaining Stage 0 item: **`llama.cpp/` is an untracked nested clone, not yet a submodule.** `main` still has no commits, so pinning it is free now and expensive later. `*.gguf` is ignored.

## Toolchain

Zig **0.16.0**, declared as `minimum_zig_version` in `build.zig.zon` and what is installed. The code uses the 0.16 std APIs — `std.process.Init` as the `main` parameter, `std.Io` threaded explicitly through constructors, `std.Io.Writer` rather than the old writer interfaces. Do not fall back to pre-0.16 idioms.

`vibe-jinja/` is a sibling checkout like `llama.cpp/` — its own repository, untracked here, carrying a Zig 0.16 migration on top of upstream's 0.15.2 release. `build.zig.zon` takes it as a path dependency.

The library must never gain external dependencies. This is a hard constraint, not a preference: the point of the port is that `zig build` alone produces the binary, and anything linking `libllamazig` inherits nothing.

**One exception has been granted, and only one:** `gremlin-labs/vibe-jinja` (pure Zig, MIT) for chat templates, confined to `cli/`. See `PLAN.md` Decisions 25 and 27. Pin an exact commit, never a branch. Do not add a second dependency without the same explicit grant.

## Commands

```sh
make build        # zig build
make dist         # zig build --release=fast
make clone        # clone llama.cpp and check out v0.3.0
make buildmacos   # reference CMake build of llama.cpp for arm64 macOS
make buildios     # reference CMake build for iOS
make qwen35       # download a Qwen3.5-2B GGUF (also qwen35xs, qwen35xl)
make port         # our ported CLI
make ref          # the same completion loop against the stock C libraries
make cli          # our CLI in interactive conversation mode
make ref-chat     # upstream's llama-cli REPL (different thing -- see below)
```

**`make port` and `make ref` must produce identical output.** They run the same
tokenize/decode/sample loop over the same prompt; `ref` links the stock C
libraries, `port` links the port. If they ever diverge, that is a real porting
bug and not a configuration difference.

Both take `MODEL`, `PROMPT`, `NPRED`, `TEMP`, `SEED`, and `ARGS`:

```sh
make port TEMP=0 PROMPT="def fibonacci(n):"   # greedy
make ref  TEMP=0 PROMPT="def fibonacci(n):"
make port SEED=$RANDOM                         # sample freely
```

**Both targets print the same timings line**, `[ Prompt: N t/s | Generation: N
t/s ]`, on stdout. `harness/raw_completion.c` runs the same warmup our CLI does
— throwaway decode, memory clear, `llama_perf_context_reset` — because without
it the first `llama_decode` carries Metal pipeline compilation and that lands in
the prompt counter.

**Do not read the two throughput figures as a comparison of the port against
the C.** Two confounds: the prompt figure swings by over 100 t/s run to run on
this machine, and `make ref` links the *Apple-clang* CMake libraries, so a
compiler difference is folded in. For a like-for-like build comparison use
`make parity`, which diffs two builds of the same C.

**`TEMP` and `SEED` are shared and `SEED` is fixed at 42**, where upstream's
default seed is random. A pair of targets meant to be diffed has to be
reproducible — and at temp 0.80 an unlucky seed sends a 2B model into a
repetition loop, which reads as a port bug and is not one.

**`make ref-chat` is a different thing and will not match.** It runs upstream's
`llama-cli`, which has **no raw-completion mode**: its `-st` is a single *turn
of a conversation*, so it applies the model's chat template, prints a banner and
a slash-command list, and answers as an assistant. The model is being asked a
different question. That is why `scripts/parity-cli` links its own C driver
(`harness/raw_completion.c`, shared with `make ref`) rather than calling
`llama-cli`.

`make ref-chat` needs `make zmake` first. The older `make cli` target runs
upstream's `llama-cli` from the *CMake* reference build against a model path
this repository does not contain; `make ref-chat` supersedes it.

`make buildmacos` builds the **reference implementation**, not our code. Its flag set is the Stage 1 target configuration: static, Metal on with the library embedded, Accelerate on, BLAS and OpenMP off, no curl. It is what a ported binary gets diffed against for correctness. When `build.zig` replaces CMake in Stage 2, these become the hardcoded macOS defaults rather than flags a caller passes.

Note that `make buildmacos` uses CMake's `-G Xcode` generator, which drives `xcodebuild` and therefore selects Apple clang regardless of `CMAKE_C_COMPILER`. Stage 1 — compiling llama.cpp with `zig cc` — will need `-G Ninja` instead. See `PLAN.md`.

## Code conventions

Followed consistently across `src/` and `cli/`; match them. They are the house style for ported code too, which matters: transliterated C and C++ will not naturally come out looking like this, and making it do so is part of the work.

- **Barrel per directory.** Each source directory has a `module.zig` re-exporting every public type. Siblings import through the barrel (`@import("module.zig")`), never each other directly. Consumers import the library as `@import("llamazig")`.
- **`src/root.zig` is docs-only.** Zig's autodoc cannot root a module at `module.zig`, so `root.zig` exists purely as the `docs` step's root and re-exports the barrel. Don't put code in it.
- **Allocator first, and the owner frees.** `init(allocator, ...)` stores the allocator as a field; helpers reach for `self.allocator` rather than taking one again. Whatever the struct's allocator produced, the struct's `deinit` frees. Slices handed to callers are borrowed and documented as invalid after the next call or `deinit`.
- **`init` returns an error union even when it cannot fail** — noted in the doc comment as "reserved for the lifecycle convention".
- **Doc comments carry a `Parameters:` list and a `Return:` line** on every public function, including what the return borrows and how long it stays valid. Struct-level `///` explains why the type exists, not just what it is.
- **Tests live at the bottom of the file they cover**, under a `// ---` / `// Unit Tests` banner, opening with `test { std.testing.refAllDecls(...) }`. Shared fixtures (`test_subject`, `test_greeting`) live in `src/module.zig` so every file asserts against the same values.
- **Executables stay thin.** `cli/main.zig` only unpacks `std.process.Init` and delegates to `cli.Client`, so all behavior stays reachable from unit tests.
- **`std.testing.allocator` is the leak check.** Tests that call an allocating method more than once exist specifically to prove the previous allocation was released.

## Porting notes

For when Stages 3 and 4 begin:

- **`zig translate-c` output is a draft to read, never a deliverable to commit.** It produces `c_int`-everywhere, pointer-arithmetic-heavy Zig that violates every convention above.
- **Correctness is measured against the reference build,** not against reading the code. Diff token output from the ported binary and the `make buildmacos` binary on the same model, prompt, and seed.
- **ggml's macro-heavy headers hold real logic in the preprocessor.** `ggml-impl.h`, `ggml-common.h`, and `simd-mappings.h` become `comptime` functions and generics — a rewrite, not a translation.
- **The Metal kernels under `ggml/src/ggml-metal/kernels/` are never ported.** They are Metal Shading Language compiled by the GPU driver; they travel as embedded data.
- **Zig has no ACLE intrinsics, and mostly does not need them.** `src/ggml/cpu/quants/arm/neon.zig` reproduces the 82 the NEON kernels use; all but one are a line of `@Vector` arithmetic. Three traps, each measured rather than reasoned about: **NEON integer arithmetic wraps** (so every op there uses `+%`, `*%`); **`vaddvq_f32` is a pairwise reduction**, not the ordered one `@reduce(.Add, ...)` gives; and **`vshlq_u8` shifts *right* on a negative amount**, which `q2_0` depends on.
- **`vdotq_s32` has no portable equivalent and does not need one.** It is integer, so the widening-multiply form is bit-identical rather than merely close. Measured cost: none detectable. Generation is 222-234 t/s either side of the port across three runs each, and the prompt figure varies by more than 100 t/s run to run — so the honest claim is "no measurable change", not a number. Metal carries the model; the quantized CPU path is barely on the critical path here.
- **Only a third of `arch/arm/quants.c` compiles here.** 1,556 of 4,319 lines; the rest is behind `__ARM_FEATURE_SVE` and `__ARM_FEATURE_MATMUL_INT8`. Measure before estimating a file in that directory — `zig cc -E` plus the linemarkers gives the live-line count.

## Building the reference tree

Facts established by getting `zig build reference` working. They are easy to
rediscover the hard way, so they are recorded here.

- **`zig cc`, `zig c++`, and Obj-C all work** against the macOS SDK under Zig 0.16.0, including `-framework Foundation` and libc++.
- **Do not pass `-fobjc-arc`.** Upstream's `.m` files use manual reference counting and bridge freely between `void *` and Obj-C object pointers; ARC rejects them outright.
- **`sanitize_c` must be `.off`.** Zig enables the C sanitizers in Debug, and upstream is not UBSan-clean — it aborts in `llama-graph.cpp` on `applying non-zero offset to null pointer`. Reference code compiles as shipped; we do not fix it.
- **The Metal shaders need no `xcrun metal`.** With `GGML_METAL_EMBED_LIBRARY`, the `.metal` *source* is flattened and `.incbin`-ed into a `__DATA,__ggml_metallib` section, and the driver compiles it at load time. `build/metal_embed.zig` does the flattening. Never enable the non-embed path: it requires `xcrun -sdk macosx metal`, which Zig can never replace.
- **CMake's `-G Xcode` cannot drive `zig cc`.** It shells out to `xcodebuild`, which picks Apple clang and ignores `CMAKE_C_COMPILER`. Moot now that `build.zig` owns the build.
- **Metal is reproducible run-to-run.** Three greedy runs on the same model and prompt produced token-identical output, so the token-parity gate in `PLAN.md` is usable as written.
- **CMake is not on PATH and `/usr/local` is root-owned.** `make cmake` fetches 4.4.3 into `.tools/`. The Makefile's `CMAKE` variable prefers that over PATH. `ninja` is still absent, which is why the generator is `-G "Unix Makefiles"` — `make` is present.
- **The parity gate works and passes.** `scripts/parity` diffs the Apple-clang and zig-cc `llama-cli` binaries at `--temp 0` with a fixed seed. 4/4 prompts token-identical. Use it after any change that could affect arithmetic.
- **`zig cc` and Apple clang do not select the same ARM features.** CMake's probe through `zig cc` reports `HAVE_MATMUL_INT8 - Failed` and `HAVE_SVE - Failed` where Apple clang may not. Output is identical regardless, so this costs throughput rather than correctness — but the two builds are not running the same quant kernels. Revisit when Stage 3 ports `arch/arm/`.

## macOS SDK resolution

`build.zig` has an `Xcode` helper that resolves the Apple Silicon macOS SDK via `std.zig.system.darwin.getSdk` and adds it as a framework path to the library, and separately to the docs library. It runs only when `builtin.os.tag == .macos`. If linking against system frameworks fails, that resolution is where to look.
