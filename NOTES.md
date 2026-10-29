# NOTES.md

The record of how the port was actually done: what each completed step cost,
what each gate caught, and the false passes that had to be found by injection.

`PLAN.md` is the forward plan — what is decided and what comes next. This file
is what stands behind it. Nothing here is a to-do; it is all finished work,
kept because the lessons are the expensive part and a plan that still carried
them would be unreadable.

Written against `llama.cpp` at tag **v0.3.0** (commit `c1d0e7a00`), Zig
**0.16.0**, macOS arm64.

---

## The incremental-swap trial: inconclusive, parked

**Result: the mechanism links but does not work, and the reason is not
understood.** Recorded in full because the failure mode is more instructive than
the idea was.

**What was tried.** All 320 ported `ggml.c` symbols renamed via `-D` on `ggml.c`
alone, with `src/ggml/` imported into the ggml module. Every checkpoint looked
right:

- Compiled clean, no duplicate symbols.
- `nm` showed our Zig defining the real names, the C originals renamed to `_c`.
- `scripts/parity-port`: **6/6 prompts identical**.

**Why that was wrong.** Fault injection, which is the only thing that
distinguishes a working gate from an inert one:

| Fault | Expected | Observed |
|---|---|---|
| `ggml_aligned_malloc` → panic | crash | **crashed** — some swapped symbols do run |
| `ggml_fp16_to_fp32` × 1.0001 | tokens change | identical |
| soft_max scale × 1.01 | tokens change | identical |
| `ggml_rope_ext` freq_base × 0.5, 252 call sites | garbage output | **identical** |
| `ggml_rope_ext` → `@panic` | crash | **never fired** |

Halving the RoPE frequency base cannot produce identical output. Our
`ggml_rope_ext` is not executing, although `libggml.a` contains it, `libllama.a`
references it, and no other object defines that name. The panic string is
present in the archive and **absent from the linked binary**.

**Two lessons that outlive the experiment:**

1. **A parity pass proves nothing until it has been made to fail.** Four of the
   five faults above were too weak to flip a greedy argmax; only the
   unabsorbable one exposed the problem. Every future gate needs a
   deliberately-broken run before it is trusted.
2. **Green does not mean running.** A symbol can be defined, referenced, and
   linked without being executed. Verification therefore needs a *positive*
   check that ported code is on the path — the `@panic` probe, kept as a
   standing test — not only a comparison of outputs.

**Parked, not abandoned.** For the all-or-nothing swap the ambiguity disappears:
with `ggml.c` removed from the build entirely there is no other definition to
resolve to. The idea is worth revisiting only if the linker behaviour is
understood.

---

---

## The old rationale, for the record

This plan has said throughout — and `CLAUDE.md` still says — that a translation
unit must be ported completely before any of it can be tested, because a
half-ported file cannot link beside the original. **That appears to be wrong**,
and the consequence is large enough to state plainly.

The C preprocessor can rename a symbol from the command line, without editing
the file. Compiling `ggml.c` with `-Dggml_add=ggml_add_c` renames both the
definition *and* every internal call within that translation unit, leaving the
external name free for a ported Zig version. Verified on a minimal case:

```
tu.c:    int helper(int x) { return x * 2; }
         int entry(int x)  { return helper(x) + 1; }
mine.c:  int helper(int x) { return 999; }

zig cc -c tu.c -Dhelper=helper_c   # rename inside the C only
=> entry(10) = 21     internal call still reaches the original
=> helper(10) = 999   external callers reach ours
```

Applied to `ggml.c`: compile it with one `-D` per ported symbol, link both
objects, and libllama calls our Zig while `ggml.c`'s internals stay
self-consistent. **This does not edit anything under `llama.cpp/`** — the
renames are build flags, which is exactly what `build/llamacpp.zig` is for.

**What it would change:**

- The 320 symbols ported today could be parity-checked **now**, instead of after
  all 381 land.
- A parity failure becomes bisectable: halve the `-D` list and rebuild.
- The pressure to finish `ggml.c` before learning anything disappears, which
  weakens the reasoning behind Decision 28.

**What it does not do.** Internal calls inside `ggml.c` keep reaching the C
implementations, so a ported function is only exercised on paths that enter from
outside the translation unit. Coverage is real but partial, and full coverage
still arrives only at the complete swap. It is a diagnostic and staging
mechanism, not a substitute for finishing.

**Unverified at scale.** The technique is proven on three lines of C. Applying
it to 320 symbols could hit collisions — a name appearing in a context the
preprocessor should not touch, or a macro in `ggml.h` expanding unexpectedly
inside the renamed translation unit.

**Decision 34: trial it, and do that next.** The shape of the trial:

1. Pick five ported symbols with **no internal callers inside `ggml.c`**, so a
   rename cannot disturb the file's own behaviour. `ggml_status_to_string`,
   `ggml_type_name`, `ggml_version`, `ggml_commit`, `ggml_guid_matches` are
   candidates; confirm with a call-graph check rather than by eye.
2. Add the `-D` renames to `build/llamacpp.zig` for exactly those five.
3. Link our Zig definitions alongside.
4. Run `scripts/parity-port`. Byte-identical output means the mechanism holds.

**What it settles.** If it works, the 320 already-ported symbols can be
parity-checked in batches rather than waiting for all 381, a failure becomes
bisectable by halving the `-D` list, and Decision 28's ordering loosens. If it
fails, an hour is spent and the plan is unchanged.

**What it does not settle.** Whether the mechanism scales to symbols that *do*
have internal callers — the majority. Those are the interesting case, and the
trial deliberately avoids them so that a failure is unambiguous. A second round
would extend to one such symbol before anything is planned around it.

---

---

## The graph diff

Decision 31 puts this ahead of `test-backend-ops`, because it targets the code
that exists today rather than the code that comes later.

**What it does.** Build the same graph twice — once against stock ggml, once
against ported ggml — and dump every node's op, shape, strides, `op_params` and
`src[]` wiring as text. Diff the dumps. A constructor bug surfaces as a named
node with a named field differing, *before* any arithmetic runs, so there is
nothing to bisect and nothing to interpret.

**Synthetic, per Decision 33.** The driver calls each ported constructor
directly with fixed inputs rather than building a model's graph. That trades
realism for breadth: it reaches `conv_3d`, `ssm_*`, `rwkv_*`, `win_part` and
everything else no Qwen3.5 graph would touch, and needs no model file. It cannot
catch a shape that is individually right but wrong in context — that is what
whole-model parity is for.

**Scoped to what computes, per Decision 35.** Roughly 150 constructors: anything
that derives a shape or packs `op_params`. Pure getters and predicates are left
out — `ggml_type_name` reads a table already checksummed against the C, and
`ggml_nelements` multiplies four numbers. Extend the set only if a bug escapes
it, which is the signal that the line was drawn wrongly.

Every call site is hand-written, with shapes valid enough to satisfy each
constructor's own assertions. That is the cost, and it is why the scope is not
all 320.

This is what catches the errors the ported code is actually prone to: RoPE's
16-slot `op_params` layout, `set_rows`' deliberately odd `src[]` order, a shape
formula off by a stride. Whole-model token parity would notice all of these and
tell you only that the model says something different.

Mechanically it reuses `scripts/parity-port`'s pattern: one driver, built twice
against different libraries. Decision 33 settles what graph it should build.

### Built, and it earned its keep immediately

`harness/graph_dump.c` calls 131 constructors across every family;
`scripts/graph-diff` links it against `libggml.a` and against
`libggml-ported.a` and diffs the dumps. `make graph-diff` runs it, and
`make validate` runs it too when a reference build is present.

**`libggml-ported.a` is the whole point of the setup.** The first run reported
131 nodes identical and meant nothing: the driver was linked against
`libggml.a`, which still contains `ggml.c`, so it was comparing C to C. The
build now produces a separate archive from `src/ggml/ported.zig` alone. Any
future comparison harness needs the same care — *what is actually linked* is
the question, not what the step is named.

**It found a real bug on its first honest run.** `ggml_permute` had matching
`ne` and matching `nb` and a missing `op_params` write; the C records the four
axes there after applying them to the strides. The port applied them and did
not record them. Backends read that field back, so the node was structurally
plausible and wrong. Nothing else in the harness — not the 21 unit tests, not
`parity-port`'s 6/6, not the smoke test — had any chance of seeing it, because
Qwen3.5 does not exercise the path that reads it back.

**Negative-tested, per the rule below.** Adding `0.0001` to one RoPE
`beta_fast` makes it fail and names both `ggml_rope_ext` and
`ggml_rope_ext_back`. For scale: whole-model token parity absorbed a *1%*
attention-scale change during the swap trial without a flicker. The graph diff
sees four orders of magnitude finer, because it compares the graph rather than
what the model says about it.

**What it still cannot see.** Anything past construction — every kernel, every
numeric path. A constructor can be perfect in all 131 slots and the arithmetic
underneath it still wrong. That gap is exactly `test-backend-ops`, and this
tool does not shrink it.

---

---

## Positive-execution probes

Green does not mean running. `-Dprobe-ported` compiles a `@panic` into
`ggml_gallocr_alloc_graph`; `scripts/probe-ported` builds twice and requires
the marker to be *absent* with the probe off and *present* with it on.
`make probe` runs it.

**It works in both directions** — restoring `ggml-alloc.c` to the build makes
it fail, which is the case that matters, since that is precisely the bypass it
exists to detect.

**The linker fact that makes this necessary.** Two definitions of a symbol in
one static archive is not an error. The linker picks a member and says nothing.
So a ported file can sit in the build, contribute no code, and every gate stays
green. Every future swap needs a probe before its parity result is worth
anything.

---

---

## Diagnostics for the first swap

Decision 28 means 381 symbols land in one commit. If `scripts/parity-port` then
reports different tokens there is no bisect, and the only signal is that the
model says something else. Decision 29 builds the tool that turns that into a
localised fault.

**`test-backend-ops`** (`llama.cpp/tests/test-backend-ops.cpp`) asserts that
multiple backends computing the same ggml op agree, op by op, with a gradient
mode as well. Run against our ported ggml it names the operation that broke
instead of naming the model. `test-rope` and `test-quantize-fns` are narrower
versions of the same idea, and `llama-eval-callback` dumps intermediate tensors
for a tensor-by-tensor diff.

**It has to be built here, against our libraries.** Reusing a prebuilt binary
does not work, and the reason is worth stating because it is easy to assume
otherwise:

- `llama.cpp.zmake` does not build it — that repository builds `llama-cli` and
  the libraries, nothing else — and adding a target there is ruled out by
  Decision 18.
- The binary in `cmake-build/zig/` links **stock** ggml statically. Running it
  exercises upstream's code, not ours. A test that cannot see the ported code
  cannot find a bug in it.

So this repo's `build.zig` compiles the upstream C++ test against our
`libggml.a`. That makes it the first place here where upstream C++ is built
against ported Zig — a useful rehearsal in itself for Stage 4.

**Timing.** Before the swap, not after. It is cheap insurance against the one
scenario where the port has no diagnostic path, and until `ggml.c` is swapped in
it is also the only per-op coverage available for the 320 symbols that have
none.

**Scope.** Decision 29 records that this is not core to the port. It exists to
validate, and should not grow beyond that.

### What `test-backend-ops` does not catch

Checked against the source rather than assumed, because the answer matters:
`test-backend-ops` calls `build_graph(ctx)` **once** and evaluates the result on
two backends (`eval(backend1, backend2, ...)`, line 1316). Both backends
therefore see the *same* graph — built by whichever ggml is linked in.

So if a ported **op constructor** computes a wrong shape, packs `op_params` in
the wrong slot, or wires `src[]` wrongly, both backends compute the same wrong
thing, agree with each other, and **the test passes**.

That matters here more than it would in most projects, because **the 320
symbols ported so far are almost entirely constructors** — shape arithmetic,
`op_params` packing, assertions. `test-backend-ops` is aimed at the *kernels*,
which are Stage 3 steps 3-5. Step 3 is under way; see "Porting `ggml-quants.c`".

It is still worth building: it is exactly the right tool for those later steps,
and it is the only per-op coverage available for anything. But it is not the
diagnostic for the risk the port carries today, and treating it as one would be
a false sense of safety. `make graph-diff` is what turned out to be.

---

---

## The CLI, built now

Decision 26 moves `cli/` out of Stage 4's tail and into current work, alongside
the port rather than after it. Nothing blocks this: libllama's C ABI is stable
and already builds, so the CLI links against the library as it is today —
mostly C, progressively more Zig — and keeps working throughout.

**What it buys.** The argument surface gets validated against upstream now
rather than at the end, when "flag-compatible" turning out to be harder than
assumed would be expensive. And the binary-level gate is ready the moment
`ggml.c` swaps in, instead of being written afterwards.

**What it does not fix.** The 320 ported `ggml.c` symbols still have no parity
check, because `ggml.c` is not swapped in. A CLI does not change that. It
changes only how quickly the check can run once it becomes possible.

**Shape.** `cli/` links our libllama, mimics upstream's argument set for what we
support (Decision 20), rejects the rest (Decision 22), and uses `zigjinja` for
chat templates (Decision 25). Written to the conventions in `CLAUDE.md`.

**Two stages, per Decision 30.** One-shot first — prompt in, tokens out, exit —
which is everything the parity gate needs and unblocks it soonest. Interactive
conversation mode follows, and lands **before the port is called complete**,
because upstream's `llama-cli` is interactive by default and that is how people
actually use it. Upstream spends 1,326 lines of C++ on the REPL, slash commands,
tab completion and multi-line input; the Zig equivalent is smaller but not
trivial.

Until interactive mode exists, running without `-st` should say so plainly
rather than behaving differently from upstream in silence (Decision 22's
principle, applied to a mode rather than a flag).

**Sequencing.** Decision 28 puts `ggml.c` first. The CLI starts once the graph
machinery is done and the swap is being verified, so the two do not compete for
attention at the point where the port's first real gate is finally reachable.

### One-shot: done

`cli/` is four files and about 900 lines:

- `args.zig` -- the flag surface, defaults taken from `common/common.h` so
  omitting a flag behaves as omitting it upstream.
- `session.zig` -- model, context, sampler chain, decode loop.
- `main.zig` -- thin, per the conventions.
- `upstream_flags.zig` -- a generated inventory of all 525 flag spellings
  upstream accepts, used *only* for diagnostics.

`make parity-cli`: **6/6 prompts token-identical** against a greedy C reference
driver. Negative-tested -- a `--temp` that parses but never reaches the sampler
fails all six.

**Decision 22, made concrete.** An unsupported flag is refused with exit 1, and
the message distinguishes two cases a user cannot otherwise tell apart:
`--mirostat` is "a llama.cpp flag that llamazig does not support yet", while
`--mirostatt` is "unknown flag". The first says the command line is right and
the port is behind, which is true and worth saying. That is what
`upstream_flags.zig` exists for; a unit test asserts no flag appears in both
lists.

`-cnv`, `-i`, `-if` and `-mli` are refused for the same reason: they ask for
interactive mode, which does not exist here, and accepting one while quietly
running one-shot would hand the user a plausible answer to a different
question.

**`-st` was wrongly in that list, and that was a real defect.** The two
upstream binaries disagree about how to ask for one shot:

- `tools/cli`, the `llama-cli` this repo builds, takes **`-st`** and *rejects*
  `-no-cnv`.
- `tools/main` takes **`-no-cnv`** and has no `-st`.

We accepted `-no-cnv` and refused `-st` -- exactly inverted from the binary a
user is most likely to have. A working upstream command line failed against us,
which is the opposite of what Decision 20 promises. All four spellings are
accepted now.

It also had Decision 30 backwards. That decision says *"running without `-st`
should say so plainly rather than behaving differently from upstream in
silence"*; the implementation refused `-st` and silently one-shot without it.
Now the absence of any one-shot flag prints a note to stderr -- stderr so it
cannot land in a redirected completion, and so `parity-cli` is unaffected.

### Two things the sampler chain got wrong before it got right

Worth recording, because both looked correct and neither would have failed a
test:

- **A greedy short-circuit is wrong.** `--temp 0` invites "add
  `llama_sampler_init_greedy` and skip the rest". Upstream has no such branch:
  `llama_sampler_temp_impl` (llama-sampler.cpp:270) does the argmax at
  `temp <= 0` *after* the penalty sampler has altered the logits. Short-
  circuiting would make `--temp 0 --repeat-penalty 1.1` choose a different
  token, on a path where both look right.
- **`min_keep` is 0, not 1.** It is the floor on how many candidates a
  truncating sampler must leave. Passing 1 changes what top-p and min-p do at
  their edges. The plausible-looking value is the wrong one.

Chain order is the behaviour, and it comes from the default `params.samplers`
list in `common/common.h`, not from `sampling.cpp` reading top to bottom.

### Timings

`--show-timings` / `--no-show-timings`, default on, printing
`[ Prompt: X t/s | Generation: Y t/s ]` -- the same format and the same default
as `cli-context.cpp:651`, so a script scraping upstream's output reads ours.

Two things had to line up for the numbers to be real:

- **`llama_context_default_params()` sets `no_perf = true`**, so the counters
  `llama_perf_context` reads are not collected at all. The first version
  printed a well-formatted `0.0 t/s | 0.0 t/s`. Collection is now tied to the
  flag: upstream splits it (`--perf` collects, `--show-timings` displays) but we
  do not take `--perf`, and there is no reason to pay for counters we will not
  print.
- **The line goes to stdout**, as upstream's does, so a redirect captures it
  with the completion. `scripts/parity-cli` therefore passes
  `--no-show-timings`; without it the gate would compare a timings line against
  a reference driver that has none, and fail for a reason that is not a porting
  bug.

### Warmup, and what the timings are worth

`--no-warmup` was parsed and **ignored** -- a flag accepted and silently
dropped, which is exactly what Decision 22 forbids. It now does what upstream's
`common_init_from_params` does (common.cpp:1511): one throwaway decode, clear
the KV cache, synchronise, then `llama_perf_context_reset`.

**That last call is the point.** Warming up and not resetting the counters
would fold the warmup's own time into the measurement and report throughput
*worse* than the truth -- worse than not warming up at all.

Measured on Qwen3.5-2B Q4_K_M, six runs each:

| Prompt | without warmup | with warmup |
| --- | --- | --- |
| 5 tokens | ~400 t/s, 1.8x spread | ~540 t/s, 1.6x spread |
| ~150 tokens | 3538-4498 t/s, first run an outlier | 4537-4703 t/s, 3.7% spread |
| generation, 128 tokens | - | 224-228 t/s, 1.8% spread |

So warmup raises the mean and removes the first-run outlier, but **it cannot
rescue a short prompt**: at five tokens the eval is a few milliseconds and the
rate stays too noisy to compare anything. For a real measurement use a prompt
of at least ~100 tokens and generate at least ~100.

This matters beyond the CLI, because prompt throughput is how the kernel work
in Stage 3 steps 4 and 5 will be judged. A 2x swing that turned out to be noise
is a good reminder that a benchmark needs its variance measured before its mean
is believed.

Upstream also re-seeds its samplers after warmup. Ours does not need to --
nothing in the warmup path samples -- and that was checked rather than assumed:
seeded output at `--temp 0.8 -s 42` is identical with and without it.

### Still owed, at the time — since delivered

Interactive conversation (Decision 30's second stage) and chat templates
(Decision 25) were both outstanding when the one-shot CLI landed. Both are
built; see "Chat templates" below.

---

---

## The `ggml.c` swap

All 381 symbols are ported and the file is out of the build. `scripts/port-coverage`
derives the contract by compiling `ggml.c` and comparing symbol lists, so
"381/381" is a name-level check only; what actually proves completeness is that
the library links with nothing compiling `ggml.c`.

**Verified, in this order:**

- `ar`/`nm` confirm every ggml symbol resolves from `libggml_zcu.o`, the Zig
  unit, and that **no ggml symbol is defined twice** anywhere in the archive.
  That second check is the one that matters: a duplicate would not be a link
  error, just a silent choice.
- The smoke harness loads Qwen3.5-2B and generates coherent text, compiling
  Metal kernels and running SSM ops through the ported graph code.
- `scripts/parity-port`: 6/6 prompts token-identical against the Apple-clang
  reference.
- A probe in `ropeImpl` fires 144 times inside the parity driver itself, so the
  ported code demonstrably runs there rather than merely linking.

### What the parity gate actually catches

Measured by fault injection, each one confirmed to have changed the built
archive before the gate was run:

| Injected fault | `parity-port` |
| --- | --- |
| `ADD` node emitted as `SUB` | **FAIL** 6/6 |
| RoPE `freq_base` forced to 1.0 | **FAIL** 6/6 |
| RoPE `freq_base` x 2.0 | pass |
| RoPE `freq_base` x 1.01 | pass |
| softmax scale x 0.5 | pass |

So token parity is a structural gate, not a numeric one. Halving the softmax
scale and doubling the RoPE base both survive 6 prompts x 48 greedy tokens.
This is why `test-backend-ops` is the next step and not an optional extra:
nothing in the current harness would catch a kernel that is subtly wrong.

The softmax result has a likely explanation worth checking before relying on
it -- llama.cpp uses `FLASH_ATTN_EXT` on Metal by default, so `SOFT_MAX` may
not be on the attention path at all for this model.

### Two traps this swap created

**The comparison harnesses can now compare the port against itself.**
`zig-out/lib/libggml.a` used to be the C; it is the port now. `scripts/graph-diff`
survives only because its reference side is the Apple-clang CMake build, which
is the last genuinely-C ggml in the tree. Repointing either side of it at
`zig-out` would silently turn it into a tautology. The same hazard applies to
`scripts/parity-port`, which is why repointing it at `llama.cpp.zmake` is now
load-bearing rather than housekeeping.

**A failed build leaves the previous library in place, and every gate passes on
it.** This bit twice during the fault-injection work above: a compile error
scrolled past, `parity-port` ran against the stale archive, and reported a
clean pass. Any injection harness must assert the build succeeded *and* that
the artifact's checksum changed.

---

## What `test-backend-ops` does and does not prove

`make backend-ops` builds upstream's `tests/test-backend-ops.cpp` against
`zig-out/lib/libggml.a` and runs it: **21,093 op configurations, 2/2 backends
passed.** `--diff` also builds it against the Apple-clang reference and
compares output: **25,176 lines identical.**

This is the broadest exercise the port has -- every op, across types, shapes,
broadcasts and permutations, actually executed rather than merely inspected.
`FLASH_ATTN_EXT`, `SSM_SCAN`, `RWKV_WKV7` and the rest of the ops written last
all run.

**But it compares Metal against CPU, not against the C.** Both backends read
the same `op_params` from the same ported constructor. A constructor that is
consistently wrong makes both backends wrong identically, and they agree.

Confirmed by injection rather than assumed. Halving the softmax scale in
`ops.zig`, with the rebuilt archive's checksum verified changed:

| Gate | softmax scale x 0.5 |
| --- | --- |
| `make backend-ops` | **misses** -- "2/2 backends passed", exit 0 |
| `make backend-ops --diff` | **misses** -- 25,176 lines still identical |
| `make graph-diff` | **catches** -- names both SOFT_MAX nodes, `3f800000` -> `3f000000` |

The `--diff` mode misses it because the test's case descriptor prints the
scale the *test* chose, not the value the constructor wrote.

So the three gates are genuinely complementary, and the table in "The `ggml.c`
swap" extends to:

| Fault | graph-diff | backend-ops | parity-port |
| --- | --- | --- | --- |
| `ggml_permute` dropped its `op_params` | **caught** (found for real) | untested | untested |
| softmax scale x 0.5 | **caught** | missed | missed |
| `mul_mat` shape wrong when `a->ne[1] > 100` | **caught** (via `conv_2d`) | - | - |
| `ADD` emitted as `SUB` | - | - | **caught** |
| RoPE `freq_base` = 1.0 | - | - | **caught** |
| RoPE `freq_base` x 2.0 | - | - | missed |

**`graph-diff` remains the sharpest gate for the port's actual failure mode**,
which is a constructor writing the wrong thing. `backend-ops` earns its place
by catching what a 131-node dump cannot reach: a constructor that crashes,
trips an assertion, or emits a shape a backend has no kernel for, on one of
thousands of configurations. `parity-port` is the end-to-end structural check.

An attempt to build a fault that `backend-ops` catches and `graph-diff` misses
did not succeed: a `mul_mat` shape error gated on `a->ne[1] > 100`, chosen
because the dump's own `mul_mat` cases use 16, was still caught -- `conv_2d`
builds a large `mul_mat` internally. The 131 nodes reach further than their
count suggests.

---

---

## Porting `ggml-quants.c`

5,667 lines, 79 symbols. **Complete and swapped in.** Nothing under
`llama.cpp/ggml/src/` compiles as C any more: `ggml.c`, `ggml-alloc.c` and
`ggml-quants.c` are all ported, and `ggml_base_sources` is empty.

Split across `src/ggml/quants/`: `blocks` (27 layouts, all asserted against the
C), `helpers` (the shared scale fits), `legacy`, `k`, `ternary`, `iq1`-`iq4`,
`iq_dequant`, `codebook` (the lattice construction), `chunks`, `validate`, and
two generated files -- `grids.zig` (4,608 codebook values) and `golden.zig`.

### Two things that shaped the approach

**`ggml-common.h` imports cleanly, tables included.** Unlike `ggml-impl.h`,
which `arm_neon.h` makes unparseable, this one yields the block layouts *and*
the ~1,900 lines of i-quant codebooks -- `iq2xxs_grid`, `ksigns_iq2xs`,
`kvalues_iq4nl` and the rest -- behind `GGML_COMMON_DECL_C` and
`GGML_COMMON_IMPL_C`. The ported quantizers index the same arrays the C does.
Hand-transcribing those tables would have been the single largest source of
silent error in the file, and it is avoided entirely.

**The block layouts are still restated by hand, in `quants/blocks.zig`.** Not
because the import fails, but because several blocks wrap their scale pair in
an anonymous union so it can also be read as one `ggml_half2`. `translate-c`
names anonymous members positionally -- `block.unnamed_0.unnamed_0.d` -- and
the number is assigned per translation unit, so an unrelated struct added
earlier in the header renumbers it. `layoutMatches` asserts every restated
layout against the imported C struct at compile time, and **it immediately
earned that**: `q4_1`, `q5_1` and `q8_1` align to 4 rather than 2, because the
`ggml_half2` in the union carries its alignment out to the enclosing struct.
The `align(4)` in those three is load-bearing.

### The gate

`make test-quants`, also run by `make validate`. Every format is checked
against **checksums captured from the C**, generated by
`scripts/quants-golden` into `quants/golden.zig`.

A round-trip test -- quantize, dequantize, assert the error is small -- was the
obvious alternative and is worthless here: it passes with the quantizer and the
dequantizer wrong in matching ways, which is exactly how a packing mistake
survives. The 4-bit and 5-bit formats pack element `j` with element `j + qk/2`,
the two *halves* of the block rather than adjacent elements, and a round trip
notices nothing if both ends agree on the wrong pairing.

Negative-tested, four ways, each confirmed to fail:

| Injected fault | Result |
| --- | --- |
| `q4_0` halves swapped | 21/23 |
| `q8_0` round -> truncate | 22/23 |
| `q1_0` mean scale -> sum | 22/23 |
| `mxfp4` exponent bias off by one | 22/23 |

### Four bugs the gates caught, and one they did not

- **A pointer walked the wrong way.** The C recovers a neighbour list with
  `kneighbors - kmap[u] - 1`, where `kmap[u]` is *negative* -- so it is pointer
  **addition**. Transcribed as subtraction it compiles, runs, and reads off the
  front of the array. Found by `make backend-ops`, which aborted; the golden
  tests could not see it because the i-quants had none yet.
- **The 3-bit grids are `uint32_t`, not `uint64_t`.** Read through the 64-bit
  helper, `iq3_xxs` and `iq3_s` silently index every second entry. Caught by
  the goldens once they existed.
- **`iq1_s` used `iq1_m`'s epsilon** -- 1e-7 where the C uses 1e-12, five
  orders of magnitude apart.
- **`makeQpQuants` traps where the C truncates.** Three stores narrow an `int`
  to `uint8_t` without clamping below; `@intCast` panics, the C wraps.

The one they did not: **the i-quants had no golden test at all** until the
pointer bug surfaced downstream. Ten quantizers and nine dequantizers were
written, compiled, and believed. `iq_test.zig` now drives every format from the
golden record, so a format cannot be silently skipped.

### Two places the C's own answer is unspecified

Both are excluded from the goldens rather than matched, and both were measured
rather than assumed:

- **`quantize_row_iq4_nl_ref` reads uninitialized memory** for an all-zero
  block: it takes an early-out, never writes `L`, then packs from it. Two
  consecutive calls in one process return *different bytes*. Our version zeroes
  the buffer, so an all-zero block quantizes to all-zero quants.
- **The 1-bit split search depends on `qsort`'s ordering of equal elements.**
  When a block's weights are identical, two splits score the same in exact
  arithmetic and the winner comes down to the rounding of a sum accumulated in
  sorted order. macOS libc returns `31 1 2 ... 30 0` for 32 equal elements;
  glibc would differ. Our comparator breaks ties on index, so our output
  depends on nothing but the input. Real weights do not produce exact ties.

### A refinement of the duplicate-symbol lesson

The ported quantizers cannot go in `test-port`: it compiles `ggml-quants.c` for
the traits table's function pointers, and the two definitions collide. **Here
that collision is a hard link error**, where the same clash between two
*archive members* was resolved silently (see "The `ggml.c` swap"). The
difference is object files handed straight to the linker versus members of an
archive. Both are worth knowing; only one of them is safe.

Hence `test-quants` and `src/ggml/quants_root.zig`: a separate root keeps
`ggml-quants.c` out of that link, which is what allows the file to be checked
against its goldens *while it is being written* rather than only at the swap.
For a 5,667-line translation unit, that is the difference between incremental
feedback and none.

---

---

## Porting `ggml-cpu.c`

**Done: 65/65 symbols, swapped in.** 3,900 lines of C become
`src/ggml/cpu/{defs,traits,convert,features,tensor,mulmat,forward,plan,threading}.zig`.
All five gates pass with it in place, and `make port` and `make ref` are still
byte-identical.

### Roughly a third of the file never compiled here

The C carries Windows atomics, SRW locks, an OpenMP path, Linux NUMA
enumeration, x86 and RISC-V SIMD conversion loops, and four platform arms each
for thread affinity and priority. On macOS arm64 with `GGML_USE_OPENMP` unset,
most of that is `#if`-ed out. The port carries only the arms that compile, and
says so at each site rather than leaving the reader to wonder what happened to
the other three.

That is why 3,900 lines of C came out as roughly 2,600 lines of Zig including
doc comments and 25 unit tests.

### What was mechanical, and what was not

Mechanical: the traits table, the element accessors, the `ggml_cpu_has_*`
predicates, the work-size arithmetic, and both dispatch switches. The two
switches were **generated from the C** by a throwaway script and then edited,
rather than transcribed — 102 ops in one and 102 in the other, and a
hand-transcribed case label is exactly the sort of error no gate would catch.
The generator also proved the switches exhaustive: 102 arms against 102
members of `enum ggml_op`.

Not mechanical, and worth recording:

- **Zig 0.16 has no `@fence`.** `ggml_barrier` and the polling worker both want
  a standalone `atomic_thread_fence(seq_cst)`. The port uses the C's own
  thread-sanitizer fallback — `fetchAdd(0, .seq_cst)` on the field the C names
  — which is a path upstream already ships.
- **`std.Io.Mutex` is unusable here** (Decision 39), so the pthread calls the
  C's macros expand to are made directly.
- **Zig reserves `i1`, `i2`, `i3`, `i11`, `i12` and `i13` as type names**, and
  `mul_mat` uses every one as a loop index. They are `j11`, `j12`, `j13` and so
  on, digit for digit, so the index arithmetic still reads against the C.
- **`ggml_compute_params` is an ABI contract, not an internal type.** It is
  declared in `ggml-cpu-impl.h`, which is unimportable for the usual
  `<arm_neon.h>` reason, and `ops.cpp` and `sgemm.cpp` read it directly. Its
  field offsets are asserted in a unit test.
- **The `nrows` column of the traits table branches on i8mm**, which selects
  two-row kernels. The port reads the same target feature Zig knows about and
  `zig cc` sets, so the table and the still-C kernels cannot disagree — a
  `nrows = 2` against a kernel built without the instruction reads past its row.
- **`q8_1` and `q8_K` are both right-hand-only types with no `vec_dot`, and the
  C gives them different `nrows`** — 1 and 0. It looks like one case and is
  two; both are asserted rather than assumed.

### Two gates had to grow

- **`make probe` now takes two runs.** The allocator probe fires while the
  graph is being planned, so it would mask a CPU probe every time. `-Dprobe-cpu`
  is a second flag with a second phase in `scripts/probe-ported`. Negative-tested:
  removing the probe call makes that phase fail.
- **`make validate` was swallowing `port-coverage`'s failures.** It ran
  `./scripts/port-coverage || true`, and when the new `probe_cpu` flag broke the
  script's stand-in `config` module, validate reported OK while the coverage
  step printed a compile error. The `|| true` is gone. The format check was also
  globbing only `src/ggml/*.zig`, so `quants/` had never been checked and `cpu/`
  would not have been; it now covers both.

Both of those are the same failure this project keeps meeting: a gate that is
green because it is testing nothing.

### The fault injection

`backend-ops` is the gate that bites for this file, because it is the only one
that runs the CPU kernels at scale against an independent implementation.
Truncating `mul_mat`'s dot product by one element — `ne00 - 1` in the `vec_dot`
call — produced `[MUL_MAT] ERR = 0.002239948 > 0.000500000 ... FAIL` across the
f16 configurations and a non-zero exit. Build success and the library's
checksum were confirmed on both the faulted and restored builds.

Separately, an abort at the top of the ported dispatch fired during ordinary
`zig build smoke` inference, which answers the question the probe exists for:
with Metal carrying the model most ops never reach the CPU, but some do.

### What is still C underneath

The dispatch calls ~90 kernels in `ggml-cpu/ops.cpp`, the vector helpers in
`vec.cpp`, the extra-buffer hooks in `traits.cpp`, and `llamafile_sgemm`. Those
are Stage 4. The traits table points at `quants.c` and `arch/arm/quants.c`,
which are Stage 3 step 5 — the last C in `ggml/src/`.

Linking those into `ported-lib` and `test-port` meant adding every `ggml-cpu/`
source except `ggml-cpu.c` itself, plus `-framework Accelerate`:
`GGML_USE_ACCELERATE` sends the vector helpers to vDSP, and without the
framework the ported test binary does not link.

---

---

## Porting `ggml-cpu/quants.c`

**Done: 45/45 symbols, swapped in.** 1,339 lines become
`src/ggml/cpu/quants/{rows,legacy,k,ternary,iq}.zig`, plus a generated
`golden.zig` and a `testing.zig` fixture. All gates pass; `make port` and
`make ref` stay byte-identical.

### The two quant files are independently swappable

`quants.c` exports 45 symbols and `arch/arm/quants.c` exports 28, with **no
overlap**, and `arch/arm/quants.c` needs nothing from `quants.c` — only
`quantize_row_q8_K_ref` and `ggml_table_f32_ue4m3`, both already ported. So
what the plan had as one 5,658-line all-or-nothing step is two, and the gates
run twice instead of once.

### 25 of the 45 symbols are unreachable

`arch-fallback.h` renames nothing from `quants.c` on ARM, and
`arch/arm/quants.c` supplies all 28 real entry points. So every
`ggml_vec_dot_*_generic` is exported and never called. `use_ref` does not route
to them either — it only disables fusion, the Hadamard fast path, and
flash-attention tiling.

**`test-backend-ops` cannot see them. Neither can token parity.** That was
worth establishing before writing a line: it decides that the gate has to be
goldens captured from the C, and it means the port would otherwise have had
none.

The 20 row quantizers *are* live — 17 are what `type_traits_cpu` calls, and
three are the `_generic` fallbacks the NEON file replaces.

### The goldens, and the first version that was worthless

`harness/vecdot_golden.c` runs each of the 25 kernels on five input patterns
and emits the **raw `f32` bits**, not a decimal or a tolerance: a dot product
right to six decimals and wrong in the last bit is a porting bug, and catching
those is the point.

The first pattern set was nearly useless and the numbers said so: `alternating`
cancelled to exactly zero for 22 of the 25 kernels, and `tiny` underflowed the
fp16 delta to zero for all 25. **72 of 125 golden values were
`0x00000000`** — a kernel that did nothing but `*s = 0` would have passed most
of them. Replaced with `signs` (asymmetric magnitudes, coprime periods) and
`lopsided` (operands four orders of magnitude apart, so a swapped scale shows
up): now 28 of 125 are zero, and 25 of those are the `zeros` pattern, where
zero is the right answer.

`testing.zig` asserts that bound so it cannot regress, and asserts the LCG's
first output — if the generator and the fixture ever disagree about the inputs,
every comparison silently tests something else and passes.

### Reinstating the second test root

`quants.c` could not be swapped until all 45 symbols existed, but while the C
was still in the build its `quantize_row_q4_0` collided with the ported one —
and two *object files* is a hard link error, unlike two archive members. So
`src/ggml/ported_cpu_quants.zig` and `zig build test-cpu-quants` came back,
exactly the `quants_root.zig` device the `ggml-quants.c` port used, with
`quants.c` omitted from that step's C sources. Both are gone again now that the
swap has landed.

### Four bugs, and which gate found each

| Bug | Found by |
|---|---|
| `q2_K`'s `shift` typed `u3`; the C's `shift += 2` runs once more than it is used, ending at 8 | Debug-build overflow panic |
| `q3_K` and `q5_K`'s rotating mask `m <<= 1` overflowing `u8` on its last, unread shift — the C truncates | Same |
| `Scratch.accumulate`'s parameter shadowing a method named `scale` | Compiler |
| My own `tq1_0` unit test asserting the wrong packing | The test itself, against passing goldens |

The last one is worth keeping. The golden comparison passed while my
hand-written test of `digit()` failed, which meant the *port* was right and my
*understanding* was wrong: `tq1_0` does not pack five base-3 digits into a byte
directly. It builds `d0*81 + ... + d4` in 0..242 and then rescales into the
byte range with a ceiling division, so the byte is `0.d0 d1 d2 d3 d4` read as a
fraction of 256 and the extraction is a fixed-point digit shift. The test now
round-trips all 243 combinations, and the file's header describes the real
encoding rather than my first guess at it.

### Fault injection

Two, both confirmed to fail and then reverted:

- Dropping `q4_0`'s bias of 8 failed `q4_0 dot matches the C` and nothing else.
- Dropping `iq2_xxs`'s trailing `0.125` reported `want 0x412108DC (10.064663),
  got 0x42A108DC (80.5173)` — exactly eightfold, naming the pattern and both
  values.

### Two more gate leaks closed

- **`make validate` was swallowing `port-coverage`'s failures** — `|| true`.
  Removed during the `ggml-cpu.c` step; it caught a real breakage here.
- **The format check had never covered `src/ggml/quants/`**, and now covers
  `src/ggml/cpu/quants/` too. Globbing one directory at a time is a standing
  hazard in this repo.

---

---

## Porting `ggml-cpu/arch/arm/quants.c` — Stage 3 complete

**Done: 28/28 symbols, swapped in.** `src/ggml/cpu/quants/arm/` holds
`neon.zig`, `rows.zig`, `legacy.zig`, `ternary.zig`, `k.zig`, `iq.zig`, a
generated `golden.zig`, and nothing else. **No C compiles anywhere under
`ggml/src/` any more.**

### It was a third the size it looked

**1,556 of 4,319 lines survive the preprocessor on this target.** The rest sits
behind `__ARM_FEATURE_SVE` and `__ARM_FEATURE_MATMUL_INT8`; `q4_0` alone has
240 dead lines against 50 live ones. Measured with `zig cc -E` and the
linemarkers, not estimated — the raw line count would have put this at three
times `quants.c` when it is comparable.

Of 1,015 live intrinsic calls across 82 distinct intrinsics, exactly one has no
portable Zig equivalent.

### `neon.zig`, and the three traps in it

Zig has no ACLE intrinsics but it has `@Vector`, so 81 of the 82 are a line
each. The value of gathering them is that the traps end up in one place, and
all three were **measured**, not reasoned about:

| Trap | What goes wrong |
|---|---|
| NEON integer arithmetic **wraps** | Zig's `+`, `*` panic in a safe build. Every integer op there uses `+%`, `*%`. |
| `vaddvq_f32` is **pairwise** | It lowers to two `faddp`, giving `(a0+a1)+(a2+a3)`. `@reduce(.Add, ...)` on floats is *ordered*. For `{1e8, 1, -1e8, 1}` they give 0 and 1 — a silent 1-ULP bug in 13 places. |
| `vshlq_u8` shifts **right** on a negative amount | `q2_0` passes `{0,-2,-4,-6}` to extract four 2-bit fields in one instruction. The first version of `shlq` asserted the amount non-negative. |

`vdotq_s32` is the one with no equivalent, and needs none: it is integer, so the
widening-multiply-and-shuffle form is **bit-identical**, not approximate. The
only question was throughput, and the answer is no measurable change:
generation is 222-234 t/s either side of the port across three runs each, and
the prompt figure swings by over 100 t/s run to run, so a point comparison
would be reading noise. Metal carries the model, and the quantized CPU path is
barely on the critical path.

### Naming fusion sites, soundly this time

This file already records why `@mulAdd` was abandoned in the quantizers: the
sites are implicit across 5,000 lines and the guess broke. **Here it was the
right call, and the difference is an oracle.**

`iq4_nl` came out one ULP low. The obvious reading — `sumf += SA*PA + SB*PB` —
was wrong, and so were two guesses at the association. So
`harness/vecdot_prefix.c` was written: it asks the shipped kernel for every row
prefix from one block to sixteen. `nblk = 1` exercises only the scalar tail and
`nblk = 2` only one unrolled iteration, which showed the divergence was in the
*term*, not the summation. Eight candidate orderings later, exactly one
reproduced all sixteen values:

```
vector loop:  sumf += fma(SA, PA, SB*PB)     // contract fuses the left multiply
scalar tail:  sumf  = fma(d, sumi, sumf)     // and the tail's, into its accumulate
```

Every K-quant then needed the same treatment on its float epilogue, and all
five were one ULP out until each `sum += d * isum` became `@mulAdd`.

**The rule this settles:** name a fusion only when something can tell you you
got it wrong. `scripts/vecdot-prefix` is that something, and it stays in the
tree.

### Three gate holes, all found by injection

- **Every `nvfp4` ARM golden was `0x00000000`.** `GGML_CPU_UE4M3_TO_FP32` is a
  *table* lookup on NEON — unlike the generic kernel's arithmetic form — and
  the harness never called `ggml_cpu_init`, so every scale was zero. A kernel
  returning nothing would have passed all six patterns. The harness now calls
  it, and `testing.zig` refuses a golden row that is entirely zero.
- **No pattern hit an exact tie.** Swapping round-half-to-even for
  round-half-away-from-zero passed every golden, because pseudorandom input
  never lands on `n + 0.5`. The `ties` pattern pins element 0 at 127 so
  `amax == 127`, `d == 1`, `id == 1` and every other element is an exact tie.
  The same injection now fails on it.
- **A filtered test run hid a compile error, and a filter that matched nothing
  looked green.** `-Dtest-filter="_K dot matches the C\|widening product"` is a
  literal string, not an alternation; it selected zero tests and reported a
  comfortable row of passes, and q2_K and q3_K were claimed byte-exact on that
  basis when they had never run. Separately, Zig skips codegen for unselected
  tests, so a `usize` shift bug in `iq.zig` only surfaced on an unfiltered run.

### The scaffolding, twice put up and twice taken down

`src/ggml/ported_arm_quants.zig` and `zig build test-arm-quants` came back for
the third time in this project, and went away again on the swap. One new wrinkle:
unlike `quants.c`'s unreferenced `_generic` names, these 28 symbols are named by
`src/ggml/cpu/traits.zig`, so omitting the C left them undefined. The root
carried a `pending` list of aborting stubs — safe because they panic — and the
list emptying to zero was the progress measure.

---

---

## The citation gate — `make port-links`

Every ported declaration already carried the C it replaced, the file and the
line: ``Ports `ggml_hash_set_new` (ggml.c:6516)``. The convention was in
`CLAUDE.md` from the start and had been followed in all 42 files. Nothing
checked it.

**216 of the first 790 citations were stale.** Drifted by a handful of lines
each, mostly from reading a definition's body line rather than its first line;
two pointed at the wrong file entirely — `NGRID_IQ1S` cited `ggml-quants.c`
when it lives in `ggml-common.h`, and a citation of `ggml_abort_callback_t`
named a type where the definition is the variable `g_abort_callback`. That is
what a convention does when nothing enforces it, and it is the failure mode
this whole gate exists to prevent: **a stale line number is worse than no line
number, because it sends you to the wrong function and says nothing.**

### What it checks

`harness/port_links.zig`, driven by `scripts/port-links`. Zig rather than bash
or Python, so the dependency set stays exactly `{zig}`.

- **The cited line defines the cited symbol.** Either the name is on that line,
  or the line opens a brace block closing on a line that names it — which is
  how `ggml-common.h`'s anonymous `typedef struct { … } block_q4_K;` gets
  cited at its *first* line, where a diff actually lands.
- **A citation may name several symbols**, `(ggml-impl.h:173, 179, 185, 191)`;
  names and numbers pair up in order. Names are read from the citation's own
  line, widening to the doc comment block it closes when the citation wrapped.
- **The file resolves unambiguously.** Two `common.h` exist and eight
  `quants.c`; a suffix matching more than one is an error, not a guess. This
  cost 47 citations a path prefix and caught a `GGML_FA_TILE_Q` reference that
  a "shortest path wins" tiebreak had silently sent into `common/common.h`.
- **The commit.** Every citation of upstream must name one, and the cited line
  is read from that commit's blob rather than from the working tree.

### The commit goes on every citation

First attempt put it once per file, in the module doc comment, reasoning that
all 850 were captured at one commit and 850 copies of one string is 850 places
for it to rot. **That reasoning does not survive contact with an actual
upstream sync.** A sync does not move a whole file at once — one function gets
re-read against a newer commit while its forty neighbours stay where they
were, which is precisely what a diff produces. A single per-file commit cannot
express that, and becomes a lie about the other forty the moment anyone does
it.

So the commit is on the citation: `(ggml.c:6516 @c1d0e7a00)`. The checker reads
each cited file **as it was at that citation's own commit**, via `git show
<sha>:<path>`, falling back to the working tree only when the citation is at
the checkout's `HEAD`. A file may cite as many commits as it needs to, and the
summary line reports how many are in play.

The "rot" worry was real but misplaced: what protects the strings is that every
one of them is *checked against the commit it names*, not that there is only
one of them.

Citations pointing at our own `harness/` — `testing.zig` names the C that
captured its goldens — carry no upstream commit and are not asked for one.

### The checker's own false passes

Three, all found by making it stricter rather than by it reporting anything:

- **It only looked for the word `Ports`.** Every `Mirrors …` citation in
  `blocks.zig` — most of the block-layout types — had never been checked, and
  most had drifted.
- **A malformed line number was skipped, not reported.** A citation this work
  had itself mangled to `(common/common.cpp:)` passed unnoticed, because the
  parser did `parseInt(…) catch continue`.
- **Ambiguous file resolution guessed.** As above.

Each is the same shape as the `|| true` in `validate` and the format check that
globbed one directory: **green because it was testing nothing.**

### Fault injection

Seven, each confirmed to behave as stated and revert. The last two are the ones
that prove the per-citation commit is doing real work: `MAX_FREE_BLOCKS` is at
`ggml-alloc.c:14` at `HEAD` and at `:13` at `e57f52334`, so the two cases have
*opposite* expected outcomes and only a checker actually reading the older blob
gets both right.

| Injection | Expected | Result |
|---|---|---|
| A line number drifted by two | fail | `-> ggml-alloc.c:16 @c1d0e7a00: that line is //#define GGML_ALLOCATOR_DEBUG` |
| A line number deleted | fail | `citation of ggml-alloc.c has no usable line number` |
| A symbol renamed | fail | `ggml_nope_op_can_inplace -> ggml-alloc.c:22 @c1d0e7a00: that line is bool ggml_op_can_inplace(...)` |
| The commit omitted | fail | `names no commit -- write (ggml-alloc.c:14 @c1d0e7a00)` |
| A commit that does not exist | fail | `cannot read ggml/src/ggml-alloc.c at commit abcdef123` |
| `:13 @e57f52334` — right there, wrong at `HEAD` | **pass** | passed |
| `:14 @e57f52334` — right at `HEAD`, wrong there | fail | `-> ggml-alloc.c:14 @e57f52334: that line is ``` |

The first attempt at this reported three of the four as passing. The `sed`
patterns had not matched, so nothing was injected, and `$?` was being read
after a command substitution had already clobbered it. **A fault injection that
does not verify the fault was injected proves as little as the gate it is
testing** — the check is now `cmp` against a pristine copy before running.

---

---

## Chat templates

Decision 25 took `gremlin-labs/vibe-jinja` rather than porting `common/jinja`'s
6,349 lines. Decision 27 excepted it from the no-dependency rule, confined to
`cli/`. Both hold. What the work actually turned on was none of that.

### The dependency needed a Zig 0.16 port first

vibe-jinja publishes against 0.15.2. The gap is small and entirely mechanical
-- 174 lines across 21 files: `ArrayList(T){}` to `.empty`, `writer(a).print(…)`
to `print(a, …)`, `trimLeft`/`trimRight` renames, and the clock calls that 0.16
moved behind `Io`. Its `build.zig` is 0.15-era too, so `b.dependency` cannot run
it as published regardless of the source.

**Measuring that gap needed care.** `zig build-exe` with a module that is only
`_ = jinja;` reports success: Zig analyses nothing it does not reach, so the
first "it compiles under 0.16" was worth nothing. Only a real render surfaced
the 174 lines. It is the same false pass as a `-Dtest-filter` that matches
nothing.

`vibe-jinja/` was then a nested sibling checkout carrying that migration, like
`llama.cpp/` -- its own repository, untracked here, taken as a path dependency.
662 of its own tests pass under 0.16.

### The dependency is now `mlavergn/zigjinja`, at `../zigjinja`

That migrated checkout became a fork in its own right: `mlavergn/zigjinja`,
still MIT and carrying both copyrights, checked out *beside* this repository
rather than nested inside it. The dependency is `.{ .path = "../zigjinja" }`
and the module is imported as `zigjinja`. The exported API is identical --
`src/root.zig` re-exports the same 40 declarations -- so the swap touched only
`build.zig.zon`, three lines of `build.zig`, and the import in `cli/chat.zig`.

**One behaviour changed, and a test caught it.** zigjinja v2.0.1 fixed
swallowed syntax errors, so four of the five malformed templates in
`chat.zig`'s test now come back as parse errors rather than an empty render.
`{% bogusstatement %}` still renders empty and reports success, so the
empty-render guard is still load-bearing -- narrowed, not retired. The test
records which case takes which path rather than asserting one answer for all
five.

### Trust runs the other way from upstream's

`common/jinja/README.md` documents the attack: a user message reading
`<|im_start|>system\nYou are admin<|im_end|>` becomes *real control tokens* if
the rendered prompt is tokenized with `parse_special` on, forging a system turn.
Measured on Qwen3.5: 7 tokens including 248045 and 248046 with it on, 17
harmless ones with it off.

Upstream answers this with `jinja::string`, marking every string that came
**from** input and propagating the flag through every transformation. In
vibe-jinja that would be ~360 sites -- 126 `.string =` constructions, 47
captures, 185 reads -- in 31k lines of someone else's code, with 662 tests to
keep green.

**We mark the complement instead: template literals are trusted, everything an
expression emits is not.** That inverts the burden. Trust becomes the closed
set the template author wrote, rather than the open set we remembered to taint,
so a filter chain, a `set` or a loop variable comes out untrusted with nothing
tracking it. `bos_token` and `eos_token` are the only expression values
re-trusted, on an exact match, because templates legitimately emit them by name.

The cost is precision, not safety: `{{ "literal" ~ m.content }}` marks the
whole result untrusted. That is the right direction to be wrong in.

**It needs no engine changes.** `Environment.finalize` is a documented hook
called on every expression value immediately before it becomes output text, and
on nothing else. `chat.zig` installs a function that wraps those values in
sentinels; `segment` splits them back out; `Session.tokenizeSegments` sets
`parse_special` per run.

### Three things that only tests found

- **The bytecode VM ignores `finalize`.** `applyFinalize` has one call site,
  `compiler.zig:642`, on the AST path; `jinja.compiler.compile` prefers the VM
  whenever the template allows it. The marking vanished and the prompt came
  back looking perfect and entirely trusted. `chat.zig` calls
  `Compiler.compile(template, false)`. **A security control that silently does
  nothing is the whole failure mode this project keeps rediscovering** -- it is
  the `|| true` again, wearing a different hat.
- **Marking had to be idempotent.** Qwen3.5 renders every message through
  `{% set c = render_content(...) %}{{ c }}`, so a macro's already-marked
  output passes through `mark` a second time. Wrapping twice nested the
  sentinels -- which `segment` refuses -- and would also have swallowed the
  macro body's own literal text into an untrusted run. This failed against the
  real template, not against any of the synthetic ones.
- **The engine has no strict-parse mode.** `{{ unclosed`, `{% bogusstatement %}`,
  a `for` with no `endfor` and `{{ 1 + }}` each render as **empty output and
  report success**. An empty prompt reaches the model as empty context and
  reads as a model fault. `chat.zig` rejects a render that produced nothing
  from a non-empty conversation.

### Fault injection

Five, each confirmed to fail and revert, run against the new `test-cli` step
because the library's tests cost a minute and these needed iterating:

| Injection | Result |
|---|---|
| `env.finalize = mark` removed | 5 fail |
| bytecode path restored (drops finalize) | 5 fail |
| content sanitising removed | 1 fail |
| every expression value re-trusted | 5 fail |
| fail-closed guard replaced with `unreachable` | 1 crash |

### Opt-in, against upstream's default

Upstream sets `use_jinja = true` and templates by default. Ours does not,
because raw completion is what `make port` and `make ref` diff against each
other, and Decision 23 makes that the gate for the whole binary. Templating by
default would have changed it silently. Any chat flag turns it on; `--no-jinja`
vetoes. Recorded in SPEC section 6.1.

### Interactive mode, and the bug that only a conversation found

Decision 30 is closed: `-cnv` reads turns from stdin, appends each reply to the
history, and re-renders the whole conversation every turn. Re-rendering rather
than appending is deliberate -- a template decides for itself where the system
prompt goes and how a turn is framed, and Qwen3.5's rewrites assistant turns to
split reasoning content out of them. `Session.generateTurn` diffs the new
prompt against the decoded tokens and keeps the common prefix, so the cost is
tokenization rather than decoding.

**`llama_memory_seq_rm` returns a bool, and on Qwen3.5 it returns false.** The
first version discarded it. Recurrent and hybrid models cannot drop a range
from the middle of the cache, so the removal silently did nothing, the next
turn decoded on top of a stale cache at wrong positions, and the model answered
turn two with `Madrid<|im_end|>\n</think>\n\nMadrid` -- literal control-token
text, which reads as a template fault and is not one.

Nothing in the test suite could have caught it: it needs a model, three turns,
and a hybrid architecture. It was found by holding a conversation and reading
the output, then bisected by disabling the prefix reuse. `generateTurn` now
checks the result and clears the cache when the refusal comes back.

**The lesson is the project's own, again:** a C API that reports failure in its
return value reports it to nobody if the caller writes `_ =`.

### Still open

- **A `chat-parity` gate.** `common/jinja` is seven self-contained `.cpp` files,
  so upstream's engine can render the same template and the two outputs be
  diffed. Nothing yet checks that two independent Jinja implementations agree
  on the templates in the wild, which is an assumption rather than a fact.
- **No gate covers the conversation loop.** `parity-cli` runs one-shot prompts;
  multi-turn behaviour, KV prefix reuse and the hybrid-cache fallback are
  checked by hand only.
- **`-mli` and `-if` stay refused.** Both change what a turn is.
- **Slash commands stop at `/exit`, `/clear` and `/regen`**, matching upstream's
  set minus the multimodal ones.

---

---

---

## The first two C++ translation units

Stage 4 opened with one question the plan could not answer by reading: how much
C++ linkage actually crosses a translation-unit boundary. Zig emits no mangled
names, no vtables and no RTTI, so anything that does cannot be replaced.

### The measurement, and the bug in the first version of it

Every exported symbol of every ggml C++ object, intersected with the undefined
symbols of every other object in `libggml.a` and `libllama.a`. The answer is
that **20 of the 24 C++ translation units in ggml present a pure C ABI
outward**, and the four that do not each name their own constraint:
`ggml-backend-dl.cpp` (functions taking `std::filesystem::path &`),
`ggml-cpu/traits.cpp` (vtables and RTTI), `ggml-cpu/ggml-cpu.cpp` (one function
returning `std::vector`), and `ggml-metal/ggml-metal-tuning.cpp` (a C++
namespace).

**The first run of that analysis said "zero mangled symbols anywhere", and it
was wrong.** Mach-O prefixes every C symbol with `_`, so an Itanium-mangled name
appears as `__Z…`; the script stripped the prefix with `lstrip("_")`, which
removes *both* underscores, and every mangled name came out looking unmangled.
The result was a clean sweep that would have sent the port straight into
`traits.cpp` expecting a C ABI. Stripping exactly one character produced the
real answer.

The lesson is the project's own, in a new place: a measurement is a gate, and a
gate that returns the comfortable answer everywhere is the shape a broken one
takes.

### `ggml-threading.cpp` — 12 lines, and the point was the mechanism

Two functions and a `std::mutex` around ggml's process-wide critical section.
On this platform libc++'s `std::mutex` *is* a `pthread_mutex_t` initialised to
`PTHREAD_MUTEX_INITIALIZER`, and Zig's `std.c.pthread_mutex_t` carries those
same field values as its defaults — so `.{}` is a static initialisation with no
runtime setup, exactly as libc++'s `constexpr mutex()` is.

One deliberate difference: the C++ global has a destructor that runs at exit
through `__cxa_atexit`. The port does not destroy the mutex. Destroying a
process-wide lock during exit while another thread may still be inside it is a
hazard RAII forces and nothing gains from.

**Proved to be executing, not merely linked.** A `@panic` in
`ggml_critical_section_start` fires on a real generation. The first attempt at
that injection put the panic before the lock call with no `if`, which is
unreachable code — the build failed, the old binary stayed in place, and the
"no marker found" result meant nothing. That is the same false pass this file
records four other instances of, and it took ten seconds to walk into.

### `ggml-backend-reg.cpp` — and `ggml-backend-dl.cpp` with it

593 lines, 16 exported symbols. The measurement had flagged
`ggml-backend-dl.cpp` as unportable, and it is — but it is also *only* called
by the registry, so porting the registry removed the last reference and the
file left the build. Its three wrappers became the direct `dlopen`, `dlsym` and
`dlerror` calls they always were.

**Zig 0.16 put the filesystem where the mutex already was.** `getcwd` and
directory iteration now live behind `std.Io`, which takes an `Io` on every
call, and a function entered through a C ABI has none. So the search-path scan
uses libc's `opendir`/`readdir`/`getcwd` — which is what `std::filesystem`
calls underneath anyway. Decision 39 was about locks; it generalises.

**`GGML_USE_METAL` had to become a build option.** The C++ constructor registers
Metal behind `#ifdef GGML_USE_METAL`, and three of this repo's build products
compile without it: the ported test root, `libggml-ported.a`, and
`scripts/port-coverage`'s stand-in module. A ported `#ifdef` is a `comptime`
branch on a `config` flag, and it has to be `comptime` — a runtime `if` still
emits the reference and the link fails. Found by `make validate`, twice: once
for the test root, and again for `libggml-ported.a` after the first fix was
applied to the wrong one of two identical-looking call sites.

### Three things only the gates found

- **The Metal backend is not called "Metal".** A unit test asserting
  `ggml_backend_reg_by_name("Metal") != null` failed: `GGML_METAL_NAME` is
  `"MTL"`. The test now asserts the names the backends actually register, which
  is the only version of it worth having.
- **`port-links` reads code as well as comments.** `loadBackend(best_buf[0..best_len :0].ptr, …)`
  parses as a citation — parentheses, a dot, a colon and digits — and was
  reported as one with an unusable line number. Hoisting the sentinel slice
  into a local fixed the call site. The checker's willingness to scan
  non-comment lines is a latent false-positive source; it has not been changed,
  because narrowing it is a change to a gate and deserves its own negative test.
- **`ggml-backend-reg.cpp` is an ambiguous citation.** A second file of that
  name lives under `ggml-virtgpu/`, so `port-links` refused every citation until
  they were written `src/ggml-backend-reg.cpp`. That is the rule working: the
  first version of this port cited a filename that resolves to two files, and
  nothing but the checker would have noticed.

### The C++ contract, and what `port-coverage` now does

`scripts/port-coverage` takes `LANG_=cxx`, compiles with `zig c++ -std=c++17`,
and filters the contract down to the unmangled exports plus a drop of
`__clang_call_terminate` — a compiler-emitted landing-pad helper that belongs
to the runtime rather than the translation unit.

**Negative-tested.** Removing `export` from `ggml_critical_section_mutex` makes
it report 2/3 and name the missing symbol. Worth knowing what it is not: it
always exits 0, so it is a progress report rather than a gate. The gate is the
linker, which cannot resolve a missing symbol once the C++ file is out of the
build.

---

## Porting `gguf.cpp` — the first STL-heavy translation unit

1,706 lines of C++ into `src/ggml/gguf.zig`. The first port where the C++ is
actually *C++*: `std::vector` and `std::string` throughout, two `std::map`
lookup tables, a template-dispatched reader, and a virtual writer hierarchy.
`ggml-threading.cpp` (a mutex) and `ggml-backend-reg.cpp` (eight entries) were
warm-ups by comparison.

### The contract is 61 symbols, not the 44 `PLAN.md` recorded

Measured with `zig c++ -c` at the pin: 2,002 exported symbols, of which **61**
are unmangled. `PLAN.md` said 44 and has been corrected.

Two of the mangled names are real functions rather than template noise, and
both are correctly excluded:

- `gguf_type_size` and `gguf_write_to_buf`, declared at `ggml-impl.h:781-782`
  under the comment "expose GGUF internals for test code" and **inside `#ifdef
  __cplusplus`**, which gives them C++ linkage.
- Their only caller anywhere in the tree is `tests/test-gguf.cpp`, which this
  project does not build. Nothing in `libggml.a` or `libllama.a` references
  either.

So the plan's rule held exactly as written: the contract is the unmangled
exports, plus any mangled symbol another TU actually references — and here that
second set is empty. They survive as the file-private `typeSize` and
`writeToBuf`.

### Exceptions do not cross the boundary — checked, not assumed

The symbol measurement in `PLAN.md` counted *linkage*. It says nothing about
exceptions, which Zig cannot participate in at all: if `gguf.cpp` threw and
`llama-model-loader.cpp` caught, the port would change behaviour with no linker
error to warn anyone. So it was checked directly.

Every `throw` in the file is caught inside the file. The file writer raises
`std::runtime_error` on a short `fputc`/`fwrite` (gguf.cpp:1585, 1594) and
`gguf_write_to_file_ptr` (gguf.cpp:1665) catches it, returning `false`. Three
more sites catch `std::length_error` and `std::bad_alloc` around vector growth
while parsing a malformed file. The only caller-side `catch` anywhere near this
code, `llama-model-loader.cpp:1120`, catches `std::out_of_range` from its own
map lookup, not from gguf.

**One case is deliberately not reproduced.** `gguf_write_to_buf` has no
try/catch, so a `bad_alloc` from vector growth escapes it and reaches
`std::terminate`. This port aborts through `ggml_abort` instead, as
`impl.ggmlMalloc` already does. Nothing in this build calls that function.

### The virtual writer became a comptime generic

`gguf_writer_base` declares three pure-virtual methods and comments them "we
bet on devirtualization" (gguf.cpp:1438). `gguf_write_out` is *already* a
template over the writer type (gguf.cpp:1622), so the vtable exists only to
give the two concrete writers a shared base. `Writer(Impl)` settles the bet at
compile time and the vtable disappears — this is one of the places where the
Zig is a better expression of the intent than the C++, not merely an equal one.

### Two deliberate differences

- **`gguf_tensor_info.t` is zero-initialised.** The C++ declares it without an
  initialiser (gguf.cpp:632) and fills in only `name`, `ne`, `nb`, `type` and
  `offset`, leaving `buffer`, `data`, `op`, `src`, `view_src`, `extra`, `flags`
  and `op_params` indeterminate before `push_back` copies the struct whole.
  Nothing on the read path reads them — but both writers branch on
  `info.t.buffer` (gguf.cpp:1565, 1609), so a context read from a file and then
  written out would branch on a garbage pointer. Same call as
  `quantize_row_iq4_nl_ref`: where the C's answer is unspecified, reproducing it
  is not the goal.
- **Byte buffers are `u8`, not `int8_t`.** Every use of `gguf_kv::data` is a
  `memcpy`, a `push_back` or a reinterpreting read; no arithmetic depends on the
  sign, and the only ABI-visible use returns `const void *`.

### What the tests caught — and the two that got away first

Twelve tests, of which the round-trips are the substance: they drive the buffer
writer and the reader against each other through the public C ABI, so a
constructor, an accessor, the serialiser and the parser all have to agree.
`gguf_get_meta_size`/`gguf_get_meta_data` produce a complete GGUF whenever there
is no tensor data to follow, so none of this needs a model file.

Eight faults were injected. **Three escaped the first version of the tests**,
and each exposed a real gap:

| Injection | First result | Why it escaped |
|---|---|---|
| Magic check disabled | **passed** | The bad-magic fixture was `"XGUF"` + zeroes, which the *version* check rejects anyway. Now it corrupts byte 0 of an otherwise valid file. |
| Tensor offset not padded | **passed** | The fixture tensor was 8×4 f32 = 128 bytes and `GGML_PAD(128, 32) == 128`. There was no padding to get wrong. Now it is 5 f32 = 20 bytes, padded to 32. |
| `bool` written as `2` | **passed** | Round-trip proves the writer and reader agree, not that either matches the format: `read(bool &)` compares against zero, so `2` survives. Closed with a byte-level golden. |

That last one is the general lesson, and it is the same one the `_generic` dot
products taught: **a round-trip test cannot catch a fault that is symmetric
across the round trip.** The fix is a golden taken from the *specification* —
the GGUF layout documented at the top of `gguf.h` — rather than from our own
output. It pins all 64 bytes of a minimal one-key file, so field order, integer
widths, the `i32` type tag, the length-prefixed key with no terminator and the
alignment padding are all nailed down at once.

After closing those, all eight are caught: magic, offset padding, string length
as `u32`, `bool` as `2`, trailing `ne` filled with 0, enum tag not widened to
`i32`, `n_kv`/`n_tensors` swapped, and `swapRemove` in place of
`orderedRemove`. The last needed four keys with the removal in the middle —
with two keys the two removals are indistinguishable.

### The swap is real at the archive level

`ar t zig-out/lib/libggml.a` no longer lists `gguf.o`, all 61 `gguf_*` symbols
resolve to `libggml_zcu.o`, and no mangled `gguf` symbol remains. Worth doing
explicitly: `port-coverage` compares symbol *lists* and would look identical if
the archive had never been rebuilt.

### Blocked: every link-time gate

**This machine cannot currently link C++ at all.** `zig c++` on a three-line
hello-world fails with `sub-compilation of libcxx failed` / `use of undeclared
identifier 'INFINITY'` in Zig's own `libcxx/src/random.cpp`. It reproduces on a
Zig file containing nothing but `test "x" {}`, and on every SDK on the box
(26.0, 26.5, 27.0), so it is neither this port nor the project's `build.zig`.

Consequence: `zig build lib` and `zig build reference` work (static archives,
no link step), and everything that produces an executable does not — `zig build`
itself, `test`, `test-port`, and therefore `make validate`, `make port`,
`make ref`, `make graph-diff`, `make backend-ops`, `make probe` and
`make parity-cli`.

The unit tests above were run by hand-linking a C++-free test root: the ported
ggml core is pure Zig, so `gguf.zig` can be exercised without `-lc++` once the
`ggml-backend.cpp` entry points the core references are stubbed. That is a real
run of real tests, but it is **not** `make port` loading a 1.2 GB GGUF through
the ported reader, which is the gate this file was chosen for. That gate is
still owed.

---

## Apple's libc++, because Zig's will not build against the macOS 27 SDK

A toolchain workaround, not a port decision, and it should be reverted when the
toolchain is fixed. `make -C testcase cxx20` passing is the signal.

**Reverted under Zig 0.17**, which builds its bundled libc++ against SDK 27.
`testcase/` is gone; see "Zig 0.17: the reference, libc++, and the CPU pin"
at the end of this file.

### What broke

**Zig 0.16.0 cannot link any C++ on this machine.** A three-line hello-world
fails; so does a Zig file containing nothing but `test "x" {}`. Everything that
emits an executable was blocked — `zig build`, `test`, `test-port`, and with
them `make validate`, `make port`, `make graph-diff`, `make backend-ops`,
`make probe` and `make parity-cli`. Static archives were unaffected, which is
why `zig build lib` and `zig build reference` kept working and masked how bad
it was.

### The cause, in three facts

1. **macOS SDK 27 made `INFINITY` conditional.** C23 moved it to `<float.h>`,
   so the new `math.h` defines it only when modules are *off*. SDK 26.x defined
   it unconditionally.
2. **`-std=c++20` and later turn `__has_feature(modules)` on**, which falsifies
   the one term of that guard that was holding it up.
3. **Zig builds its bundled libc++ with `-std=c++23`**, and libc++'s own
   `__random/clamp_to_integral.h:47` uses `INFINITY` without including
   `<float.h>`.

An upstream libc++ bug that the new SDK exposes. Not ours, and not
`build.zig`'s.

### Why not simply force `-std=c++17`

Because it works and is unreachable. Replaying Zig's *actual* compile command
for `libcxx/src/random.cpp` with only `-std` changed gives 1 error at c++23 and
**0 at c++17** — `make -C testcase c17` does exactly that. But the `-std=c++23`
is hardcoded inside the compiler binary. Measured, all three negative:

- `zig c++ -std=c++17 hello.cpp` still fails; your `-std` does not reach it.
- `-fno-modules` and `-fno-cxx-modules` leave `__has_feature(modules)` on.
- `-nostdlib++` does not stop Zig building its libc++ anyway.

An older SDK does not help either: Zig picks the SDK itself through
`std.zig.system.darwin.getSdk` and ignores `-isysroot`, and neither `SDKROOT`
nor `DEVELOPER_DIR` redirects it.

### What was done instead

**Link Apple's libc++ — its headers and its `.tbd` together.** That is a
matched pair, and it is the same C++ runtime the CMake reference build uses, so
it is the closer match to `make buildmacos` as well as the workable choice.

`build/llamacpp.zig` grew `appleCxxFlags` and `linkAppleLibcxx`; seven
`link_libcpp = true` became `false` (four there, three in `build.zig`).

Three things had to be learned the hard way, each a separate failed build:

- **`-I`, not `-isystem`.** `-isystem` places the C++ headers *after* clang's
  own include paths, and `<cstdio>` then fails with "tried including
  `<stdio.h>` but didn't find libc++'s `<stdio.h>` header".
- **`link_libcpp = false` also switches off the C++ header search**, so the
  include path has to be added by hand rather than merely dropped. Without it,
  `<cstdio>` is simply not found.
- **`libc++abi.tbd` is needed as well as `libc++.tbd`.** The `__cxa_*` guard,
  exception and personality symbols live there and `libc++.tbd` does not
  re-export them. This one only surfaced at `zig build test-port`.

And one subtlety about *where* to attach it: a static archive never links, so
the library modules look like they do not need the runtime — but
`reference.ggml_module` is also the root module of the `ggml_tests` executable,
and `portedTestModuleInner`'s module becomes the `test-port` executable. Both
need it. The symptom was ~40 undefined `__ZNSt3__1…` symbols from
`ggml-backend.o` and friends, long after `zig build` itself was green.

### What it does and does not restore

`zig build`, `zig build test`, `zig build test-port` and the gates that run on
our own binary all come back. **`make ref` does not**: it needs
`llama.cpp.zmake`'s `llama-cli`, which hits the same toolchain wall, and
Decision 18 says that repository is not ours to modify. So the two-sided
parity diff stays blocked until the toolchain is fixed properly.

`testcase/` holds the reproducer and the full diagnosis.

---

## Porting `ggml-backend.cpp` — the scheduler

2,443 lines into `src/ggml/backend.zig` (1,664) and `src/ggml/backend_sched.zig`
(1,740). The file the whole inference path runs through, and the one `PLAN.md`
calls "the hard one".

### It is C in a `.cpp` file

The first line of the C++ is `// Note: porting this file to C++ is a work in
progress`, and the measurement bears it out. The whole translation unit has
**zero `throw`, zero `catch`**, no `std::string`, no `std::map`, no
`std::function`, no `std::unique_ptr`, no templates and no virtual functions.
Two `std::vector`s and one lambda, all inside
`ggml_backend_sched_compute_splits`.

So the thing that made it hard was not the C++ — it was the algorithm. This
ported closer to `ggml-alloc.c` than to `gguf.cpp`, and took about as long,
because the five-pass assignment in `ggml_backend_sched_split_graph` has to be
reproduced step for step and no gate would catch getting a pass subtly wrong
except the end-to-end ones.

### The contract is 102 symbols, not 82

`PLAN.md` recorded 82; measured at the pin it is **102** of 243 exports. As
with `gguf.cpp`, both the plan's figures for this group were low, and both are
now corrected.

The linkage measurement held exactly: **not one of the 141 mangled exports is
referenced from any other object** in `libggml.a` or `libllama.a`. Checked by
intersecting this object's defined mangled symbols against every other object's
undefined list, not assumed from the earlier survey.

### Split in two, along the C's own seam

`backend.zig` takes the thin dispatch over the `iface` vtables — buffer types,
buffers, backends, devices, registries, events, the multi-buffer, the graph
copy and the CPU buffer. `backend_sched.zig` takes the scheduler.
`ggml-backend.cpp` marks the boundary itself with a `// scheduler` comment at
line 750. One 2,500-line Zig file whose halves never speak to each other would
have been worse.

### What the port had to do differently

- **`ggml_cgraph` is opaque in the `@cImport`**, so anything touching its
  fields uses `impl.CGraph`, as `alloc.zig` already did. The vtable entries
  still take the import's type, so the two are `@ptrCast` at each boundary.
  Same for `ggml_graph_view`, which is `graph.zig`'s and returns `impl.CGraph`
  by value.
- **`TENSOR_ALIGNMENT` moved into `impl.zig`.** It is a `ggml-impl.h` define
  and `context.zig` wants it too.
- **The four addressing macros became inline functions returning pointers.**
  `tensor_backend_id(t)` is an lvalue in the C — it appears on the left of an
  assignment — so `tensorBackendIdPtr` hands back a `*c_int` and the call
  sites dereference.
- **`SET_CAUSE`/`GET_CAUSE` are the `#else` arm.** The debug arm is behind
  `#if 0`, so they expand to nothing and `""`. Kept as empty functions rather
  than deleted, so the call sites still read against the C.
- **The `copy_experts` lambda became a named function taking six parameters.**
  Zig has no capturing closures; the C's captures are its arguments.
- **`i1` and `i0` are Zig integer type names.** Renamed `j1` and `j0`, digit
  for digit, as `CLAUDE.md` requires and `cpu/mulmat.zig` already does.

### The trap that cost the most: `[*c]` and `.src[i]`

`tensor.src[i]` on a `[*c]c.ggml_tensor` does not index the `src` array — Zig
resolves the subscript against the *pointer* and hands back the array type.
`@TypeOf(p.*.src[0])` and `@TypeOf(p.*.src)` both report
`[10][*c]struct_ggml_tensor`, which is how it was found; the error that
surfaces is `comparison of '[10][*c]...' with null`, twenty lines away from the
cause.

The fix is the convention `graph.zig` already had: narrow once at the
extraction site with `impl.one`, then the body reads like the C. Seven sites.

### Gates

Everything green, with the scheduler in the path:

| Gate | Result |
|---|---|
| `scripts/port-coverage` | 102 / 102 symbols, swapped into the build |
| `scripts/port-links` | 1,186 citations in 47 files |
| `make port` | loads Qwen3.5 and generates |
| `make parity-cli` | 6/6 prompts identical to the C |
| `make graph-diff` | 131 nodes identical |
| `make backend-ops` | 21,093 op configurations, 2/2 backends |
| `make probe` | ported code on the execution path |

`ar t libggml.a` no longer lists `ggml-backend.o`, and all 121 `ggml_backend_*`
symbols resolve to `libggml_zcu.o`.

**`ggml-backend.cpp` citations must be written `src/ggml-backend.cpp`.** A
second file of that name lives under `ggml-virtgpu/`, so the bare filename is
ambiguous and `port-links` refuses all 152 of them at once — the same trap
`ggml-backend-reg.cpp` hit, and the checker catching it twice is the rule
working.

### The scheduler had no gate, and now has one

The first injection round against `ggml-backend.cpp` was **badly designed**, and
saying so is the point: four of five faults "escaped", but three of those were
paired with a gate that structurally cannot see them. `make graph-diff`
exercises *constructors* and never builds a scheduler; pointing scheduler
faults at it proves nothing.

The harness itself was verified first — forcing `sched->n_splits = 1` aborts
`make port` — so the escapes were the pairing, not a broken injector. That
check is the reason the round was salvageable at all.

Re-paired, the picture is:

| Fault | Result |
|---|---|
| `buft_get_max_size` default `SIZE_MAX` → 0 | caught by `test-port` |
| multi-buffer tag always false | caught by `test-port` |
| split input copy not rewired | caught by `parity-cli` |
| pass 4 `view_src` propagation off | **escaped everything** |
| pass 2 CPU skip removed | **escaped everything** |

**Why parity is blind to a scheduling fault.** At `--temp 0` the sampler takes
an argmax, and `make backend-ops` has already shown Metal and CPU agree on all
21,093 op configurations. So moving an op from one backend to the other shifts
the last bits and not the chosen token. Token parity is the wrong instrument
for *where* an op ran — it only ever measured *what* it computed.

**`make sched-diff` closes most of it.** `harness/sched_dump.c` stands up two
stub devices that differ only in which ops they claim — FAST takes `GGML_OP_ADD`
and nothing else, SLOW takes everything — and prints the split count and the
backend of every node. `scripts/sched-diff` builds it against the reference C
and against the port and diffs: 16 assignments, identical. Negative-tested:
turning off pass 4's `view_src` propagation makes it fail.

**One fault still has no gate, and it is worth naming.** Removing pass 2's CPU
skip changes nothing observable, because pass 3 re-derives the same answer: the
skip leaves the adjacent nodes unassigned, and pass 3's "backend supporting the
most inputs" rule then puts them where pass 2 would have. Constructing a case
where the two disagree would need the two backends to share a buffer type, and
pass 3's upgrade rule converges there too. So the skip is close to semantically
neutral on small graphs — an optimization for large multi-GPU ones, where this
project has no hardware to test it. Recorded as uncovered rather than papered
over.

**A test expectation taken from the port is not a test.** The first version of
`"the scheduler splits where the supported op changes"` asserted three splits,
by reasoning about the passes. The C says **two** — the third node is an ADD
that FAST supports, but after the MUL its source sits in SLOW's buffer type,
which FAST cannot read, so pass 3 pulls it over. The number now comes from
`sched-diff` running the reference, not from reading our own code.

### One deliberate difference, found by review rather than by a gate

At `src/ggml-backend.cpp:1218` the C evaluates
`tensor_backend_id(src->view_src)` whenever `tensor_backend_id(src)` is -1 —
and `view_src` is null for most tensors, so `hash_id` inserts a **null key**
into a hash set of tensors. It burns one slot per reset and returns an id whose
stored value is -1, so the condition is false either way. The port guards it.
Nothing observable changes; no gate would have caught either choice. Same call
as `quantize_row_iq4_nl_ref`: where the C's answer is accidental, reproducing
it is not the goal.

---

## Reordering Stage 4, and what the measurement showed

Decision 40 moves `ggml-backend-meta.cpp` and `ggml-opt.cpp` to the end of the
ggml half, behind the `ggml-cpu/` C++. The reason is coverage, not size.

**`ggml-backend-meta.cpp` is ~96% unexecutable on this machine.**
`ggml_backend_meta_device` is constructed only under `LLAMA_SPLIT_MODE_TENSOR`
(`llama.cpp:176` and `:217`), our CLI never sets `split_mode`, and there is one
GPU. `alloc.zig:1318` guards the allocator path behind `buft_is_meta`, which is
always false. Of its 2,495 lines only the three `is_meta` predicates ever run,
and they always return false. Porting 2,400 lines of multi-GPU tensor-parallel
machinery that can be neither executed nor gated is the worst value in the
project; it waits until nothing else is left.

**`ggml-opt.cpp` carries an unspecified-behaviour case**, the third in this
project. `ggml_opt_dataset_shuffle` calls `std::shuffle` with a
`std::mt19937`. MT19937 is fully specified and portable — Zig's std has no
implementation, but it is about forty lines. `std::shuffle`'s *distribution* is
not: `std::uniform_int_distribution` is implementation-defined, so libc++ and
libstdc++ produce different orders from the same generator state. Same
treatment as `quantize_row_iq4_nl_ref` and the `qsort` tie-break — implement
the generator faithfully, document the divergence, do not chase it.

**Every symbol count the plan carried for this stage was low.** gguf 44 → 61,
backend 82 → 102, meta 4 → 8, opt 9 → **37**. Four for four. The original
survey counted something other than the unmangled exports; run `port-coverage`
before estimating anything in Stage 4.

## Porting `ggml-cpu/binary-ops.cpp`

154 lines into `src/ggml/cpu/binary_ops.zig`, 4 symbols. The first of the
`ggml-cpu/` C++.

**Its only gate is `backend-ops`, and finding that out cost a round of
injections.** All four faults — `add` computing `a - b`, the vDSP routine
swapped for the wrong one, vDSP's operands reversed, the broadcast modulo
dropped — passed `make parity-cli` 6/6. Even `add` becoming `sub`.

The reason is not subtlety, it is that **the code never runs**. Putting
`impl.abort` at the top of `ggml_compute_forward_mul` and running `make port`
completes normally and never hits it: on a Metal machine the model executes
entirely on the GPU, and the CPU op kernels are dead during inference. Token
parity cannot gate a kernel that is never called.

`make backend-ops` is the one gate that reaches them, because it explicitly
runs CPU against Metal across 21,093 op configurations. Re-run against it, all
four faults are **caught**:

| Fault | `parity-cli` | `backend-ops` |
|---|---|---|
| `add` computes `a - b` | escaped | caught |
| vDSP routine swapped for the wrong one | escaped | caught |
| vDSP operands reversed | escaped | caught |
| broadcast modulo dropped | escaped | caught |

That generalises to the whole group — `ops.cpp`'s 8,436 live lines, `vec.cpp`,
`unary-ops.cpp`, the repack cluster. **There is exactly one gate for the CPU op
kernels, and it must be run after every one of these files.** A green
`make validate` says nothing about them: it does not run `backend-ops`.

### Templates map one-for-one onto comptime

The C++ is three nested function templates — over the scalar operation and
over the three element types — expanded by a seven-arm `if` on the runtime type
triple. Zig's `comptime` parameters express exactly that, so `applyBinaryOp`
takes the same four compile-time arguments and `binaryOp` is the same seven-arm
chain. Nothing moved between compile time and run time in either direction.
This is the case `PLAN.md` predicted when it said templates "usually come out
cleaner"; it is the first time that has actually been true.

The one place the shapes differ is the operation itself. The C++ passes
`op_add` and friends as function-pointer template arguments and then, under
Accelerate, *compares that pointer* against `op_add` to pick a vDSP routine.
A comptime enum carries both jobs and makes the second a switch rather than a
pointer identity test.

### The Accelerate path is live, and keeping it is not optional

`build/llamacpp.zig` defines `GGML_USE_ACCELERATE`, so the all-`f32` case goes
through `vDSP_vadd`, `vDSP_vsub`, `vDSP_vmul` and `vDSP_vdiv` instead of the
scalar loop. These are element-wise with no accumulation, so they are
IEEE-exact and the choice costs nothing numerically — it is taken for
throughput. Dropping it would be a silent performance regression that **no gate
in this project measures**, which is exactly why it is called out here.

`vDSP_vsub`'s argument order is the trap: the C passes `(src1, src0)` and that
is what yields `src0 - src1`. Reversing it is an easy, plausible-looking edit.

### Two more citation traps, both the ambiguity class

- `binary-ops.cpp:49` is the `template <...>` line; the definition starts at
  50. `port-links` caught the off-by-one.
- **Five files named `common.h` live in the reference tree** (`common/`,
  `ggml-cpu/`, `ggml-cann/`, `ggml-cpu/amx/`, `ggml-metal/kernels/`), so the
  bare name is ambiguous and every citation to it was refused. Qualified to
  `ggml-cpu/common.h`. That is the third time the checker has caught this —
  after `ggml-backend-reg.cpp` and `ggml-backend.cpp`.

### And more reserved integer names

`i01`, `i02`, `i03`, `i10`, `i11`, `i12`, `i13` all parse as Zig integer types.
Renamed digit for digit, as `cpu/mulmat.zig` already does. The list in
`CLAUDE.md` names `i1`, `i2`, `i3`, `i11`, `i12`, `i13`; it is really *any*
`i` followed by digits, and this file hit four more of them.

### One thing `port-coverage` needed

Anything under `src/ggml/cpu/` imports `../impl.zig`, which a per-file module
root cannot see — `error: import of file outside module path`. Its
`report_unit` entry is rooted at `ported.zig` like the other `cpu/` units,
rather than at the file itself.

## Porting `ggml-cpu/unary-ops.cpp`

337 lines into `src/ggml/cpu/unary_ops.zig`, 23 symbols, compiled clean on the
first attempt. Same gate situation as `binary-ops.cpp`: **`backend-ops` only**.

### libm, not Zig's builtins

Every transcendental goes through an `extern` declaration of the libm function
the C calls — `expf`, `tanhf`, `logf`, `sqrtf` and the rest — rather than
`@exp`, `@tanh` or `std.math`. Zig's builtins lower to LLVM intrinsics, which
are free to differ from the platform's libm in the last bit.

The reason to care is that **nothing here would catch it**: `make backend-ops`
compares CPU against Metal with a *tolerance*, not bit-for-bit, and these
kernels never run under `make port`. Unlike the NEON dot products, there is no
`vecdot-prefix` equivalent to pin the shape. Calling the same function the C
calls removes the question instead of measuring it, which is the right trade
when no oracle exists.

### Three traps in the arithmetic

- **`op_expm1` is `expf(x) - 1.0f`, not `expm1f(x)`** (unary-ops.cpp:76). The
  two differ near zero, and `op_elu` four functions earlier *does* use
  `expm1f`. Reading the file quickly, this looks like a bug to tidy; it is
  not ours to tidy.
- **The two template families have different assertions.** `apply_unary_op`
  requires `ggml_is_contiguous_rows` on both tensors; `apply_unary_op_functor`
  requires only `ggml_is_contiguous_1`. Collapsing them into one generic —
  which Zig makes tempting, since the bodies are otherwise identical — would
  silently tighten one path or loosen the other.
- **`xielu` reads `op_params` indices 1 to 4**, not 0 to 3.

### The entry points are generated, and the C's are not

The C writes out twenty-two one-line forwarders by hand. Here they come from a
`comptime` loop over the `Op` enum with `@export`, so the list cannot drift out
of step with the operations it dispatches to. `port-coverage` confirms the
generated names match the C's exactly, 23/23 — which is the check that makes
the generation safe rather than clever.

### Not ported, deliberately

`unary_op_params` (unary-ops.cpp:157) is a third template family over
`float (*)(float, ggml_tensor *)`. Nothing instantiates it, so the C++ emits no
code and it has no symbol. Recorded so the absence reads as a decision.

### What `backend-ops` can and cannot see, measured

Five faults injected, gate = `backend-ops`:

| Fault | Result |
|---|---|
| `relu` threshold `>0` becomes `>1` | caught |
| `hardsigmoid` divisor 6 becomes 5 | caught |
| `xielu` reads `op_params` 0..3 instead of 1..4 | caught |
| `op_expm1` uses `expm1f` instead of `expf(x)-1` | **escaped** |
| `softplus` cutoff 20 becomes 2 | **escaped** |

The two escapes are not a reachability problem — putting `impl.abort` in the
generated `softplus` makes `backend-ops` exit 134 (SIGABRT), so the kernel is
definitely run. They are a *metric* problem.

`test-backend-ops` compares with **NMSE against a `1e-7` threshold**, over
inputs drawn uniformly from **[-150, 150]**. The softplus fault changes the
result only for `x` in (2, 20), by `log(1 + e^-x)` — an error that decays
exponentially — while the signal is dominated by large `|x|`:

    Σerr²  ≈ ∫₂²⁰ e^(-2x) dx / 300  ≈ 3e-5
    Σa²    ≈ ∫₀¹⁵⁰ x²    dx / 300  ≈ 3750
    NMSE   ≈ 8e-9  <  1e-7

`expm1f` escapes for the same reason from the other end: it differs from
`expf(x)-1` only near zero, a thin slice of a wide range.

**So the rule for this group is: `backend-ops` catches faults that are wrong
across the input domain, and misses faults confined to a band of it.** That is
a different limitation from the one already recorded for it ("cannot catch a
consistently-wrong constructor"), and it is the reason the float dot products
in `vec.cpp` get goldens rather than relying on this gate — see
`scripts/vec-golden`.

### `port-links` earned its place again

Seven citations in this one file pointed one to three lines early, all at
`template <...>` headers rather than the definition beneath them. Same class as
`binary-ops.cpp:49`. Writing these by eye does not work; the checker is the
only reason they are right.

## Porting `ggml-cpu/vec.cpp` — and the gate that had to exist first

484 lines into `src/ggml/cpu/vec.zig`, 10 symbols. The file is small; the gate
in front of it was the work.

### Three dot products, three different accumulator shapes

`ggml_vec_dot_f32`, `_f16` and `_bf16` look interchangeable and are not. Each
accumulates differently, and each difference is load-bearing:

| | body | reduce | tail |
|---|---|---|---|
| `f32` | 4 × `f32x4`, `vfmaq_f32`, step 16 | tree 2→1, then **pairwise** `vaddvq_f32` | `f32` scalar |
| `f16` | 4 × `f16x8`, step 32 — **half precision** | tree, widen both halves, add, pairwise | **`double`** |
| `bf16` | no SIMD arm compiles here at all | — | **`double`**, plain loop |

A port that reads the f32 kernel and then "tidies" the other two to match is
wrong three ways: the f16 body really does accumulate in half precision, and
the two tails use a wider type than their bodies.

### The gate did not exist, so `vec.cpp` could not be ported first

`PLAN.md` had flagged this as a decision to take deliberately rather than by
default. Nothing in the project could see these kernels:

- **`make port` and `make parity-cli` never reach them.** On a Metal machine
  the CPU kernels do not run during inference — measured on `binary-ops.cpp`,
  by putting an `impl.abort` in one and watching generation finish.
- **`make backend-ops` compares with NMSE at `1e-7`.** The `unary-ops.cpp`
  round had already shown that misses a fault confined to a band of the input
  domain.
- **`scripts/vecdot-prefix` and `scripts/vecdot-golden` cover only the
  *quantized* kernels.**

So `harness/vec_golden.c` + `scripts/vec-golden` + `src/ggml/cpu/vec_golden.zig`
+ `src/ggml/cpu/vec_testing.zig` were built first, against the C, and the port
was written afterwards. Same split as `src/ggml/cpu/quants/`: a generated
golden file and a hand-written driver. Contraction **on**, because the fusion
sites are explicit `GGML_F32_VEC_FMA` intrinsics rather than expressions a
compiler chose — the rule from `CLAUDE.md`, "Float contraction".

### The gate passed, then failed, and the failure is the point

The port matched all six patterns at both lengths on the first run. That is a
reason for suspicion, not confidence, so the faults went in — and **one
escaped: replacing the pairwise `vaddvq_f32` with an ordered
`@reduce(.Add, ...)` passed every pattern at both lengths.**

That is the exact trap `CLAUDE.md` already warns about, invisible to the gate
built to catch it. Three measurements pinned down why:

- A brute-force search over random 4-lane vectors puts the two reductions at
  odds **23.5% of the time**. Six patterns agreeing was luck, not a property
  of the reduction.
- They differ only when the two **pairs** differ enough in magnitude for the
  grouping to change a rounding: `vaddvq_f32` computes `(l0+l1) + (l2+l3)`,
  an ordered sum computes `((l0+l1)+l2) + l3`.
- **Lane `k` of the surviving accumulator holds every element with
  `i % 4 == k`**, so keying magnitude on `i % 4` is the only way to shape the
  lanes from the input. `lopsided` varies magnitude per *element* and not per
  *lane*, which is precisely why it cannot reach this.

Hence a seventh pattern, `skewed`: three small lanes against one dominant one,
`x[i] = lcg() * (i % 4 == 3 ? 1.0f : 0x1p-8f)`. The scale is a power of two so
the scaling is exact and the pattern tests the reduction rather than the
multiply. Predicted `0xC035D443` from a standalone reproduction before
touching the harness; the regenerated golden agrees.

With it, 5/5:

| Fault | Result |
|---|---|
| f32 reduce: tree → left-to-right | caught |
| f32 reduce: pairwise → `@reduce(.Add, ...)` | caught (**escaped before `skewed`**) |
| f16 accumulator widened to `f32` | caught |
| f32 FMA unfused (`a*b` then `+`) | caught |
| bf16 accumulates in `f32`, not `double` | caught |

### Two injections that were wrong, and what they cost

Worth recording because both looked like escapes and neither was:

- **"f16 body accumulates in f32"** computed the product in `f32` and then
  stored back to `f16` every iteration, so the accumulator was still `f16`
  between steps — nearly a no-op. Rewritten to widen the accumulator itself,
  it is caught.
- **"f16 tail product widened before multiply"** is a *provable* no-op. `f16`
  has an 11-bit significand, the product needs at most 22, and `f32` holds 24,
  so the `f32` multiply is exact and widening either side of it cannot change
  the result. Checked exhaustively over 576M pairs: zero differ. Dropped
  rather than counted as an escape.

**An injection that does not change the computation is not evidence about the
gate.** Both of these would have been written up as holes in the golden set.

### A correction

An intermediate measurement here reported `lopsided` as bit-identical to
`random`, which would have made it a dead pattern. It is not — the reproduction
had mistranscribed its second scale as `1e4f` where the harness says `4.0f`.
Against the actual code it gives `0x3B575ABF` and discriminates summation order
as documented.

### Two transitions the swap forced

- `ggml_table_gelu_f16` and `ggml_table_gelu_quick_f16` (65,536 entries each)
  were *defined* by `vec.o` and declared `extern var` by `cpu/convert.zig`,
  which fills them. They are now `pub export var` in `vec.zig`, and
  `convert.zig` imports them instead.
- The NEON helpers `vec.cpp` needs that the quant kernels do not —
  `dup_n_f32`, `sub_f32`, `div_f32`, `fms_f32`, `fma_f16`, `add_f16`,
  `cvt_f32_f16_half` — are local to `vec.zig` rather than added to
  `quants/arm/neon.zig`. `neon.addvq_f32` and `neon.fma_f32` are imported from
  there, because those two are the subtle ones and there should be one
  definition of each.

### `port-links` again

One citation wrong: `ggml_float` given as `ggml-impl.h:60`, where it is
`vec.h:15`. Fifth file in which writing a citation by eye produced a wrong one.

## `scripts/ops-diff` — a value-level oracle for `ggml-cpu`, built before the port

`ops.cpp` is 8,436 live lines and 92 symbols, the largest file left in ggml.
Before writing a line of it, the question from the `vec.cpp` round was applied:
**what oracle does this need?** The answer was that it had none.

- `scripts/graph-diff` runs with `no_alloc = true`. It checks the *structure* a
  constructor produces and never computes, so a kernel that builds the right
  graph and fills it with wrong numbers is invisible to it.
- `make backend-ops` executes, but compares CPU against Metal with **NMSE at
  `1e-7`**. The `unary-ops.cpp` round measured what that misses.
- `make port` and `make parity-cli` never reach the CPU kernels on a Metal
  machine at all.

So `harness/ops_dump.c` + `scripts/ops-diff`: 109 ops executed on the CPU
backend with fixed inputs, output compared on **bits** against the stock C.
Same two-build shape as `graph-diff` and `sched-diff`. One thread on both
sides — the CPU backend sums per-thread chunks, so a different thread count is
a different summation order and a different last bit.

### `--diff` is not what it sounds like

Worth stating plainly because it was nearly relied on. `CLAUDE.md` said
`backend-ops --diff` "compares output against the C reference (25,176 lines
identical)". It compares the two **logs**. `test-backend-ops` prints an op
descriptor and `OK`, and prints the error value *only when it exceeds the
threshold* (`test-backend-ops.cpp:1476`), so a clean log carries no computed
values whatsoever. The diff is structural. It does not make `backend-ops` a
value-level oracle, and the wording has been corrected.

### The baseline failed four times, and only one was a port bug

`ops.cpp` was unported and byte-identical on both sides, so the baseline should
have been clean. It was not, four separate times:

| What failed | Cause | Kind |
|---|---|---|
| `xielu`, Q6_K `mul_mat` | reference built by **Apple clang** | gate |
| `rope`, `rope_ext`, `timestep_embedding` | reference built at a **different optimization level** | gate |
| `xielu` | **real: unnamed FMA** in `unary_ops.zig` | **port** |
| Q6_K `mul_mat` | harness **quantized inputs with the library under test** | gate |

Three of the four were defects in the gate. That ratio is the argument for
building a gate and breaking it before trusting it, and for building it
*before* the port rather than after: had `ops.cpp` already been ported, every
one of these would have read as a porting bug in 8,436 lines of new code.

**The compiler confound is specific to this gate.** `graph-diff` and
`sched-diff` link the Apple-clang CMake build and are right to: structure does
not depend on the compiler. Bits do. `ops-diff` links `llama.cpp.zmake` — the
stock C built by the same Zig toolchain — so the port is the only variable, and
it must be built `--release=fast` to match.

### The real bug: `xielu` was 1 ULP out

The C compiles `alpha_p * x * x + beta * x` at `-ffp-contract=on` and fuses it;
strict Zig did not. `backend-ops` passes it, which is the band-blindness
already recorded for that gate showing up again.

Which multiply clang fuses is the real question, since **two feed each add**.
Measured over 200k inputs rather than guessed: for `x > 0`,
`fma(alpha_p*x, x, beta*x)` matches **all 99,596** positive samples, where
fusing the `beta` term matches 91% of them. That is the **left** operand of the
`+`. The negative branch is ambiguous — both fusings agree on every sample — so
the same left-operand rule settles it rather than the data.

This is the project's contraction rule working as written: *name a fusion only
when something can tell you you got it wrong*. For the quantizers nothing can,
so they stay plain. For `xielu` something now can, so it is named.

### The quantizers disagree, and the harness has to route around it

The Q6_K `mul_mat` difference was not the dot product. Measured directly:
`ggml_quantize_chunk` produces **different Q6_K bytes** in the two builds,
while Q4_K and Q8_0 match. That is the documented, accepted consequence of
leaving the quantizers' FMA sites unnamed — the deliverable reads model files
and never writes one.

But it meant the two `mul_mat` kernels were being fed different weights, so the
comparison was not about the kernel at all. The reference run now writes its
quantized blocks to a file and the ported run replays them. The LCG is advanced
identically on the reading side so the two runs stay in step.

### Input amplitude is part of the gate

First injection round: `softplus` with its cutoff moved from 20 to 2
**escaped** — the same fault that escaped `backend-ops`. Not a weakness in the
comparison this time: the LCG produces values in `[-1, 1)`, so no input ever
reaches a threshold of 2 *or* 20 and both arms take the same branch for every
element. The fault was unreachable.

The activation ops are now run twice, at amplitude 1 and at 30. **A gate's
input distribution is as much a part of it as its comparison**, and a
threshold fault is the case that shows the difference.

## Porting `ggml-cpu/ops.cpp`

88 symbols, 8,436 live lines, ported in one sitting and swapped in whole.
`src/ggml/cpu/ops/` holds it in 19 files that follow the C's own
`// ggml_compute_forward_xxx` banners, plus `common.zig` and `vecinline.zig`
for what they share.

### How: the oracle first, then seven workers

The `vec.h` inlines every family calls — `mad`, `scale`, the gelu and glu
variants — were ported first into `vecinline.zig`, by hand, because several
have live NEON arms that compute **in half precision** (`vfmaq_f16`,
`vmulq_f16`) for the first `n & ~31` elements and in `f32` for the rest. The
split is real and a unit test pins it with a value that rounds differently on
each side of it.

Then the remaining 75 symbols went to seven parallel workers, one group of op
families each, every one writing only its own files and checking them with a
per-file compile of the contract (`scripts/ops-check --only`, since removed:
`port-coverage` covers the contract now that the port is swapped). While they
worked, `harness/ops_dump.c` grew from 109 cases to ~300 covering every
family, and `scripts/ops-diff` started running the whole suite twice, at one
thread and at three. The baseline — both sides still the C `ops.cpp` — had to
pass before anything was swapped, and it did not, at first; see below.

`refAllDecls` is not a compile check for an `inline` or generic helper: it
does not analyse their bodies. An injected type error in an unused `inline fn`
compiled clean. `vecinline.zig`'s tests call every helper for that reason.

### The baseline failed on code ported before this sitting

Extending `ops-diff` before the swap paid for itself before a line of the
port was wired in. With `ops.cpp` identical on both sides, three new conv
cases failed by 1 ULP. All three reach the already-ported
`ggml_vec_dot_f32`, and its scalar tail was the bug — though not the bug it
first looked like.

The C's tail is `sumf += x[i]*y[i]`, one expression, so clang contracts it.
Fusing every step fixed the two-element dot products and broke the
eleven-element ones. The disassembly of the reference `vec.o` showed why: the
loop vectorizer turns that loop into a **strict, in-order reduction** —
products in groups of four with `fmul.4s`, **unfused**, added into `sumf` one
lane at a time — and only the `t % 4` left over run scalar, as `fmadd`,
**fused**. `fmuladd` lets the backend choose, and it chooses differently in
the two halves. That model matches the reference bit for bit at every `n`
from 1 to 47, 3,000 random rows each. It is `vectorizedTail` in `vec.zig`.

The `vec_golden` gate passed the unfused tail, the fused one *and* the split
one: its 519-element case leaves a seven-element tail after a sum near 8,
where none of the three moves a bit. It now has six short lengths. Two were
not enough — an all-unfused tail still passed with 11 and 27, because only a
few of the seven input patterns discriminate, each about a third of the time.
With six, both wrong variants fail.

### What the swap itself found

With the port wired in, 13 of 227 cases failed. Three causes:

- **Apple's merged sine and cosine.** Every f32 rope and
  `timestep_embedding`, a few ULP on rare elements. Clang folds `sinf(x)` and
  `cosf(x)` of one argument into a single `__sincosf_stret` call, and that
  routine does not round as the separate ones do: over 10.9M arguments, `sin`
  differs in 4.0% and `cos` in 0.5%. The reference `ops.o` imports
  `___sincosf_stret` and neither `_sinf` nor `_cosf`. `common.sinCos` calls
  the same routine. f16 rope passed throughout — its rounding to half
  precision hid the difference.
- **The vectorized reduction again**, in `ssm_conv`'s window sum and in
  `ssm_scan`'s state update — Mamba-1's despite an `expf` in the loop body:
  the vectorizer scalarizes the call per lane. Here it also showed the
  guards, read off the LLVM IR of the same loop: more than three iterations,
  and for every array the loop loads, `(store_row - load_row)` as an unsigned
  byte difference at least 64. `ssm_scan` reads the previous state from the
  buffer it is writing for every token after the first, so those tokens take
  the all-fused scalar loop and the first one does not.
  `common.strictLoopVectorized` encodes both.
- **A harness bug, not a port bug.** `im2col_back` varied run to run on both
  sides. `ggml.h` labels its first argument "convolution kernel"; the kernel
  reads `src[0]` as the gradient, and `ggml.c:7032` passes `(grad, kernel)`.
  Passed in header order it read past the end of a 108-float tensor.

**The rule this adds to "Float contraction" in `CLAUDE.md`**: `@mulAdd` is
right for an elementwise `a*b + c`, which the vectorizer keeps fused as
`fmla`. For a *reduction* loop over contiguous memory it is right only for the
scalar remainder. Which loops the vectorizer takes is a cost-model decision,
so each such site is one the oracle confirmed, not one read off the source.

### Fault injection

| Fault | Caught by | Cases |
|---|---|---|
| `rwkv_wkv6` state update unfused | ops-diff | 1 |
| `gated_delta_net` drops its last row | ops-diff | 5 |
| imrope condition inverted | ops-diff | 4 |
| flash-attn softcap rescale skipped | ops-diff | 1 |
| `getThreadRange` drops the remainder rows | ops-diff, **3 threads only** | 4 |
| split-KV partial merge unfused | ops-diff, **3 threads only** | 2 |
| tiled flash-attn GEMM unfused | ops-diff | 6 |
| tiled flash-attn sink merge unfused | ops-diff — **escaped first** | 2 |

Two of eight show only at three threads: at one thread every row split is the
trivial one and the split-KV path never splits. The escape was the input
distribution again: with sinks in `[-1, 1)` a sink rarely beats a row's
maximum score, the `ms = 1` branch is exact, and fused and unfused agree. The
sinks are now drawn from `[-4, 4)`. The flash-attention fast paths needed
their own cases at all — split-KV wants one query row against ≥512 keys,
tiled wants ≥64 query rows — and the first shapes reached neither.

### CPU-only inference found three more, and a compiler bug

All of that passed, and so did every token gate — none of which reaches a CPU
kernel on this machine. `scripts/parity-port` was repointed at
`llama.cpp.zmake` (the open item from Stage 0) and given `--cpu`, which loads
the model on the CPU device alone. **4 of 6 prompts diverged.**

Putting the C `ops.cpp` back did not change that, so it predated this port.
`harness/node_dump.c` — the eval callback hashing every node of a decode —
named the first differing node: a `MUL_MAT` with **Q5_K** weights. `ops-diff`
had Q4_K, Q6_K and Q8_0 only. With every quantized type added at 1 and 7
columns, three failed:

- **`q5_K`**: `sumf += d * sumi - dmin * sumi_mins`. Clang contracts the
  right-hand side alone — `fnmul`, `fmadd`, `fadd` in the reference — and the
  port had a fused chain into `sumf`.
- **`tq1_0` and `tq2_0`**: `sumf += d * (float) isum` fuses; the port's shared
  epilogue returned `d * isum` for the caller to add.

All three had passed their vec_dot goldens. With them fixed, CPU-only parity
is 6/6 and every node of the prompt and one decode step is bit-identical on
three Qwen3.5 quantizations at one and four threads. That check is now
`make node-diff`.

`node-diff --gpu` then failed on outputs that were *identical*: the
scheduler's input copies carried an `op` of `GET_ROWS` where the C's are
`NONE`. Traced to `dupTensorLayout`:

```zig
for (0..c.GGML_MAX_DIMS) |i| dup.*.nb[i] = tensor.*.nb[i];
```

**Zig 0.16 types `p.*.arr[i]` as the whole array when `p` is a C pointer**,
and indexes in steps of the array's size. That loop copied 128 bytes from
`nb`: the source's `op`, `op_params`, `flags` and `src[0..2]` landed in every
copy. `p[0].arr[i]`, `*T`, `[*]T` and `?*T` all index correctly; only
`[*c]T` deref does not, in Debug and ReleaseFast alike. Its unit test passed
because it compared `t.*.nb[i]` with `dup.*.nb[i]` — two equally wrong reads.
It was the only occurrence in the tree. The test now checks through
single-item pointers and asserts the copy has no op, flags or sources; it
fails on the old loop.

### Where the C's answer is unspecified

- **`argsort` and `top_k` among equal keys.** `std::sort` and
  `std::partial_sort` order ties in a libc++-specific way. The port sorts
  stably with the C's comparator, so ties resolve by index. Identical whenever
  keys are distinct, which every `ops-diff` input is.
- **`flash_attn_back`** is ported but unreachable: its constructor aborts at
  `ggml.c:5523`, "TODO: adapt to ggml_flash_attn_ext() changes".

### Not ported, deliberately

The C's `assert`s compile out under `NDEBUG`. Most of the port keeps them as
`std.debug.assert`, the convention elsewhere in the tree — but in ReleaseFast
that is an optimizer **assumption**, not a no-op, so one that can be false at
run time is undefined behaviour where the C simply carries on.
`soft_max`'s `assert(sum > 0.0)` is one: a fully masked row gives a NaN sum. It
is a comment, not an assert.

## Porting `ggml-cpu/llamafile/sgemm.cpp`

530 lines of Zig from **391 live** C++ lines of 4,164 — the templated x86 bulk
is behind `__AVX__`, `__AVX512F__` and `__AVX2__`. One exported symbol,
`llamafile_sgemm`, which returns **false** when it has no kernel for a shape
or type pair; `cpu/mulmat.zig` then computes the product itself. Returning
false is a normal outcome, not an error.

Four instantiations are live and nothing else: `tinyBLAS<4, f32x4>`,
`tinyBLAS<8, f16x8>`, and `tinyBLAS_Q0_ARM` over `block_q8_0` and
`block_q4_0`. `BF16`, `Q5_0` and `IQ4_NL` all reach a `return false` here.
The C's six template parameters collapse to three, because `D == V` and
`TA == TB == T` at both float instantiations and `TC` is always `f32`.

### The gate was three-quarters there, and the quarter mattered

`PLAN.md` said the gate already existed — `ops-diff` runs f32 and quantized
`mul_mat` at shapes that reach it. **Measured, that was true of three of the
four instantiations.** The f16 kernel needs `n >= 8` (sgemm.cpp:3975) and the
harness topped out at `n = 7`, so nothing exercised it.

Two `n = 8` cases closed it, and reachability was then *proved* rather than
assumed: disabling the `llamafile_sgemm` call in `mulmat.zig` moves exactly
five rows of `ops-diff` output — the f32 `n=4` case, the two new `n=8` cases,
and q4_0/q8_0 at `n=7` — at both thread counts. That is the positive-execution
probe the project already uses for the CPU dispatch, applied to a fast path.

**A gate that covers a file is not the same as a gate that covers every arm
of it**, and the plan's one-line claim could not tell the difference.

### The gate caught the port's one real bug immediately

The C dispatches `mnpack<4, 6, 4>(m, n, SIZE_N, 12)`. The trailing `12` is
`BN`, an ordinary argument; the `4` before it is `BM`, a template parameter.
The port took the `12` as `BM`, so the kernel asserted `m % 48 == 0` where the
C asserts `m % 16 == 0`, and aborted on the first run. Positional template
arguments invite exactly this, and one `ops-diff` run named it.

### Injection: 6/6, and one non-fault

| Fault | Result |
|---|---|
| f32/f16 FMA unfused | caught |
| `hsum` pairwise → ordered `@reduce` | caught |
| q4_0 low-nibble bias 8 → 7 | caught |
| q4_0 high-nibble shift 4 → 3 | caught |
| q8_0 dot reads `lo` twice instead of `lo`+`hi` | caught |
| f16 entry threshold 8 → 2 | caught |
| Q0 tile-shape cap `min(.,3)` → `min(.,2)` | **no-op** |

The last one is not an escape. The tile shape only decides which output
elements are grouped and which thread computes them; each element is still
one thread's exact dot product over the same `l` order, written once. It
cannot change a bit. Third time the "confirm the injection changes the
computation" rule has paid for itself.

### The f16 NEON helpers moved before they were copied

`cpu/vec.zig` had kept `fma_f16`, `add_f16` and `cvt_f32_f16_half` local,
deliberately, while it was the only user. `sgemm.zig` is the second, so they
moved to `quants/arm/neon.zig` and `vec.zig` aliases them — the morning's
tidy rule applied *before* the third copy existed rather than after.

### `port-links` needed 18 corrections in this file, nearly all one class

The citations backticked `hsum(float32x4_t)`, `class tinyBLAS`,
`tinyBLAS::gemm`. The checker wants the **bare identifier** before the
parentheses, with the overload or class named in the prose. Worth knowing
before porting another file full of C++ member functions.

## The vtable cluster, measured

`traits.cpp`, `repack.cpp`, `arch/arm/repack.cpp` and `ggml-cpu.cpp` — 6,180
live lines. The plan says they move as one unit. **They do, and the reason is
five symbols, not the 261 mangled exports the four objects carry.**

The first measurement asked the wrong question: intersecting the cluster's
mangled exports with the undefined symbols of every *other* object in
`libggml.a` gives **zero**, so the cluster's *external* contract is a pure C
ABI like the other twenty. That says nothing about coupling *within* it,
which is what forces the unit.

Measuring that instead, the whole cross-reference matrix is:

| from | to | refs | kind |
|---|---|---|---|
| `repack.cpp` | `arch/arm/repack.cpp` | 28 | **all unmangled** — `ggml_gemm_*`, `ggml_gemv_*` |
| `arch/arm/repack.cpp` | `repack.cpp` | 7 | **all unmangled** — the `_generic` fallbacks |
| `repack.cpp` | `traits.cpp` | 4 | **mangled** |
| `traits.cpp` | `ggml-cpu.cpp` | 1 | **mangled** |
| `repack.cpp` | `ggml-cpu.cpp` | 1 | unmangled — `ggml_backend_cpu_reg` |

The five C++ ones are precisely:

- `ggml::cpu::tensor_traits::~tensor_traits()`
- `ggml::cpu::extra_buffer_type::~extra_buffer_type()`
- `typeinfo for ggml::cpu::tensor_traits`
- `typeinfo for ggml::cpu::extra_buffer_type`
- `ggml_backend_cpu_get_extra_buffer_types()` — returns
  `std::vector<ggml_backend_buffer_type_t> &` (ggml-cpu.cpp:42)

So the shape of the work is: `traits.cpp` declares two abstract base classes,
`repack.cpp` derives from them, and `ggml-cpu.cpp` hands the derived instances
out through a `std::vector` reference. The bases become a Zig vtable struct,
the derived types become structs carrying a pointer to one, and the vector
becomes a static array — the same move `backend_reg.zig` already made for the
registry.

**`repack.cpp` and `arch/arm/repack.cpp` are mutually dependent across 35
plain C symbols** and would have to move together regardless of any C++.

## `GGML_USE_CPU_REPACK` was off, and `repack` was dead code

Found while checking the vtable cluster's gate before porting it. The chain,
each step measured:

- **Upstream's CMake defaults `GGML_CPU_REPACK` to `ON`**
  (ggml/CMakeLists.txt:152), and `cmake-build/apple/CMakeCache.txt:407`
  confirms the reference CMake build has it.
- **Neither `build/llamacpp.zig` nor `llama.cpp.zmake/build.zig` defined
  it.** Compiling `ggml-cpu.cpp` both ways: **0** references to the repack
  buffer type without the flag, **1** with it.
- So the buffer type was never registered, the extra-buffer list was empty,
  and `ggml_cpu_extra_compute_forward` always returned false.
- **`repack.cpp` (3,164 live lines) and `arch/arm/repack.cpp` (2,528)
  compiled into the library and could never execute** — 5,692 lines in the
  same position as `hbm.cpp` and `amx/`.

Confirmed by execution, not inference: panicking on the one entry point,
`ggml_cpu_extra_compute_forward`, a CPU-only decode of Q4_K_M and IQ4_XS
completed without hitting it. With the flag on, the same probe fires.

The flag is now set on both sides and the baseline is green with it:
`ops-diff` 596/596, `node-diff` 2750 nodes at 1 and 4 threads,
`parity-port --cpu` 6/6.

### Two false conclusions along the way

Worth recording because both looked like findings:

- **"repack never fires with the flag on."** The probe script grepped the
  *parity harness's* stdout, and the panic goes to the subprocess's stderr.
  The detection was broken, not the path.
- **"enabling the flag broke the ported CPU path."** `parity-port --cpu`
  came back `ported=134` (SIGABRT). The library still had the panic probe
  compiled in — the source was restored but not rebuilt. Rebuilt clean, it
  passes 6/6.

**Restore the source *and* rebuild before reading a gate**, and check where
a probe's output actually goes before concluding from its silence.

### The guard this needed

`scripts/check-reference-pin` already asserted that the reference compiles
the same upstream commit. A commit is not the whole contract: a build
**flag** that differs changes which code compiles on each side, and with
`GGML_USE_CPU_REPACK` on one side only every bit-exact gate diverges for a
reason that has nothing to do with the port. The guard now checks the flag
too.

Negative-testing it caught a bug in the check itself: `grep -c` prints `0`
**and** exits 1 when there is no match, so `|| echo 0` appended a second
line and the integer comparison never ran. The check was silently useless
until the fault injection showed it.

## Porting the vtable cluster: `traits.cpp`, `ggml-cpu.cpp`, both `repack.cpp`

The four files that had to move together, and the gate that did not exist
when they did.

### The symbol count said 100% for a port that could not work

`scripts/port-coverage` reads a C++ translation unit's contract as its
*unmangled* exports. For `repack.cpp` that is 36 symbols — the gemv, gemm
and `quantize_mat` kernels — and all 36 were ported before any of the
dispatch was. `cluster-check` reported `36 / 36 symbols (100%)`.

It was not close to working. Everything that decides *when* those kernels
run has C++ linkage and never appears in `nm -gU | grep -v _Z`:

- the `CPU_REPACK` buffer type and its five iface functions,
- `extra_buffer_type::{supports_op, get_tensor_traits}`,
- the sixteen `tensor_traits<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>`
  instantiations and their `work_size` / `compute_forward` /
  `forward_mul_mat` / `forward_mul_mat_id` / `repack`,
- `ggml_repack_get_optimal_repack_type` and the eleven `repack_*_to_*_bl`
  block converters it selects between.

About 940 live lines, invisible to the contract. Swapping the cluster in at
that point produced a library that linked, ran, and built a **structurally
different graph** — `node-diff` failed on the first `MUL_MAT`, because with
the buffer type stubbed to null nothing was ever repacked.

**`repack.cpp` is one of the four translation units `CLAUDE.md` names as
exceptions to the unmangled-exports contract. This is what that exception
costs if you forget it.** `scripts/cluster-check` now prints a second line
and generates a throwaway root asserting `repack.dispatch_implemented` at
comptime, so the count cannot stand alone again. Negative-tested both ways.

### The C++ that had to become Zig

Nothing exotic, in the end:

- **The template becomes a comptime-parameterised struct.** `TensorTraits`
  takes the C's four template parameters and resolves `gemv`, `gemm`,
  `quantize_mat` and `repack` through three comptime tables — the C writes
  those as explicit specialisations, one function each.
- **The vtable is explicit**, as `cpu/extra.zig` already set up for
  `traits.cpp`. `tensor_traits_base`'s extra `repack` method becomes a
  second field after the base, which is sound for the same reason the C++
  downcast in `set_tensor` is: the base is at offset zero.
- **The buffer type is a magic static**, because its `.device` member is a
  call. Same acquire-load-plus-pthread-mutex shape as `cpu_backend.zig`.
- **The five `<…, 1, 16>` instantiations are RISC-V.** They and the five
  `*_16_bl` converters and four `make_block_*x16` interleavers beside them
  are inside `#if defined __riscv_zvfh`, `static`, and unreferenced here.
  Not ported; `blocks.zig` still carries their layouts.

Three Zig traps on the way, all of them ones this file already records:
`i1`, `i11`, `i12` and `u12` are reserved integer type names and the C uses
every one as a loop index; `.?` does not narrow a `[*c]`, `impl.one` does;
and `p.*.ne[2]` on a `[*c]` types as the whole `[4]i64`, which the compiler
happened to catch here and does not always.

## `scripts/repack-diff` — the oracle 36 kernels did not have

`node-diff` passed after one fix. That was not the same as the kernels
being right.

### Why they had no gate

The 36 interleaved kernels are unreachable from every gate but one:

- **`test-backend-ops`** never allocates a `CPU_REPACK` buffer, so no op it
  builds can select a `tensor_traits`.
- **`make ops-diff`** builds its tensors in a plain CPU buffer, for the
  same reason.
- **`make port` / `parity-port`** run on Metal.
- **`make node-diff`** does reach them — and reports the *first* divergent
  node and stops. One kernel per run, with a model load and two decodes in
  between.

And half of them are unreachable even at runtime on this target:
`ggml_repack_get_optimal_repack_type` gates the `8x8` and `4x8` shapes
behind AVX2, AVX-512, SVE or `__ARM_FEATURE_MATMUL_INT8`, none of which
`zig cc` selects here. They are exported, compiled, and dead — the same
position as the 25 `_generic` dot products.

### How the reference is reached

The three reference translation units are compiled with **every one of
their 66 unmangled exports renamed to `ref_*` on the command line**, so
both implementations link into one process and can be called back to back
on identical bytes.

That is the same `-D` rename recorded above as unusable for *swapping* a
translation unit, because it renames internal callers too. Here that is
exactly the property wanted: the renamed object is self-consistent and
calls its own kernels, not ours.

The inputs are random bytes with only the `d`/`dmin`/`e` scale fields
pulled back to a finite range. Every other bit pattern is a legal quant and
none can reach a NaN, so leaving them random is what makes the test wide;
a NaN would make a bit comparison meaningless. **The block layouts in
`harness/repack_diff.zig` are written out from `repack.h` rather than
imported from `src/`** — importing them would mean a mistyped bound
produced two wrong buffers that agreed with each other.

### What it found, in one run

`node-diff` named `MUL_MAT ffn_out-0` and stopped. `repack-diff` reported
**12 of 36 kernels differing**, in four distinct faults:

1. **`ggml_gemm_q6_K_8x4_q8_K`: `half * 1024` and `half * 512` where the C
   has `half * 512` and `half * 256`.** The one fault on the live path, and
   the only one `node-diff` could see. Found in one run instead of one run
   per kernel.
2. **`bsums[0] + bsums[1]` added at `i16` width.** The C promotes both to
   `int`; Zig adds two `i16` as `i16`. In range the sums are at most ±2032
   so nothing wraps in practice — but it is illegal behaviour in a safe
   build, and against random input it turned two kernels' output into
   nonsense. Four sites, `q4_k.zig` and `q5_k.zig`.
3. **The contracted epilogue, at twenty sites.** `sumf[j] += sumi *
   GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d` fuses under clang's
   default `-ffp-contract=on`; strict Zig does not. One ULP, in ten
   kernels. `src/ggml/cpu/repack/epilogue.zig` names it.

   **Which** multiply is fused was read off the reference's disassembly,
   not reasoned about — `ref_ggml_gemv_iq4_nl_8x8_q8_0` at `-O2`:
   `scvtf` / `fcvt` / `fmul s2, s2, s3` / `fmadd s2, s2, s1, s4`. The inner
   product rounds; the outer one fuses with the add. Same shape as `xielu`:
   the left operand of the `+`.
4. **`ggml_gemv_q5_K_8x8_q8_K` deferred a bias the C subtracts in place.**
   The C's comment says `// FUSED BIAS: Compute and subtract bias
   immediately`; the port accumulated it in an integer across all four
   sub-blocks and subtracted once at the end. `sb_min` is constant across
   `sb`, so it is the same arithmetic and not the same rounding. Its 8x4
   sibling genuinely does defer, which is why only one of the two was
   wrong.

36 of 36 identical after. Three of those four faults are in code that
cannot execute on this target, which is the point: **they were found
because the gate does not care what the dispatch would have selected.**

### It covers the converters too, through the real buffer type

The kernels are exported and can be called directly. The eleven
`repack_*_to_*_bl` converters are `static`, so the only way to reach the
reference's is the way the model loader does: allocate a tensor in the
`CPU_REPACK` buffer type and write to it. That turned out to be the better
route anyway — it puts `ggml_repack_get_optimal_repack_type`, `init_tensor`
and `set_tensor` under the same comparison, and a type the dispatch does
not select repacks on neither side, so **"which instantiation did you
choose" is itself part of what is diffed**.

Reaching the two buffer types needed one thing from each side:

- **The reference's only exported name for it is mangled.** `repack.h`
  declares `ggml_backend_cpu_repack_buffer_type` outside any `extern "C"`,
  so it is `_Z35ggml_backend_cpu_repack_buffer_typev` and `-D` cannot
  rename it. The harness takes it with `@extern` by that name — the same
  fact that makes the symbol count not the contract, seen from a third
  direction.
- **Ours is reached through the registry**, via the
  `ggml_backend_dev_get_extra_bufts` proc address, which is what
  `llama.cpp` itself calls. That is pure C ABI and puts `cpu_backend.zig`'s
  extra-buffer list under the check too.

44 checks in all: 36 kernels and 8 types.

### Negative-tested 4/4, one per layer

| Injected | Where | Reported as |
|---|---|---|
| a `vdotq_laneq_s32` lane index, 2 → 3 | `arm/q4_0.zig`, a **live** kernel | `gemv_q4_0_4x4_q8_0` |
| a scale nibble mask, `0xF` → `0x7` | `q2_k.zig`, a generic **no other gate can reach** | `gemv_q2_K_8x8_q8_K` |
| the source block a scale group is packed from | `convert.zig`'s `make_block_q4_Kx8` | `q4_K`, 96 of 1152 bytes |
| the instantiation the dispatch selects, `8x4` → `8x8` | `dispatch.zig` | `q6_K`, 1339 of 1680 bytes |

The second is the one that matters for the gate's existence: the same run
printed `q2_K  not selected on this target, both sides`, so nothing that
goes through `ggml_repack_get_optimal_repack_type` could ever have
executed that kernel.

### `make validate`'s formatting list is a gate too

`zig fmt --check` is enumerated by directory in the Makefile, and
`src/ggml/cpu/repack/` and `src/ggml/cpu/repack/arm/` were not in it — 22
new files, unchecked. Added.

### The one fault only the Debug build could see

Every ReleaseFast gate passed the cluster — `repack-diff` 36/36, `ops-diff`
596/596, `node-diff` 2750 nodes × 2 thread counts, `backend-ops` 21,093
configurations, `probe`, `parity-port` 6/6, `parity-port-cpu` 6/6. Then
`make parity-cli` failed **all six prompts with SIGABRT**, before a token
was generated:

```
thread panic: cast causes pointer to be null
  src/ggml/impl.zig:225 in one
  src/ggml/backend.zig:863 in ggml_backend_dev_supports_op
  …
  llama.cpp/src/llama-context.cpp:2440 in graph_reserve
```

`ggml_backend_cpu_device_supports_op` opened with

```zig
const src0 = impl.one(Tensor, op.src[0]);
const src1 = impl.one(Tensor, op.src[1]);
```

where the C has two plain pointer copies that may be null, dereferenced
only by the arms that need them. The scheduler asks `supports_op` about
**leaf** tensors, and a leaf weight has `op == GGML_OP_NONE` and no sources
at all.

**Narrowing a null `[*c]` to `*T` is checked in a safe build and unchecked
in ReleaseFast.** Every bit-exact gate in this project builds
`--release=fast`, on purpose — they compare bits, and the optimization
level moves them. So the whole of that list is blind to this class of
fault, and will stay blind. `make parity-cli` builds the CLI in Debug and
is the only thing that runs the library with safety on against a real
model.

Two things follow:

- **`impl.one` is not a free substitute for a pointer copy.** Use it where
  the C dereferences unconditionally; keep the optional where the C does
  not. `srcOf` in `cpu_backend.zig` is that distinction written down.
- **A gate's optimization level is part of what it can see**, the same way
  `ops-diff` showed its input distribution and thread count are. Six
  ReleaseFast gates agreeing says nothing about safety-checked behaviour.

## `port-links` was silently skipping wrapped citations

Found by accident, while adding one: a citation whose `(…)` wraps across
two comment lines —

```zig
/// Ports `ggml_backend_cpu_repack_buffer_type_get_name` (ggml-cpu/repack.cpp:4745
/// @c1d0e7a00).
```

— was **not checked**. `findCitation` parses a single line, and an
unclosed `(` hits its `orelse continue`: no citation, no error, nothing.
Deliberately pointing one at the wrong line and watching `port-links` say
`PASS` is how it surfaced.

**36 citations across nine files were wrapped that way**, 16 of them
long-standing in `backend_reg.zig`, `gguf.zig`, `backend_sched.zig`,
`ops.zig` and `quants/k.zig`, and 20 of them written this sitting — several
of which I had *created* by reflowing previously-checked one-line
citations to fit 80 columns. That is the worst shape of this bug: tidying
the prose silently removed the check.

Wrapping at 80 columns is normal in this codebase, so the fix joins
continuation comment lines before parsing rather than forbidding the wrap.
`logicalLine` appends following comment lines, stripped of their `///`,
while the citation stays unclosed, up to two of them.

The checked count went **1,703 → 1,779**, and 52 problems appeared that
had always been there:

| Class | Count | Example |
|---|---:|---|
| citation names fewer symbols than numbers | 42 | `unary-ops.cpp:3, 7, 11, …, 96` named no symbol at all; now lists all 22 `op_*` |
| wrong line | 6 | `copy_experts` cited at its last call site, not its definition |
| missing commit | 5 | `ggml.c:5968, 6012, 6060` |
| symbol pairing off by one | 2 | `load_lo`/`load_hi` with a third backticked name after them |

Negative-tested: a wrapped citation with a line number drifted by two now
fails, where before it passed. **A checker that silently skips its input
is indistinguishable from one that passes it.**

## Porting `ggml-metal-common.cpp` — the Metal group's first unit

### The measurement that picked it

Before touching anything, the five Metal C++ translation units, with
"mangled reached from outside" computed the way the original Stage 4
survey did — intersecting each object's mangled exports with every *other*
object's undefined symbols:

| TU | raw | live | unmangled | mangled reached |
|---|---:|---:|---:|---:|
| `ggml-metal.cpp` | 998 | 694 | 6 | 0 |
| `ggml-metal-device.cpp` | 2,267 | 1,694 | 73 | 0 |
| `ggml-metal-common.cpp` | 457 | 299 | 6 | 0 |
| `ggml-metal-ops.cpp` | 5,369 | 4,068 | 66 | 0 |
| `ggml-metal-tuning.cpp` | 1,087 | 1,053 | **0** | **7 of 7** |

`ggml-metal-tuning.cpp` is the last of the four exceptions `PLAN.md`
names, and it is the sharpest form of the `repack.cpp` trap: **zero
unmangled exports means `port-coverage` would read it as 0/0 = 100% with
not a line written.** Its seven are all in `namespace ggml_metal_tuning`
and all reached only by `ggml-metal-ops.cpp`; the pair moves together.

The boundary with the Objective-C that stays is a pure C ABI in both
directions — the `.m` files need 9 symbols from the C++ group and supply
57 to it, none C++-linkage. So Decision 13 costs nothing structurally.

`ggml-metal-common.cpp` was chosen first because **it contains no Metal
API at all**: its own header opens "helper functions for ggml-metal that
are too difficult to implement in Objective-C", and the body is interval
arithmetic plus a reordering pass the C itself notes "is generic and not
specific to metal". It proves the swap mechanism for `ggml-metal/` with
none of the Metal surface.

### The oracle was verified before the port, not after

`make node-diff ARGS=--gpu` hashes every node of a real decode on Metal.
Two runs before porting: 2750 nodes identical at one and four threads,
both times — so Metal node hashes are reproducible run to run **and**
across builds, and the gate is usable as an exact oracle for this group.
That is a far better starting position than `ggml-cpu` had, where nothing
could see the kernels at all.

### Green did not mean running, and the probe says which gate covers it

An unconditional `impl.abort` at the top of `ggml_graph_optimize`:

- `make node-diff ARGS=--gpu` — **dies**. The function is on the path.
- `make node-diff` (CPU) — **passes, 2750 nodes identical.** The CPU
  backend never calls it.

So exactly one gate covers this file, and it is not the default one.

### Fault injection: 3 of 4, and the fourth is provably inert

| Injected | Result |
|---|---|
| `N_FORWARD` 64 → 16 (shorter reorder window) | caught, diverges at node 25 |
| `node_info::dst()` ignoring fused tensors | caught, diverges at node 8 |
| `GGML_OP_MUL_MAT` dropped from `h_safe` | caught, diverges at node 13 |
| overlap test `mr.p1 >= cmp.p0` → `>` | **passed** |

The fourth was checked rather than written up as a hole, per the rule the
`vec.cpp` faults established. Instrumenting the port to abort whenever
`mr.p1 == cmp.p0` holds with the two ranges in the same buffer and not
both sources: **it never fires** on this graph. The two spellings cannot
differ on this input, so the injection changes nothing and says nothing
about the gate.

### Allocation failure has to abort, not degrade

The C's `std::vector::push_back` and `new` throw with nothing to catch
them, so they reach `std::terminate`. The first draft returned `false`
instead — and every caller reads `false` as *no conflict found*, which
would silently produce a **different graph order** and report itself
through `node-diff` as a porting bug. Allocation failures abort.

## `port-coverage` reported 100% for a library with none of the symbols

Found by the linker, one step after the coverage report said the port was
`complete and swapped into the build`.

`src/ggml/ported.zig` and `src/ggml/module.zig` each carried their own
copy of the import list. The library is built from `module.zig`;
`port-coverage` and `zig build test-port` compile `ported.zig`. Adding
`metal/module.zig` to the second and not the first produced:

```
ggml-metal-common.cpp 6 / 6 symbols (100%), complete and swapped into the build
```

with `nm` on `libggml.a` showing **none** of the six. The link failed
afterwards only because `ggml-metal-ops.cpp` happens to call them; a newly
exported symbol that nothing references yet — a kernel reached only
through a dispatch table, which this project has by the dozen — would have
passed both.

`ported.zig`'s own header had said since it was written that it "becomes
redundant" once `ggml.c` is fully ported. It is, and everything else under
`ggml/src/` with it, so the duplicated list is gone: `ported.zig` now
re-exports `module.zig` and the two cannot diverge. Negative-tested —
dropping the metal import from `module.zig` alone now reads
`0 / 6 symbols (0%)`.

## Zig 0.16 miscompiles a small `extern struct` received by value, and it had broken a shipped function

Found while measuring the Metal group, not by any gate.

`ggml-metal.cpp` calls five `ggml_metal_tuning::` functions directly, so
porting it needs those symbols — which are C++-linkage. The question was
whether Zig can supply a mangled C++ name. It can: `@export` with
`.name = "_ZN17ggml_metal_tuning19fa_vec_baseline_cfgEii"` resolves, and
a `fa_vec_cfg_t` **returned** by value comes back correct.

A `fa_vec_cfg_t` **parameter** did not. The C++ caller passed `{42, 17}`
and the Zig callee saw `{0, 0}`. Narrowing it:

- Not the mangling. A plain `export fn zz_echo(cfg: Cfg) Cfg` fails the
  same way.
- **One-directional.** Zig as the *caller* passes small structs to C
  correctly; only Zig as the *callee* is affected.

Ten shapes, measured rather than reasoned about:

| struct | size | result |
|---|---:|---|
| one `u8` | 1 | zeros |
| one `u16` | 2 | zeros |
| `u16` + `u8` + pad | 4 | zeros |
| one `u32` | 4 | **correct** |
| one `f32` | 4 | **correct** |
| 3 × `u16` | 6 | zeros |
| 8 × `u8` | 8 | **correct** |
| 2 × `f32` | 8 | **correct** |
| 3 × `u32` | 12 | **correct** |
| `bool` + pointer | 16 | **correct** |
| one `u16 align(4)` | 4 | zeros |

Alignment is not the rule — 8 bytes at align 1 works and 2 bytes at
align 4 fails. What fits every point is **size a multiple of 4 with no
padding**. The practical rule is simpler: **do not take a small `extern
struct` by value in an exported function.** Declare the parameter as an
integer of the same size and `@bitCast`; that was measured correct.

### It had broken `ggml_bf16_to_fp32` since `ggml.c` was ported

`ggml_bf16_t` is `struct { uint16_t bits; }`. `runtime.zig` had

```zig
export fn ggml_bf16_to_fp32(x: c.ggml_bf16_t) f32 {
    return impl.bf16ToFp32(x.bits);
}
```

and a C caller got **0.0 for every input**. Confirmed against the built
`libggml.a`, not inferred.

**Nine correctness gates missed it**, and the reason is structural rather
than unlucky: `ggml_bf16_to_fp32_row` takes a *pointer* and works, so
every dequantizing path is fine; Zig-internal callers never cross the C
ABI; and nothing in a Qwen3.5 decode calls the scalar form. Only a C
caller can see it.

Auditing the rest of the exported surface found exactly one other
by-value struct parameter — `gguf_init_params`, 16 bytes, `bool` plus a
pointer — and it is **correct**, which is consistent with model loading
having always worked.

### The gate

`scripts/abi-check` + `harness/abi_structs.c`: a C driver calling the
port's by-value-struct and scalar entry points and checking the values
arrive. 11 checks. Negative-tested by restoring the broken signature — 4
of 11 fail.

**Only four, and which four matters.** With the struct broken the callee
reads zeros, so `0x0000 → 0.0` and `0x8000 → -0.0` both still compare
equal and pass. A gate whose inputs are all zero cannot see a bug that
reads zero — the same lesson `ops-diff` taught about input amplitude and
the `nvfp4` goldens taught about all-zero rows, in a third form.

## `PLAN.md` had the Metal tuning coupling wrong

The Stage 4 exception table says of `ggml-metal-tuning.cpp`: "Called only
by `ggml-metal-ops.cpp`. Small; port the pair together."

Measured, by intersecting each object's undefined symbols with the
`_ZN17ggml_metal_tuning` exports:

| object | tuning symbols referenced |
|---|---:|
| `ggml-metal.o` | 5 |
| `ggml-metal-ops.o` | 2 |
| `ggml-metal-device.o` | 1 |

**Three translation units, not one.** Since every remaining Metal C++
file needs it, tuning either goes first or all four go together as 7,509
live lines. Going first is possible because Zig can export the mangled
names — see above — so the three C++ files keep linking unchanged while
each is ported in its own step.

## Porting `ggml-metal-tuning.cpp` — and a gate that was green twice over

1,053 live lines, of which 936 are a generated table. The logic is three
bucket functions, a baseline lookup, and `fa_vec_pick`, which chooses the
`(Q, NE)` a flash-attention vector kernel is instantiated at.

### It had to go first, and Zig can supply the mangled names

Its seven entry points are the only **C++-linkage** symbols left in ggml,
and `PLAN.md` recorded them as "called only by `ggml-metal-ops.cpp`".
Measured: `ggml-metal.o` references five, `ggml-metal-ops.o` two,
`ggml-metal-device.o` one. Three translation units, so every remaining
Metal C++ file needs this one.

That left two options: all four together as 7,509 live lines, or make Zig
provide the C++-linkage symbols. `@export` with the mangled name works —

```zig
@export(&abiPick, .{ .name = "_ZN17ggml_metal_tuning11fa_vec_pickE20ggml_metal_device_idiiiixx" });
```

— and the names were taken from `nm` on the reference object rather than
constructed. `CLAUDE.md`'s note that `ggml-backend-dl.cpp` has "C++ linkage
Zig cannot provide" is still true of *that* file, whose signatures name
`std::filesystem::path`; these seven are POD. So the three C++ callers keep
linking while each is ported in its own step.

One of the seven needed the workaround from the same day's ABI finding:
`fa_vec_set_override` takes `fa_vec_cfg_t` by value, which Zig 0.16 hands a
callee as zeros, so the thunk takes a `u16` and `@bitCast`s.

### The table was transcribed by script

The C's own comment reads "Generated by `ggml-metal-tuning fa-vec`; do not
hand-edit." 936 rows is past the point where hand-copying is honest, so a
regex extracted them, **skipped none**, and the count was checked against
the C's own. `tuning_table.zig` records that.

### The gate passed, and was testing nothing — for two separate reasons

`scripts/tuning-diff` sweeps 3,528,489 lookups: every device id, every KV
type, every head-size pair, both sides of every bucket edge, the family
fallback, and the override. It passed on the first run. Then **all four
fault injections passed too.**

Two independent causes, both of them this project's standing traps:

- **The archive held both definitions.** `ggml-metal-tuning.cpp` was still
  in `build/llamacpp.zig`, so `libggml.a` carried the mangled symbols twice
  — `ggml-metal-tuning.o` and `libggml_zcu.o` — and the linker picked one
  silently. Checked with `nm` per object, which is the only way to see it.
- **The harness was compiled with the reference's `-D` renames.** Those
  renames are what turn the reference's definitions into `ref_*`; applying
  them to the harness renamed its declarations of *our* names too, so every
  comparison called the reference twice. The script now renames the
  reference translation unit alone, and the harness says so at the top.

With both fixed, 5/5 caught:

| Injected | Lookups differing |
|---|---:|
| one table row dropped | 15 |
| `ne11` bucket edge 16384 → 16383 | 1,933 |
| `baseline_ne` drops the 192/128 pair | 44,456 |
| family-9 fallback → generic | 45,375 |
| one `cfg` value changed | 10 |

### `port-coverage` would have read 0/0 = 100%

This file has **no unmangled exports at all**, so the `grep -v '^_Z'` filter
leaves an empty contract and the script would have called an untouched file
complete — the `repack.cpp` trap in its purest form. `report_unit` now takes
`MANGLED_`, an explicit list for the exceptional file, and reports 7/7.
Negative-tested at 6/7 (85%) by withholding one export.

### Which gate sees it

An `impl.abort` in the exported `fa_vec_pick`: `node-diff ARGS=--gpu`
**dies**, `node-diff` on the CPU passes 2750 nodes. Flash-attention kernel
selection is Metal-only, so the `--gpu` run is the only end-to-end gate
that reaches this file — the same split `ggml-metal-common.cpp` has.

## Porting `ggml-metal.cpp` — the backend, device and registry

694 live lines, 6 unmangled exports, pure C ABI. Nothing in it talks to
Metal: every Metal call goes through `device_c.zig` to the two `.m` files
that stay Objective-C.

### Three buffer types and two ifaces collapse to one body each

`diff`, with `shared`/`private` elided, says the two buffer ifaces are
**identical but for the polarity of one assertion** —
`ggml_metal_buffer_is_shared(ctx)` versus its negation — and the three
buffer types differ only in a name suffix (`""`, `_Private`, `_Mapped`)
and whether `alloc_buffer` asks for shared memory. So each is one
comptime-parameterised body, as `repack/dispatch.zig` did for
`tensor_traits`.

`alloc_buffer` is worth reading twice: it *asks* for shared and then picks
the iface from **what it got**, because a shared request can come back
private.

### Three facts checked rather than assumed

- **`GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)`**, the C's last line,
  expands to nothing: `ggml-backend-impl.h:264` defines it empty unless
  `GGML_BACKEND_DL` is set, and this static build does not. That is why
  `ggml_backend_init` is not among the exports.
- **Zig 0.16's `std.c` declares `getenv` but not `setenv`.** The C sets
  `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` as a macOS workaround, so the port
  declares `setenv` as an `extern fn`.
- **`supports_buft` compares `get_name` function pointers**, so the three
  buffer types need distinguishable ones. Folding them would still give
  the C's answer — all three comparisons would match the one address — but
  the test checks the behaviour rather than the addresses, so the question
  does not arise.

### `device_c.zig`, and why the headers are not imported

All three `ggml-metal-*.h` **do** import cleanly — measured; they are
`extern "C"` and include only `ggml.h`, with the Objective-C behind them.
Adding them to `impl.zig`'s `cImport` was tried and worked.

It was reverted. That `cImport` is shared by every ported file, so the
Metal headers would widen the `c` namespace the whole library sees and
oblige six build and script sites to carry a new include path. A second
`cImport` inside `metal/` is not an option: two of them produce two
incompatible `*ggml_tensor`, which is the trap `cli/c.zig` records for
`llama.h`. So `device_c.zig` hand-declares the 39 functions and the one
struct this directory needs.

**That makes `ggml_metal_device_props` a hand transcription, which
`CLAUDE.md` calls "the single largest source of silent error".** It is 19
fields of which the port reads seven; the other twelve exist only to place
those seven at the right offsets, and a field typed `c_int` where the C
has `size_t` would shift every later one and report nothing.

So `make struct-layout` asks the C compiler: `sizeof`, `alignof`, every
`offsetof`, and every field's width, against values the Zig exports. 41
facts. Negative-tested 3/3 —

| Injected | Caught as |
|---|---|
| `max_working_set_size` as `c_int` | `sizeof` 4 vs 8 |
| a dropped `bool` field | field count 18 vs 19, plus four shifted offsets |
| `name: [127]u8` | `sizeof(name)` 127 vs 128, `offsetof(desc)` off by one |

**And it caught a real error on first use.** The struct doc, the harness
comment and a unit test all said "22 fields". It is 19. The field-count
check is what surfaced it — which is the whole reason that check is in
there rather than just the offsets.

### `port-links` caught 32 wrong line numbers in one file

Every citation in `backend.zig` was written from the structure survey
rather than checked against the C, and 32 of them were wrong. That is the
gate doing exactly its job, and a reminder that a citation written from
memory is worth nothing: the whole point is that it is the diff map when
the pin moves. 1,900 citations across 101 files now resolve.

### `node-diff --gpu` sees almost none of `ggml-metal.cpp`

Six injections, and the first reading of them was wrong. Against
`node-diff ARGS=--gpu`:

| Injected | node-diff | Verdict |
|---|---|---|
| abort in `graph_compute` | **dies** | on the path |
| `CUMSUM`/`ARGSORT` alloc size not doubled | passes | **inert** — neither op is in the graph |
| `opBatchSize` reads `ne[1]` for `MUL_MAT_ID` | passes | **inert** — the op is not in the graph |
| one `guid` byte | passes | **inert** — the guid is only compared against itself |
| mapped buffer type asks for private memory | passes | **inert** — a probe shows mapped `alloc_buffer` is never called |
| `get_alignment` 32 → 64, and → 16 | passes | reachable (probed), invisible |
| **FLASH_ATTN_EXT scratch terms dropped** | **passes** | **a real escape** |

The op census settled the inert ones. A Qwen3.5 decode on Metal is:

```
520 VIEW  432 RESHAPE  374 MUL_MAT  242 MUL  158 RMS_NORM  148 GET_ROWS
144 CPY   132 ADD      72 SILU      72 SCALE  72 L2_NORM    48 SWIGLU
48 SIGMOID 36 TRANSPOSE 36 SSM_CONV 36 SOFTPLUS 36 PERMUTE
36 GATED_DELTA_NET 36 CONCAT 24 SET_ROWS 24 ROPE 12 FLASH_ATTN_EXT 12 CONT
```

No `CUMSUM`, `ARGSORT`, `TOP_K` or `MUL_MAT_ID`, so four of the switch's
five arms are never evaluated.

**The FA one is real and was nearly written off with them.** It was
checked rather than assumed: instrumenting the port to abort when any of
the four extras is non-zero made the run die, so the terms matter on this
graph — and dropping them gives 2048 bytes where the reference gives
331776, a 162× under-allocation, which `node-diff` passed. Had the first
summary stood, that would have gone down as "probably inert".

## `scripts/metal-diff` — the gate `ggml-metal.cpp` was missing

Both registries in one process: the reference's six exports renamed with
`-D`, then the buffer-type and device vtables asked the same questions
over tensors built to reach every arm of the switch. 65 answers.

Negative-tested 6/6, including the two `node-diff` could not see:

| Injected | Answers differing |
|---|---|
| FA scratch terms dropped | 4 — 2048 vs 331776, and three more shapes |
| `get_alignment` 32 → 16 | 1 |
| `CUMSUM`/`ARGSORT` not doubled | 2 |
| `TOP_K` arm dropped | 1 |
| device type GPU → CPU | 1 |
| `opBatchSize` reads `ne[1]` for `MUL_MAT_ID` | 4 |

### The last one escaped the first version of this harness too

Its `MUL_MAT_ID` probe was a single tensor with `ne[1] = 2, ne[2] = 16`.
`offload_op` compares the batch size against
`op_offload_min_batch_size`, which `ggml-metal-device.m:1196` defaults to
**32** — and 2 and 16 are both below it, so reading the wrong dimension
gave the same answer. The probes now straddle 32 in both directions,
which took the gate from 44 checks to 65 and closed it.

Third time this project has learned it: **a gate's input distribution is
as much a part of it as its comparison.** `ops-diff` needed amplitude 30,
the `nvfp4` goldens needed a non-zero row, `abi-check` needed a non-zero
bf16, and this needed a batch size on the far side of a threshold.

### `test-port` links no Metal, and that broke `validate`

`metal/backend.zig` calls 39 functions that live in the Objective-C and in
`-device.cpp`/`-ops.cpp`. The `test-port` root links none of them on
purpose — its own comment reads "the ported registry must not reference
the Metal backend" — so importing `metal/module.zig` unconditionally made
`make validate` fail with 39 undefined symbols.

`backend_reg.zig` already had the pattern: guard on `config.use_metal`.
`module.zig`'s import of `metal/module.zig` now does the same. The real
library keeps its Metal exports; only that one test root skips them.

**And the failure was in a sweep I had already called green.** The
`validate` step failed forty lines above the passes that got quoted. Read
the whole output, not the tail.

## Porting `ggml-metal-device.cpp` — and what `node-diff --gpu` actually reaches

`ggml-metal-device.cpp` (2,267 raw, 1,694 live, 73 unmangled exports) is
`src/ggml/metal/library.zig`. It is almost entirely one shape repeated 68
times: build `base`, the MSL function name; build `name`, the cache key —
the same string unless the kernel is specialised by function constants, in
which case the key carries them; look the key up; compile on a miss. A
`Name` helper holds the C's `char[256]` and `getOrCompile` holds the
look-up-then-compile pair, which is most of why 1,694 lines of C became
~1,900 lines of Zig with the doc comments.

Four things were worth the detour.

### The 112 constants are generated and checked, not typed

The kernel names and the `ggml_metal_cv_set_*` calls carry 112
`#define`s from `ggml-metal-impl.h`: 16 `FC_*` bases, 43 `OP_*_NUM`
op numbers, 52 `N_SG_*`/`N_R0_*` kernel-shape values, `SZ_SIMDGROUP`,
and the one computed define `N_MM_NK_TOTAL`.

A wrong one is **the quietest fault available in this file**. `FC_UNARY`
is the base index of a Metal function constant slot, so `FC_UNARY + 1`
instead of `+ 0` writes the op number into the slot the *next* constant
owns: the kernel compiles, runs, and does something else. `N_R0_Q4_K`
sets how many rows a dot-product kernel handles per thread, so a wrong
value reads out of bounds. Nothing in the ported Zig can notice either —
there is no assertion these could violate.

So `src/ggml/metal/impl_c.zig` is **extracted from the header by script**,
and `harness/struct_layout.cpp` includes the real header and compares all
112 by name *and* value, in order, with the count cross-checked so a
constant added upstream shows up as a mismatch rather than as nothing.
`make struct-layout` went from 41 facts to 289.

Negative-tested 3/3: an `FC_` index drifted by one, an `OP_` number
wrong, and a constant deleted — the last caught three ways at once (count,
then the name at each shifted position).

### `ggml_metal_pipeline_with_params` is returned by value 68 times

Every one of these functions returns the 40-byte struct by value. After
the `ggml_bf16_to_fp32` bug — where Zig 0.16 miscompiled a small
`extern struct` *received* by value and a shipped function returned 0.0
for every input — this was checked rather than assumed: layout, field by
field, **and** the return path, via `zz_pwp_roundtrip`. Correct. The risk
is in receiving, not returning.

### `node-diff --gpu` reaches far less of this file than it appears to

This is the part worth remembering. Six faults were injected and all six
read `PASS` on `node-diff --gpu`. None of them was evidence about the
gate.

Unconditional aborts in each site, with `node-diff --gpu` asked whether
the run dies, measured which sites a Qwen3.5-2B Q4_K decode actually
executes:

| site | reached? |
| --- | --- |
| `mul_mv`, float arm (`ne00 >= 32`) | **no** — the weights are Q4_K |
| `mul_mm`, `!has_tensor` arm | **no** — this machine has the tensor instructions |
| `flash_attn_ext`, `bc_mask` differing | **no** |
| `norm` at `n_fuse == 2` | **no** |
| `soft_max` | **no** |
| `count_equal` | **no** |
| `rope`, the `is_neox` arm | **no** |
| `mul_mv`, `r2 != r3` | **no** — both are always 1 here |
| `mul_mv`, `r3 != 1` | **no** |
| `mul_mv`, quantized arm | **yes** (the control, and it aborted) |

**Nine of ten probed sites never execute.** The one that does is the
control, and it is the only reason the nine negatives can be trusted.

So **nine** injected faults read `PASS` and not one of them is evidence
about the gate: the float-`mul_mv` suffix, the `mul_mm` shared-memory
budget, `flash_attn_ext`'s `bc_mask` threshold, `rms_norm` at
`n_fuse == 2`, `soft_max`'s suffix, `count_equal`'s simdgroup loop,
`rope`'s neox arm naming the *norm* kernel, `mul_mv` swapping `r2` and
`r3` between their constant slots, and `mul_mv` dropping `r3` from its
cache key. The last three are worth naming individually because each
*looks* like it must change the answer — a different RoPE layout, a
transposed broadcast, two shapes colliding on one pipeline — and each is
a no-op on this graph. Same lesson as the two `vec.cpp` faults that read
as escapes and were provably no-ops: **confirm the injection changes the
computation, or it is not evidence about the gate.**

The conclusion to carry forward: **`library.zig` is swapped in with every
existing gate green, and the existing gates cover a thin slice of it.**
What would actually gate it is a `metal-diff`-style harness that calls
each `get_pipeline_*` beside the reference C++ and compares the chosen
kernel name and the `pipeline_with_params` fields. That needs the `-D`
rename *and* a capture seam, because the reference builds its names into
stack buffers and hands them straight to the `.m` — there is no return
value to diff. Not attempted here.

Six process faults on the way to that table, each of which produced a
full page of green:

- The first injection run hashed `zig-out/lib/libllamazig.a` to prove the
  artifact changed. That library **does not contain the Metal module at
  all** — the barrel gates it on `config.use_metal` — so the hash was
  identical for all six runs and every gate ran against an unmodified
  library. The gates link `zig-out/lib/libggml.a` from `zig build
  reference`. An artifact-changed check is worth nothing if it watches the
  wrong artifact.
- `zig build test -Dtest-filter="a\|b"` is a literal string, not an
  alternation — already recorded here, and hit again. Worse, the
  workaround of grepping each filter's output for `error|FAIL` missed two
  **crashed** tests, which print neither. Only the unfiltered run shows
  `2 crashed`.
- **`timeout` does not exist on macOS**, and neither does `gtimeout`.
  Wrapping each gate in `timeout 600 …` made every invocation exit `127`,
  so a four-fault battery reported no verdict at all — **including its
  baseline**, which is the tell. Printing the command's return code next
  to its verdict is what caught it.
- A runner written as `{ build && gate; } || echo "NOT APPLIED"` fired the
  `||` branch on *success*, because the shell function's value was its
  last `[ … ]` test rather than the gate's. It printed `NOT APPLIED` under
  a correct build hash and swallowed the verdict. Sequential steps that
  each print, with nothing inferred from exit status, replaced it.
- A script written through a nested heredoc mangled its own embedded
  Python and hung for twelve minutes with no output. Write the script to a
  file and run the file.

**Five of these six were in the measuring apparatus, not the port.** Every
one of them would have supported a confident and wrong write-up, and what
caught each was checking a mechanical fact — the artifact's hash, the
command's return code, whether the site executes — rather than reading the
verdict line.

### Two unit tests crashed the moment the file joined the build

`library.zig`'s cache tests put `@ptrFromInt(0x1000)` in the map as stand-in
pipeline handles. That was fine while the file compiled standalone and
became a segfault the moment it was wired into the barrel and linked
against the real `.m`: `ggml_metal_pipelines_free` frees every value it
holds — as the C does — and `ggml_metal_pipeline_free` dereferences
`pipeline->obj` with no null guard, so a fake pointer and `null` are
equally fatal. The tests now make real handles with the `.m`'s own
`ggml_metal_pipeline_init`, which only `calloc`s a wrapper with `obj =
nil`; `[nil release]` is a no-op, so the real teardown path is still the
one under test.

## The `kargs` structs are generated, and the name check is the one that earns its place

`ggml-metal-ops.cpp` fills **65 `ggml_metal_kargs_*` structs carrying 936
fields** and hands each to a Metal kernel as a block of bytes. The kernel
reads them **by offset**. So a field at the wrong offset feeds the kernel a
different number, and unlike almost every other fault class in this port
there is nothing that could notice: no assertion violated, no pointer
invalidated, no symbol missing.

`src/ggml/metal/kargs.zig` is therefore generated from the header by
`scripts/gen-kargs`, and `make struct-layout` compares every struct's size
and alignment and every field's offset, width and name against
`ggml-metal-impl.h`. **3,069 facts.**

Two things worth keeping:

- **The generator must fail, not warn, on input it cannot read.** Its
  first version matched `type name;` and silently skipped seven array
  fields (`uint64_t o1[8]`, `int32_t nef1[3]` and friends) — printing a
  warning to stderr and reporting "929 fields" where the header has 936.
  Those structs would have come out the wrong size. It now `exit`s on an
  unparsed field. Same class as `port-links` skipping wrapped citations
  and the first `Mirrors …` checker: **a generator that silently drops
  input is indistinguishable from one that worked.**
- **Comparing field *names*, not just offsets, catches a fault nothing
  else does.** Negative-tested 4/4: a dropped field, a widened field and a
  shortened array all move `sizeof` or an offset. But **two adjacent
  same-size fields swapped move nothing** — identical `sizeof`, identical
  `alignof`, identical offsets for every field — and `concat.ne00`/`ne01`
  transposed is exactly the typo a 936-field transcription invites. Only
  the name comparison sees it.

## A generator that writes its own checker agrees with itself

`scripts/gen-kargs` emits both `src/ggml/metal/kargs.zig` and the harness
that checks it, `harness/kargs_layout.cpp`. That is convenient and it hid
a missing struct completely.

`ggml_metal_kargs_glu` is declared `typedef struct{` — **no space before
the brace**. The parser matched the exact string `"typedef struct {"`, so
the whole typedef was invisible: 65 structs parsed where the header
declares 66.

**`make struct-layout` reported `PASS: 3069 kargs layout facts agree`.**
It had to: the harness is generated from the same parse, so both sides
were missing the same struct and agreed with each other perfectly. The
count cross-check that was supposed to prevent exactly this
(`zz_kargs_count()` against the generator's list) compared the generator
to *itself*.

Found only because porting `ggml_metal_op_glu` needed `kargs.glu` and it
was not there — i.e. by a compile error in unrelated work, not by the
gate built to catch it.

Two fixes:

- the opening-brace matcher is now a regex, `typedef\s+struct\s*\{`;
- the generator takes an **independent** count from the header's *closing*
  lines, `^} ggml_metal_kargs_\w+;`, and exits naming the missing struct
  if the two parses disagree. Negative-tested: reverting the matcher now
  prints `the header declares 66 kargs structs but 65 were parsed. missed:
  ['ggml_metal_kargs_glu']` and exits 1.

The rule this adds to the ones already here: **a generated checker must be
cross-checked against the source, never against the generator.** Same
family as `ported.zig` and `module.zig` diverging, and as `port-links`
skipping wrapped citations — a check whose input is filtered by the thing
it is checking cannot see what the filter dropped.

The first negative test of this was also vacuous, for the record: a `sed`
whose backslash escaping did not match the file, so the "reverted"
generator still had the fix and still printed 66. Caught by asserting the
pattern was present before replacing it.

## The generated `kargs` structs catch narrowings the C hides

Two things fall out of generating these rather than hand-writing them,
both of which showed up porting the encoders.

**Zig refuses a struct literal with a field missing.** The generated
structs carry no defaults, so every field must be named. The C's
`ggml_metal_kargs_concat args = { .ne00 = …, … }` silently zero-fills
anything omitted, and a field added upstream would quietly become 0 here;
in Zig it is a compile error.

**The widths are not uniform and the compiler says so.** `kargs_pad` has
`int64_t ne0`, while the C's `GGML_TENSOR_LOCALS(int32_t, ne, op, ne)`
locals feeding it are `int32_t` — so the C widens on the way in, then
truncates back to `int` for `nth` and `nk0`. Both narrowings are implicit
and invisible. Porting it as if the field were 32-bit failed to compile,
which is how the asymmetry was noticed; `encodePad` now writes both
narrowings out.

## Porting `ggml-metal-ops.cpp` — the last Metal unit

4,068 live lines, 66 unmangled exports, zero mangled. It is the op
encoder: per graph node, pick a pipeline (`library.zig`), fill the
kernel's argument block (`kargs.zig`), bind the buffers, dispatch. 53
per-op encoders behind one dispatch switch, plus `fwht`, `snake_fused`,
`flash_attn_ext_use_vec`, the four `flash_attn_ext_extra_*` and the two
`mul_mat_id_extra_*`.

With it, **`ggml_metal_cxx_sources` is empty.** No C or C++ compiles
anywhere under `ggml/src/` any more; the two Obj-C `.m` files stay by
Decision 13.

### The exports were held back on purpose, and that is the `repack.cpp` lesson applied

A symbol count cannot tell a kernel dispatch from an `abort`. So the 66
`ggml_metal_op_*` were written as **private** Zig functions while the
bodies were stubs, and only switched to `pub export fn` once the last
body was written. Had they been exported earlier, `port-coverage` would
have read `66 / 66 symbols (100%), complete and swapped into the build`
for a file in which every op aborted — which is precisely what
`repack.cpp` reported with its dispatch stubbed.

Holding them back also made the linker do the completeness check: the
switch to exports was the final step, and nothing resolved until all 66
existed. `ops.encoders_implemented` is the comptime guard, in the same
shape as `repack.dispatch_implemented`.

### Three things in the C worth knowing before touching this file

- **`GGML_TENSOR_LOCALS` is null-guarded.** `GGML_TENSOR_LOCALS_1`
  (ggml.h:298) expands to `(pointer) ? (pointer)->array[0] : 0`, so an
  absent source contributes **zeros** to the `kargs` — not garbage, not a
  dereference. `soft_max`, `rope` and `flash_attn_ext` all depend on
  this, and writing an assertion there instead would be wrong.
- **Several branches are commented out to a constant.** `extra_pad`'s
  `use_vec` test is `if (false)`, `extra_tmp`'s is `if (true)`, and
  `has_kvpad` is forced `true` in both — all so that scratch is always
  reserved and graphs are not reallocated. Only the live arm is ported,
  each with a note; reproducing the dead arm would be inventing code.
- **`conv_2d_dw` writes `/*.nb02 =*/ nb03`** — the `nb02` field is fed
  the weight tensor's `nb[3]`. Reproduced verbatim. Whether it is an
  upstream slip cannot be settled from this side: the kernel reads the
  field by offset, so if it is one, it is one the kernel was compiled
  against. `node-diff --gpu` would be the arbiter and `CONV_2D_DW` is not
  in a Qwen3.5 graph.

### What the gates say

`node-diff --gpu` is 2750/2750 nodes identical at 1 and 4 threads, on a
graph where every Metal dispatch now goes through this file. That is the
strongest statement available, and it is still narrow in the way
`library.zig` already showed: one model, one prompt, one SKU. The ops a
Qwen3.5 decode never reaches — `conv_*`, `pool_*`, `argsort`, `top_k`,
`cumsum`, `count_equal`, the whole `dsv4_hc` group — are exercised only
by `backend-ops`, which compares with a tolerance.

## libllama is one component — and the sampler is the one piece that swaps first

Before porting anything in `src/`, the measurement that settled ggml's
method was rerun on `zig-out/lib/libllama.a`: every object's defined
symbols against every other object's undefined ones. It gives the
opposite answer. **27** C-ABI symbols cross a translation-unit boundary
and **367** C++-linkage ones do, and 25 objects reference another's vtable
or typeinfo. Strongly connected components over that graph: **22 core
translation units and all 146 model objects in one component**, with
`llama-impl`, `-mmap`, `-arch`, `-hparams`, `-chat`, `-io`, `-memory`,
`-cparams`, `unicode`, `unicode-data` (and `llama-quant` above it) hanging
off it. `llama_model` is a base class with a subclass per architecture,
`llama_model_create` names every subclass's constructor, the models call
back into `llm_graph_context`, and the kv-cache variants implement
`llama_memory_i`. The leaves are no easier: their inbound calls are mangled
and take `std::string` or `std::vector`.

So libllama swaps **once**, at `llama.h` (`PLAN.md`, Decision 42), and
is developed beside the C++ until then.

**One cluster peels off: `llama-sampler.cpp` + `llama-grammar.cpp`.**
Inbound, `llama-context.o` and `llama-graph.o` reach samplers through the C
`llama_sampler_i` vtable, `llama_sampler_chain_get`/`_n`/`llama_sampler_name`,
and two mangled functions with plain-pointer signatures
(`llama_sampler_backend_begin`, `llama_sampler_backend_n_nodes`) that Zig
exports under their `nm` names. `struct llama_sampler_chain`'s layout never
crosses. Grammar's 349 mangled exports have one caller, the sampler.
Outbound, the pair calls `llama_vocab` members whose C wrappers are exact
equivalents. Ported: `src/llama/sampler/` (eight files), `src/llama/grammar/`,
`src/llama/regex.zig`, `src/llama/libcxx.zig`. `port-coverage`: 40 / 40.

### The oracle came first

`harness/sampler_dump.c` drives both files through `llama.h` alone, so one
source links against `llama.cpp.zmake` and against ours, and
`make sampler-diff` diffs the listings. Five sections: every
vocabulary-free sampler over seven input shapes and twelve sizes through
apply, accept, clone, reset and copy; the chain API; grammar, lazy grammar,
DRY and infill on the real vocabulary; backend sampling through a context
on Metal; and the CLI's chain at temperature 0.8 over a real decode. Each
record hashes every candidate's id, logit and probability bits. **40,289
records.** It was green against the stock C++ on both sides before a line
of Zig existed.

`make parity-cli` was never going to be enough: it builds one chain at
`--temp 0` and reads one token a step. Typical, XTC, top-n-sigma, both
mirostats, DRY, adaptive-p, infill, every grammar and the whole
backend-sampling path are invisible to it.

### What the C++ leaves to libc++, reproduced from the SDK headers

Three things the standard leaves to the library decide the sampler's
output, so `libcxx.zig` reproduces libc++ 22.1.6 rather than writing any
correct algorithm:

- **The order `std::sort` and `std::partial_sort` leave ties in.**
  Candidates are sorted by logit; ties are common; a top-k cut keeps
  whichever tie the sort put first.
- **`std::discrete_distribution`**: normalise, keep n-1 partial sums,
  `upper_bound` on one `uniform_real_distribution<double>` draw.
- **`generate_canonical`**: two engine draws for `double`, **one** for
  `float` -- XTC's draw can return exactly 1.0. Reproduced, not fixed.

`make libcxx-diff` runs each against the real template, instantiated from
the same headers, in one process -- 189,684 checks.

### Two bugs the oracle found, and the holes they came through

**Typical's entropy loop is unrolled by four.** The disassembly showed one
`fmsub` for `entropy += -p * logf(p)`, so the port fused it -- and diverged
at 4096 candidates. Read in full, the loop is `and x23, x20, #~3`, then four
`fmul`/`fsub` pairs per iteration, then an `fmsub` tail: the first
`size & ~3` products are rounded before the subtract, only the remainder
is fused. The vectorized-reduction shape `CLAUDE.md` records for `vec.zig`,
here as a plain unroll because `logf` is a call. **Grepping the
disassembly for fused instructions finds the sites; only reading the whole
loop says which iterations they cover.**

**libc++ heapifies arithmetic types differently.** One adjacent swap
remained at 32000 candidates. A replay of the exact score array reproduced
it in isolation, and comparing each of libc++'s internals
(`std::__partition_with_equals_on_right`, `__insertion_sort_incomplete`,
`__partial_sort`, `__introsort`) against the Zig on tie-heavy input put it
in the heap: `__make_heap` takes `__assume_both_children =
is_arithmetic<value_type>`, sifts over the length rounded down to odd, and
finishes with a `__sift_up`. `llama_token_data` takes the general path;
typical sorts **`size_t` indices**, which take this one, and introsort only
falls back to the heap when its depth runs out -- which ties at 32000
candidates do. `libcxx-diff` had tested partial sort on `token_data` only;
it now partial-sorts indices too, and fails on the old code (284 checks).

### Gaps closed by making the gates fail

Every gate here was negative-tested, and four injections escaped first
time. Each escape was a hole in the inputs, not in the comparison:

| Fault | First result | Why | Closed by |
|---|---|---|---|
| collating name `NUL` dropped | `regex-diff` passed | random patterns only used `[.space.]` | sweeping every name in the measured tables against all 256 bytes |
| DRY penalty in `float` | `sampler-diff` passed | 1.75, 1.1 and 3.0 round alike in float and double at the exponents the history reaches, and the penalty's last bit vanished when subtracted from a logit near 1 | a DRY case at base 1.13, multiplier 0.77 (differs at exponent 3), applied to all-zero logits as well |
| infill thresholds | passed | no candidate sat between 0.2 and 0.25 or near one-third | variants with three text candidates near 0.45, 0.32, 0.23 |
| partial UTF-8 floors | passed | no grammar ranged near U+0800 or U+10000 | `[ࠀ-ࡀ]` and `[\U00010000-\U00010040]` grammars |

Three injections were **inert**, and are recorded as such rather than as
escapes: mirostat's `mu` update unfused (an ulp in `mu` only matters when
`powf(2, mu)` carries `k` across an integer), `dist`'s `>=` as `>` (needs
`sum_run == sum_tgt` exactly), and top-k one short inside the CLI chain
(top-p and min-p already cut below 40). Caught: dist summing in float,
top-p's comparison, unfused penalties, XTC's draw, a lazy trigger ignoring
its capture group, `chain_get(-1)`, the backend's uniform draw and min-p
threshold, a doubled DRY penalty, a char range's upper bound -- and, for
`regex-diff`, `.` rejecting NUL, loop captures not reset, lazy quantifiers
made greedy, `\b` on an empty remainder, an unescaped `]` accepted, the
complexity factor doubled, `search` skipping the end position, and `[[:w:]]`
without `alnum`.

### `std::regex` is ported, quirks included

A lazy grammar's trigger patterns are `std::regex`, and where the match
starts and which group is first non-empty decide the text the grammar is
fed. `regex.zig` ports libc++'s node graph -- the same `__end_` splicing,
the same backtracking driver, the same `4096 * length` complexity limit.
The parts that live in the dylib were measured with a probe: `.` rejects
exactly `\n` and `\r`; class names are case-insensitive and `[[:w:]]` is
`alnum` plus `_`; 59 collating names resolve, case-sensitively, and no
`SOH`-style control names do. `make regex-diff` runs the real trigger
patterns upstream's chat code builds plus 30,000 random patterns,
malformed ones included, and sweeps the tables: **538,416 checks.**

### Smaller things worth knowing

- **The 1-byte struct `llama_sampler_chain_params` does hit the Zig 0.16
  ABI miscompile.** Declared naturally, `no_perf = true` arrived as zero
  and the chain timed every call; `abi-check` caught it. The port takes a
  `u8` and bit-casts.
- **The `[*c]` array-field trap surfaced as a compile error** this time:
  `tensor.*.ne[0]` through a `[*c]ggml_tensor` is typed as the whole
  array. Every such site is `tensor[0].ne[0]`.
- **`libllama.a` holds two members named `llama.o`** -- the Zig module and
  `src/llama.cpp`. The linker does not care; `ar x` extracts only one.
  `zig build-obj src/llama/module.zig` is the way to disassemble the port.
- **Every throw that escapes through the C ABI terminates**, and the port
  terminates at the same points with the same text: an empty grammar stack
  after accepting a token, an invalid trigger regex, infill's negative
  `resize` on a piece over 512 bytes.

### What the gates say

`sampler-diff` 40,289 records identical; `libcxx-diff` 189,684;
`regex-diff` 538,416; `abi-check` 15/15; `parity-cli` 6/6 at `--temp 0`
with the CLI in Debug, so with Zig's safety checks on; at temperature 0.8,
`raw_completion` linked against both sides gave 12/12 identical 64-token
generations over four seeds before the `generate` section made that
permanent. `validate` runs the three oracles that need no model.

## `unicode.cpp` — the vocabulary's first piece, gated before anything can see it

`llama-vocab.cpp` is the only caller of `unicode.cpp`, and every function it
calls is C++-linkage, so `src/llama/unicode.zig` exports nothing and swaps
with the vocabulary. Until then no downstream gate can reach it. That is the
reason the oracle was built first and in-process: `scripts/unicode-diff`
compiles the unmodified `unicode.cpp` and `unicode-data.cpp` next to
`harness/unicode_ref.cpp`, a C wrapper that turns each result or exception
into plain data, and links them with the Zig port.

### The tables are generated, and the generator is not the checker

`unicode-data.cpp` is 7,034 lines of numbers that upstream itself generates.
`scripts/gen-unicode-data` reads them out of the C++ into
`src/llama/unicode_data.zig`, citations included. What checks the result is
not the generator but `unicode-diff`'s sweep of all 0x110000 code points
against the C++'s own lookups — the lesson of the `kargs` generator, which
agreed with itself.

### `unicode_regex_split` needed `std::wregex` and `regex_iterator`

A pattern set with no `\p{..}` runs as a `std::wregex` over code points; one
with a category runs as a `std::regex` over a "collapsed" byte per code
point. Both go through `regex_iterator` under `optimize | nosubs`. So
`regex.zig` became `BasicRegex(CharT)`, gained `nosubs`, a ranged search and
`Iterator`, and `regex-diff` grew to match — 538,416 checks to 43,537,552.
What the wide engine does differently was measured over every code point
rather than read: `.` rejects U+2028 and U+2029 too, a wide name with a
character at or above 127 is no name, and `__regex_traits_value` narrows
before it reads a digit, so U+0130–U+0133 are not the digits their low
bytes spell. The iterator's semantics are two flags: `__no_update_pos`
turns `at_first` off after the first match, and `match_prev_avail` lets
`\b` read the character before the range. `match_not_null` reaches into a
lookahead through the state's flags, not by any code of its own.

### Quirks carried over

- **A custom splitter that returns nothing falls through to `std::regex`.**
  That happens for every splitter on empty text, and the fallback gives no
  pieces where the splitter's offsets would have given one empty piece.
  Removing the fall-through failed 504 checks.
- **AFMoE's digit groups absorb the text before them.** The first
  `_add_token` of a digit run ends at the run's first group, so everything
  since the last token joins it: `ab1234` is `ab1`, `234`.
- **The llama3, qwen2 and qwen3.5 splitters test `flags.as_uint()` where
  gpt2 tests `flags2`.** Worked through, the two differ only for a space at
  the end of a span, and both then emit the same one-space token — kept
  verbatim all the same.
- **Kimi K2's leading-character backtrack is dead code.** `has_leading_char`
  is set only when the next code point is a non-Han letter, which is the
  loop condition that sets `has_letters`; `pos--` cannot run. The injection
  that deleted it was inert, and is recorded as such, not as an escape.
- **`unicode_cpt_to_utf8` throws above U+10FFFF, and a four-byte lead
  decodes up to U+1FFFFF**, so `\xf4\x90\x80\x80` in a prompt ends the
  split with `std::invalid_argument`. The port returns
  `error.InvalidCodepoint` at the same point; what the vocabulary does with
  it is for the vocabulary port.
- **The byte map's inverse is a `std::unordered_map` keyed by string.** Its
  keys are exactly the 256 canonical encodings, so the port decodes,
  checks the round trip and indexes; a non-canonical `\xC1\xA1` would decode
  to `a` and is refused, as `map.at` refuses it — an injection that dropped
  the round-trip check failed 94 checks.

### What the gates say

`unicode-diff` 7,070,454 checks identical, ~20 s: 5.57M over code points,
1.31M decodes, 66,052 byte-map lookups, 127,100 splits over 33 extracted
sets and 16 synthetic ones. Negative-tested 13/13 effective, one inert.
`regex-diff` 43,537,552, negative-tested 6/6 on the additions. `port-links`
2,598 citations in 119 files.

## `llama-vocab.cpp` — the vocabulary, ported and gated before the swap

`src/llama/vocab/` is all of `llama-vocab.cpp`: `load`, every accessor, the
special-token partition, `tokenize`, `token_to_piece`, `detokenize` and
`print_info`, with one file per tokenizer — SPM, BPE (with the `hybriddna`
and `whitespace` sessions), WPM, UGM, RWKV and PLaMo-2. It reads metadata
through `src/llama/gguf_meta.zig`, a port of `GGUFMeta::GKV` and
`llama_model_loader::get_key` — exact type checks, the error texts, and
`--override-kv` with its warning for a mistyped override — that the model
loader's port will reuse. Nothing is exported: libllama swaps once, so the
port is reached only through `scripts/vocab-diff`, which compiles it through
`src/llama/staged.zig` (now `harness_root.zig`) beside the C++ in
`libllama.a`.

### What the standard library decides

- **`token_to_id`'s iteration order.** Four loops in `load` walk the
  `std::unordered_map`, and the first token whose text matches becomes EOT
  (or a FIM token) and has `CONTROL` forced on it. Qwen vocabularies have
  both `<|im_end|>` and `<|endoftext|>`, so the answer is libc++'s hash-table
  layout. `libcxx.StringMap` reproduces it — CityHash64 as Apple's dylib
  computes it, prime bucket growth, head-of-group insertion and
  `__do_rehash` — and `libcxx-diff` compares order and bucket count with the
  real template. An injection that took the last match instead of the first
  failed 13 checks across five real vocabularies.
- **The merge queues' tie order.** SPM and BPE both use
  `std::priority_queue`. Flipping either comparator's `left` tie-break
  failed — SPM's on six files, BPE's 1,504 checks — so the heap is libc++'s
  `push_heap`/`pop_heap`, from `libcxx.zig`.
- **`cache_special_tokens`'s sort order**, which decides which of two
  overlapping special tokens wins. See below: it took a purpose-built case
  to see it.

### C++ semantics that are easy to port wrongly

- **`char` is signed.** RWKV's hex unescape tests `c >= 'a'`, so a byte above
  0x7f takes the digit branch; `isspace` on a byte above 0x7f is false; the
  `std::tolower` of the model name leaves those bytes alone.
- **A GGUF string is read through `const char *`**, so a token text stops at
  its first NUL whatever its stored length.
- **`std::string::operator[]` returns NUL at `size()`**, which UGM's trie walk
  reads one step past the end.
- **UGM's backtrack assigns through a reference** to the last slot each step.
  It never reads that slot back, so a value copy is equivalent — checked, not
  assumed.
- **`naive_trie::get_longest_prefix` is the longest *path*, not the longest
  stored value.**
- **A `runtime_error` inside `load` is caught** by `llama_model_load` and
  reported as `error loading model vocabulary: ...`; the same exception from a
  tokenizer escapes `llama_tokenize` and terminates. `impl.Exception` carries
  the type and message to whichever boundary decides.
- **`bpe_ranks.emplace` keeps the first duplicate merge; `token_to_id[w] = i`
  keeps the last duplicate token.**

### The gate, and the holes the injections found in it

`vocab-diff` compares, per file: what loading logged, every scalar, every
token, every piece, and tokenize/detokenize over upstream's vocab test
strings and random text with special tokens spliced in, under every flag
combination. 19 upstream files and the Qwen3.5 model reach SPM, BPE, WPM and
UGM, the last with a real precompiled charsmap. RWKV, PLaMo-2, `hybriddna`,
`whitespace` and `none` have no real files, so they are written with the
ported gguf writer from upstream's.

The first fault-injection round caught 15 of 22. **Six escaped, and each
named an input the gate had never produced:**

| Fault | Why it escaped | What closed it |
|---|---|---|
| unstable → stable special-token sort | libc++ insertion-sorts below 30 elements, which is stable | 64 overlapping equal-length specials, and texts where they meet |
| `lstrip` leaving one space | the only lstrip token was UGM's, which drops a lone space anyway | a BPE vocabulary with jina's `<mask>` lstrip |
| UGM unknown penalty 10 → 9 | no real input lets the unknown path compete | a T5 vocabulary whose scores put `a`+UNK 0.5 below `ac` |
| PLaMo-2 `round` → `floor` | no two segmentations tied | scores that tie only after rounding |
| `hybriddna` padding `A` → `C` | no padded k-mer was in the vocabulary | `GTAAAA` in it, and texts ending a span in `GT` |
| default special id `>=` → `>` | no vocabulary had exactly 11 tokens | one that does |

All six fail now, each on its own case. A seventh, WPM's `max_token_len +
1` dropped to `+ 0`, is **provably inert**: no token is longer than
`max_token_len`, so the extra length never matches. It was replaced by a
WPM fault that is not, which fails 1,920 checks.

### What the gates say

`vocab-diff` 55,016,050 checks identical over 41 files, ~15 s,
negative-tested 23/23 effective and one inert; `libcxx-diff` 452,904,
4/4 on the hash table; `unicode-diff` 7,070,454; `regex-diff` 43,537,552;
`port-links` 2,670 citations in 128 files; `validate` OK in 6 m 14 s.

---

## Zig 0.17: the reference, libc++, and the CPU pin

The port moved to Zig 0.17 in v0.0.14 and `make validate` passed. That was
not the whole of it: three things around the port had not moved with it,
and two of them would have made a bit-exact gate wrong rather than red.

### `llama.cpp.zmake` did not build

Every gate that compares on bits -- `ops-diff`, `node-diff`, `parity-port`,
`sampler-diff` -- links `llama.cpp.zmake`'s libraries, and its `build.zig`
failed under 0.17 at the first API it touched (`b.build_root`). The
libraries in its `zig-out/` were 0.16 builds from ten days earlier, so
running those gates would have compared a 0.17 port against a 0.16
reference, and a compiler moves bits.

Decision 18 keeps that repository read-only from here; migrating it was
granted explicitly for this change. What it took: `b.root` for
`b.build_root`, `cwdRelativePath`, `std.lang.Optimize`, `addPassthruArgs`,
`std.mem.find`, and two keys the `zig libc` file now requires --
`darwin_sdk_dir` and a non-empty `crt_dir`, given the values Zig's own
native detection writes. `appendGlob` also declares its directory with
`dependOnDirectoryContents`, because 0.17 reuses a configured graph until a
declared input changes. macOS, iOS and Linux all build.

### `native` stopped meaning the baseline

Zig 0.16 detected this M5 Max as the macOS baseline, `apple_m1`. 0.17 knows
it as `apple_m5`, with i8mm. Our `build.zig` had already pinned its default
CPU for that reason; nothing else had:

- **The zmake reference built native compiled 158 i8mm instructions into
  `libggml.a`**, against 0 in ours -- `__ARM_FEATURE_MATMUL_INT8` selects
  different `quants.c` and `repack.cpp` kernels. Its `build.zig` now
  defaults to the baseline as ours does.
- **Every `zig cc`, `zig c++`, `zig build-exe` and `zig build-obj` in
  `scripts/` compiled native**, 40 invocations. `repack-diff` compiles the
  reference `repack.cpp` itself, and the harnesses compile the port's Zig,
  where `features.zig` reads `builtin.cpu` -- so the reference and the
  port each picked kernels by a CPU neither library was built for. All 40
  now pass `-mcpu=baseline`. `translate-c` does not: none of the headers it
  reads mention an ARM feature macro.

### libc++

`make -C testcase cxx20` passes under 0.17 and `zig c++ hello.cpp` links,
so the workaround came out: `link_libcpp = true` at all seven sites,
`appleCxxFlags` and `linkAppleLibcxx` gone, `-lc++` back in 22 scripts,
`testcase/` deleted. `llama.cpp.zmake` likewise. The CLI now links libc++
statically, with no `libc++.1.dylib` in `otool -L`.

That is not a free switch for this project, because `libcxx.zig` and
`regex.zig` reproduce the libc++ the reference links. Measured before
switching: Zig's is 22.1.8 against the SDK's 22.1.6, `_LIBCPP_ABI_VERSION`
1 on both, no `_LIBCPP_ABI_FIX_CITYHASH_IMPLEMENTATION` on either, the
same regex constants and hardening mode; and every header the ports
reproduce -- `<regex>`, `<random>`, `__hash_table`, `hash.h`, `sort`,
`partial_sort`, `make_heap`, `sift_down`, the distribution engines -- is
byte-identical. The compiled half, `<regex>`'s tables and the bucket
primes, the gates check: `regex-diff` and `libcxx-diff` pass unchanged.

The Apple-clang CMake build (`make ref`, `make parity`) still links
Apple's libc++. It is the only reference that does now.

### What the gates say

All against the rebuilt zmake reference, both libraries baseline, both
runtimes Zig's libc++:

| gate | result |
|---|---|
| `abi-check` | 15 entry points |
| `struct-layout` | 3,103 + 298 facts |
| `metal-diff` | 65 answers |
| `tuning-diff` | 3,528,489 lookups |
| `repack-diff` | 44 checks |
| `libcxx-diff` | 452,904 |
| `regex-diff` | 43,537,552 |
| `unicode-diff` | 7,070,454 |
| `vocab-diff` | 55,016,050 |
| `sampler-diff` | 40,289 records |
| `ops-diff` | 596 op results |
| `node-diff` | 2,750 nodes, 1 and 4 threads |
| `node-diff --gpu` | 2,750 nodes, 1 and 4 threads |
| `parity-port`, `parity-port-cpu`, `parity-cli` | 6/6 each |
| `probe` | ported code on the execution path |
| `backend-ops` | 21,093 op configurations |
| `validate` | OK, 404 s |

**One false red, and it was ours.** The first `validate` failed in
`regex-diff` with `syntax error near unexpected token '}'` at line 44 --
after printing its PASS. bash reads a script as it runs it, and the
`-mcpu` edit landed in that file mid-run. The same run's `probe` and
`backend-ops` failed on a compile error in a file being written beside
them: Zig reports AstGen errors for every file an import list names, test
blocks included, so a staged file that nothing reaches still breaks the
library build. **Do not edit `src/` or `scripts/` while gates run.**

The "Zig 0.16 miscompiles" notes in `CLAUDE.md` -- `[*c]` array fields,
small structs received by value -- were probed under 0.17 with minimal
reproductions: a C caller passing `struct { uint16_t }`, a 6-byte and a
1-byte struct to Zig exports, and `p.*.x[i]` through a `[*c]` pointer,
Debug and ReleaseFast. All four came back right. **That is weaker than it
reads**: no 0.16 is installed to show those reproductions failed before,
so they may simply not be the shape that broke. The workarounds stay, and
`abi-check` still guards the entry points.


---

## The loader and the hyperparameters

Everything `llama_model_load` does before `load_tensors`, ported and gated
in-process: `llama-arch.cpp`, `llama-hparams.cpp`, and the metadata halves of
`llama-model-loader.cpp` and `llama-model.cpp`, with Qwen3.5's
`load_arch_hparams`.

### Four generators, because most of it is tables

`llama-arch.cpp` is four `std::map` initializers -- 148 architecture names,
223 key formats, 247 tensor names, 242 tensor infos -- and a page of
functions. `llama_hparams` is 156 members. `llama-model.cpp` carries the
130-entry `llm_type` name switch, a 148-architecture RoPE switch and the
architecture-to-class mapping. Typing those is the transcription error
`CLAUDE.md` warns about, so `scripts/gen-arch`, `gen-hparams` and
`gen-model-tables` read them from the source, and each is cross-checked
against an independent count of what it parsed. That cross-check paid for
itself immediately: `LLM_TYPE_8x7B` and four siblings have a lowercase `x`,
the enumerator regex skipped them, and the generator said 125 types where the
header has 130.

Two things the generators preserve rather than smooth over:

- **A map initializer keeps the first of a repeated key.**
  `LLM_TENSOR_SSM_NORM` appears twice in `LLM_TENSOR_NAMES`. The strings
  agree today; the first is the one `std::map` keeps.
- **Four RoPE cases read the hyperparameters** (`dflash`, `glm4`,
  `glm4_moe`, `hunyuan_vl`). They are named in the generator and ported by
  hand; a fifth upstream would stop the generator rather than become a
  silent fixed rule.

### `Hparams` has upstream's layout, and that is what makes the gate exact

`llama_hparams` is trivially copyable (`static_assert`) and the model
value-initializes it, so `Hparams` is an `extern struct` with the same
members in the same order, defaults from the member initializers and zero
elsewhere. `loader-diff` first checks every member's offset and size, and
the struct's size and alignment, against `offsetof`/`sizeof` in the C++ --
then reads the reference model's `hparams` **through `Hparams`** and compares
all 156 members bytewise. Negative-tested: swapping two members fails the
layout check before any value is compared.

### What the gate found

- **`split.tensors.count` is an `int`.** The loader reads it into
  `int n_tensors`, so the getter wants `INT32`; the port read `u32` and
  threw "wrong type" on every split set. First run, first failure.
- **`a % 0` is `a` on this machine.** Qwen3.5 marks recurrent layers with
  `(i + 1) % full_attention_interval`; an interval of 0 is undefined in
  C++. Compiled, it is `udiv` then `msub`, and `udiv` by zero gives zero, so
  the remainder is the dividend. The port does that explicitly
  (`remU32`), and the injection that made it 0 instead failed on
  `is_recr_impl` -- which is the reference binary confirming the model.
- **An empty log is a prefix of every log.** An architecture without a
  ported `load_arch_hparams` is compared only as far as the port gets, by
  requiring its log to be a prefix of the C++'s. The first version would
  have passed a port that logged nothing. Loaded and unported cases now
  must log on both sides; a constructor that throws first legitimately logs
  nothing on either, and its message carries the comparison.

### Negative tests

`arch-diff`, 9/9: two architecture names swapped, the suffix's dot dropped,
block and expert indices swapped, `qwen35` removed from the hybrid list, a
tensor info dropped, a missing tensor name returning instead of aborting,
`from_string` never matching, and a short `Kv` enum -- caught at compile time
by the generated length check, and at run time, with the table shortened to
match, by the past-the-end probe.

`loader-diff`, 14/14: weights in plain name order (the type-tie file),
newlines left unescaped in the metadata dump, truncation at 41 bytes,
`a % 0 == 0`, pooling defaulting to `NONE`, the RoPE scale not inverted,
`print_f` printing unsigned, boolean arrays not normalized to 0/1, the 0.8B
and 2B size classes swapped, the split index unchecked, `n_ctx_orig_yarn`'s
default lost, the tensor bounds check dropped, two `Hparams` members
swapped, and integer overrides not narrowed. One injection first failed to
*compile* -- an unused parameter -- and was rewritten until it did, because a
build failure says nothing about the gate.

### What it does not see

Only Qwen3.5 has an architecture-specific step. Every other architecture
is compared vocabulary-only, and in full only as far as the port's log is a
prefix of the C++'s; its base hyperparameters are read but not compared.
`gguf_kv` is compared in `std::unordered_map` order through
`libcxx.StringMap`, but nothing in the port reads it yet.

## i8mm for the M3: the two-row dot products and the gemms

The xcframework's macOS slice moved from the baseline (M1) to `apple_m3`.
That is not a tuning flag. `__ARM_FEATURE_MATMUL_INT8` changes which
kernels exist: `type_traits_cpu` asks the q4_0, q4_1, q8_0, q4_K and q6_K
dot products for two rows per call, `ggml_repack_get_optimal_repack_type`
picks the 4x8 and 8x8 shapes, five gemms compile a NEON body instead of
calling their `_generic`, and llamafile's sgemm is compiled out
(`ggml-cpu.c:45`).

### What building for apple_m3 did before anything was ported

`make ops-diff GATE_CPU=apple_m3` with the baseline port: every second
output row of every two-column `mul_mat` of those five types was
`0x00000000`. `traits.zig` had already asked for two rows, and each kernel
computed one: the `nrc == 1` assertion that would have said so is gone in
ReleaseFast. Nothing else differed, including f16/f32 `mul_mat` without
llamafile. ops-diff saw it only because 2 columns had just been added to its
sweep; 1 and 7 are both odd, and `mul_mat` falls back to one row at an odd
count.

### What was ported

- `quants/arm/mmla.zig`: the five `if (nrc == 2)` arms, called from the
  existing exports behind `comptime mmla.enabled`.
- `repack/arm/mmla.zig`: the five i8mm gemm arms, chosen in `fallback.zig`.
  The `q4_0` 4x8 one is hand-written aarch64, converted by script; the
  script was checked by converting the 4x4 kernel and diffing it against
  `q4_0_gemm.zig`, identical over 445 lines. `q4_K` and `q5_K` share one
  body, which their C says differs only in the weight decode.
- `neon.mmlaq_s32` calls `llvm.aarch64.neon.smmla.v4i32.v16i8` by name, one
  instruction; its portable form is kept for baseline builds and a test
  holds the two together over 100,000 inputs. The portable form was first
  written as nested `inline for` and exhausted comptime's branch quota in a
  Debug test build, sixty-four calls into one unrolled gemm; it is
  shuffles and reductions now.

### The gates, at the CPU that ships

`GATE_CPU` (root Makefile, `scripts/gate-cpu`) builds both libraries for a
CPU and checks that they were: each side reports `ggml_cpu_has_*`, compared
with what `zig cc -mcpu=$GATE_CPU -dM` says the C will see. That check
caught a real case on its first day: a concurrent `make -C metrics libs`
rebuilt the zmake reference at the baseline between two node-diff runs.

At apple_m3: ops-diff 650 results, repack-diff 60 checks, node-diff 2750
nodes on Qwen3.5 Q4_K_M, UD-Q4_K_XL (q4_K, q5_K, q6_K, q8_0 weights) and
IQ4_XS at 1 and 4 threads -- all identical to the stock C on bits.

repack-diff gained a 20-row gemm. The asm gemms do sixteen rows a pass and
finish in a four-row tail; every gemm had only ever run at 4 rows, so the
main loop of both `q4_0` asm kernels had no gate at all.

Negative-tested 10/10: dropped `q4_0` bias, `q4_1` mins added before the
reorder, swapped `q8_0` store columns, an unfused `q4_K` mins subtract and a
`q6_K` bias of 31 (ops-diff); a shift in the `q4_0` asm main loop (caught
only at 20 rows), a wrong `q5_K` fifth-bit position, unshifted `q6_K` high
bits, an unfused `q8_0` accumulate (repack-diff); a row swap in the K-quant
reorder (node-diff on UD-Q4_K_XL, which proves node-diff reaches the i8mm
gemms).

### What is still true

The M2 has i8mm too: Zig's `apple_m3` adds only `hcx` over `apple_m2`, a
hypervisor feature. The slice is documented as M3 and later because that was
the requirement; it has not been run on an M2.

`dotq_s32`'s portable form is 31 instructions where LLVM's `sdot` intrinsic
is one, measured. Every dotprod kernel pays that on the CPU path, at both
CPUs. Not changed here.

## libllama swapped: the rest of `src/`, and the single swap at `llama.h`

Everything libllama does at inference time is Zig now: the batch splitter,
the KV cache, the recurrent and hybrid memories, the graph builder, Qwen3.5's
graph and its delta-net blocks, `llama_context` with its state API, the
adapters, the quantizer, the model saver, training, and the whole of
`llama.h`'s C API. `libllama.a` is one Zig object; `build/llamacpp.zig`
compiles no C or C++ into it. On the ggml side `ggml-backend-meta.cpp` and
`ggml-opt.cpp` are ported and `ggml.cpp` is dropped, so `ggml_base_cxx_sources`
is empty. The CLI links no libc++ (`otool -L` lists frameworks, `libSystem`
and `libobjc`, nothing else).

### How the swap is wired

- **`src/llama/api/`** holds every `llama.h` function as a `pub fn … callconv(.c)`
  with the exact translated signature and **no export**. The in-process
  harnesses call them beside the C++ that defines the same names.
- **`src/llama/exports.zig`** exports every `llama_*` function in `api/` by
  name, at comptime, and is imported only by `module.zig`. Six `api/`
  functions come from `llama-ext.h`, which has C++ linkage; the reference
  exports none of them unmangled, so neither does the port. The result is
  checked with `nm`: the port's `llama_*` text symbols equal the reference's,
  **240 = 240**.
- **The logger is Zig's own, and the harnesses set it.** `impl.zig`'s
  `logInfo` and friends call `log.zig`'s `logInternal`, which formats with
  `snprintf` and calls the callback `llama_log_set` installed, as
  `llama_log_internal` does. A harness links the C++ beside the port, so it
  installs its capture callback on both: `llama_log_set` for the C++ and
  `api.impl.llama_log_set` for the port. (Until the rebase onto v0.0.18 the
  port called the C++'s mangled `llama_log_internal` and `exports.zig`
  exported a Zig variadic under that name; see below for why that went.)
- **`libllama-ref.a`** is upstream's libllama, every translation unit
  including the sampler and grammar, compiled unmodified. The in-process
  harnesses link it: `loader`, `vocab`, `arch`, `batch`, `chat`, `api`,
  `context`, `adapter`, `quant`, `opt-llama`. The end-to-end gates
  (`parity-cli`, `parity-port`, `node-diff`, `sampler-diff`) and `abi-check`
  link the port.

### `context-diff`: the gate for everything a decode reaches

`harness/context_diff.zig` loads one model twice in one process, through the
C++ and through the port, and runs one decode script per case through
`llama.h` and through `api/` alike. The script covers:

- a prompt, then greedy steps;
- an all-outputs batch;
- interleaved sequences whose outputs come back out of order;
- the memory API;
- every state path: buffer, file, per-sequence, on-device, truncated, wrong
  arch, wrong magic, restore-then-decode;
- the error returns;
- a second round after a clear, so graph reuse is exercised.

Both transcripts must be identical: every log line, every graph node
(`cb_eval`: name, op, type, shape, strides, `op_params`, sources, view,
flags, data hash), every return value, and every output row's bits.

It runs 21 context configurations over 9 model configurations:

- **gpu**, **cpu**, **partial** (10 layers offloaded).
- **proxy** — `harness/proxy_dev.c`, a CPU device that declines
  `GGML_OP_GATED_DELTA_NET`.
- **norm-on-cpu** — a buffer-type override.
- **tensor** and **tensor-uneven** — `LLAMA_SPLIT_MODE_TENSOR` over Metal
  and the CPU.

Each model configuration also compares the two load logs. The tensor
configurations additionally compare `llama_meta_device_get_split_state`
against the port's for every weight and cache tensor at 2, 3 and 4 devices.
Before the swap: **2,417,100 checks, 0 failures.**

**Its first runs were not evidence, four times over.** Each escape was a
hole in the script, not the port:

- **The output reorder escaped.** The interleaved batch named sequence 2
  with `n_seq_max = 2`, so both sides rejected it with `-1` and the
  multi-sequence path never decoded.
- **Halving the delta-net chunk size escaped.** Every device here supports
  the fused Gated Delta Net, so `resolve_fused_ops` always enables it and
  `buildDeltaNetChunking`/`Autoregressive` never run. A backend without the
  op takes them. The proxy device is that backend; with it, 23,027 nodes
  per case instead of 13,750.
- **Dropping `graph_get_cb`'s pinning of a layer's norm escaped.** The pin
  is redundant whenever the norm weight already sits on the layer's device,
  which it always does. `norm-on-cpu` moves the norm weights to host memory,
  and the injection then fails at line 105.
- **An on-device sequence restore that copied the wrong way escaped.**
  `seq_rm` does not clear tensor data, so a restore that copies nothing
  restores the right state. The script now overwrites the sequence with
  another decode between save and restore.

After those: **16/16 injections caught**. They were kq_scale, output reorder,
reuse count, recurrent `seq_cp`, input reuse, causal mask, chunk size,
autoregressive decay, norm pin, callback naming, and six in the state code.
Two more went into `meta_split.zig`: rotation ignored, and Q granularity not
doubled. Both caught.

### Two real bugs it found

- **`output_resolve_row`'s negative index.** `n_outputs + i` is `uint32_t`
  arithmetic in the C++, so `-2` with one output wraps to `4294967295`.
  The "negative index out of range" branch can never fire, and the error is
  "corrupt output buffer". The port computed it in `i64`.
- **`Model.params.tensor_split` pointed into a dead stack frame.**
  `Model.create` returns the model by value after pointing
  `params.tensor_split` at its own `tensor_split_owned`, so the copy kept a
  pointer into `create`'s frame. An even split never reads it. Under
  `LLAMA_SPLIT_MODE_TENSOR` with `tensor_split = {3, 1}`, the KV and
  recurrent buffers were sized from whatever had since been written there,
  and the first decode died in a Metal `mul_mat` with `ne00 == ne10`.
  `loader-diff`'s tensor-split case had passed because the frame had not
  been overwritten yet during load. `Model.ownTensorSplit` re-points it once
  the model is at its final address. The split states themselves were right
  all along, as the direct comparison showed.

### What the C++ itself cannot do

Measured, and so excluded rather than matched:

- **The meta backend cannot compute a graph `cb_eval` has cut at every
  node** (`GGML_ASSERT(i_start == cgraph->n_nodes)`,
  ggml-backend-meta.cpp:2207). The tensor configurations therefore record
  each node's structure when the scheduler asks, and answer no.
- **`offload_kqv = false` under tensor parallelism asserts on the
  attention's split axes** (ggml-backend-meta.cpp:826).
- **`seq_add` asserts for M-RoPE** (llama-kv-cache.cpp:575).

### The other gates, built alongside

| Gate | Covers | Checks | Injections |
|---|---|---:|---:|
| `api-diff` | llama.h's model, vocab, loading, chat, batch, logger entry points at the C boundary | 85,848,657 | 19/19 |
| `adapter-diff` | LoRA, aLoRA, control vectors; libc++ `unordered_map<ptr>` order | 278,212 | 13/13 |
| `quant-diff` | `llama_model_quantize` on every ftype, the type rules for 146 archs, the saver | 120,329,357 | 17/17 |
| `opt-diff` | `ggml-opt.cpp`, libc++ `shuffle` and `uniform_int_distribution` | 89,924 | 13/14, 1 provably inert |
| `opt-llama-diff` | `llama_opt_init`/`_epoch` on a synthesized F32 Qwen3.5 | 50,856 lines + 6 aborts | 8/8 |
| `meta-diff` | `ggml-backend-meta.cpp` on {Metal,Metal}, {Metal,CPU}, 1–4 devices; 94.8% of its lines | 76,645 + 2×6,528 lines | 14/14 |
| `batch-diff` | `llama_batch_allocr` and its splits | ~10.2M | 12/12 |
| `chat-diff` | the built-in chat templates | 952,391 | 11/11 |

Things those gates established that are worth knowing:

- **`opt.zig` reproduces libc++'s `shuffle` exactly**, against PLAN's earlier
  "document the divergence". It was measured against the real template,
  including the two-draw path.
- **Decision 40's "~96% unexecutable" for the meta backend was wrong.**
  It accepts the same device twice, so {Metal, Metal} gives a real
  all-reduce on one GPU. What stays unreached here is the `ggml_backend_comm_*`
  collective (NCCL on CUDA), the non-null host-buffer arms, and a few
  propagation arms no local model has.
- **An upstream use-after-free in the meta backend's `graph_compute`.**
  When `max_nnodes` grows, only the first `n_subgraphs` subgraphs are
  rebuilt. libllama reserves its largest graph first and never hits it. The
  port keeps the C++'s behaviour.
- **Upstream behaviours reproduced rather than fixed:**
  - the quantizer's `prune_layers` loads a renamed tensor's data under its
    new name;
  - the saver aborts on a precompiled charsmap;
  - saving a CPU_REPACK model segfaults the C++.
- **Training Qwen3.5 is mostly impossible on both sides.** Any weight
  upstream of a cache write fails the backward build ("inplace operations
  are currently not supported", ggml.c:7318), and the attention output
  gate's `sigmoid` has no backward. `opt-llama-diff` trains what can be
  trained and compares the six aborts.
- **Token-embedding LoRA is load-only in the gate.** Qwen3.5-2B ties its
  head to `token_embd.weight`, and the C++ aborts applying it.
- **An RWKV `token_to_piece` with a negative length differs on purpose.** The
  C++ compares against `(size_t)length` and writes past the buffer.

### After the swap

`parity-cli` 6/6, `parity-port` 6/6, `parity-port-cpu` 6/6, `node-diff`
2,750 nodes identical on CPU and GPU at 1 and 4 threads, and `sampler-diff`
40,289 records — all on the all-Zig library. `abi-check` grew from 15 to 34
entry points. Every `*_default_params` now *returns* a struct from a Zig
callee, one of them the one-byte `llama_sampler_chain_params`, and the
batch functions return and take `llama_batch` by value. All 34 arrive
intact.

Then every other gate was re-run against the swapped tree, with every
in-process harness now linking `libllama-ref.a`. All pass:

| Gate | Result |
|---|---|
| `make validate` | OK; it now includes `batch-diff`, `chat-diff`, `api-diff` and the `api/` and `models/` format check |
| `context-diff` | 2,417,100 checks |
| `adapter-diff` | 278,212 checks |
| `quant-diff` | 120,329,357 checks |
| `opt-diff` | 89,924 checks |
| `opt-llama-diff` | all cases, plus 6 abort cases |
| `meta-diff` | 76,645 end-to-end lines |
| `ops-diff` | 650 results |
| `repack-diff` | 60 checks |
| `metal-diff` | 65 answers |
| `tuning-diff` | 3,528,489 lookups |
| `struct-layout` | — |
| `probe` | — |
| `backend-ops` | 21,093 configurations |

### Left for later

- **Architectures other than Qwen3.5** (Decision 36's follow-on milestone).
  - The graph builders and `load_arch_hparams` exist for Qwen3.5 only.
  - The quantizer therefore quantizes end to end only for Qwen3.5, though
    its type rules are gated for every architecture.
  - `createMemory`'s iSWA arm aborts.
  - Encoder–decoder (T5) cross-attention in `encode` aborts.
- **The xcframework (`xcfwk/`) still links Zig's libc++, libc++abi and
  compiler_rt** into each slice by `ld -r`. Since the swap nothing in either
  library needs the C++ runtime, so that step carries dead weight and can go.
  `xcfwk/` was another session's work and is untouched here.

### Rebased onto the Linux port (v0.0.18)

v0.0.18 made the tree build and run on Linux aarch64 against a C++
libllama. Rebasing the swap onto it conflicted only textually, in
`loader.zig`, where `File` had moved to `mmap.zig` and taken 0.0.18's
`errno` fix with it. But the Zig libllama had only ever been compiled for
Darwin, and on Linux it neither built nor loaded a model:

- **`llama_log_internal` was a Zig variadic**, which Zig 0.17 refuses to
  define on aarch64 outside Apple and Windows. Unlike ggml's three, which
  `ggml.h` exports and C callers reach with varargs, nothing outside
  libllama calls it: it is C++-linkage and absent from `llama.h`. So rather
  than a per-target C shim, the variadic is gone. `logInternal` takes its
  arguments as a tuple and formats with `snprintf` -- Zig *calls* a C
  variadic on every target -- and there is one path on every target. The
  mangled export went with it; the harnesses install their callback on the
  port's logger directly.
- **Darwin's stdio and `errno` names**: `__stderrp` in `log.zig` and
  `opt.zig`, `stdout()` in `quant.zig`, `__error` in `mmap.zig`.
  `impl.cStdout` and `cStderr` read the right variable per OS. glibc's
  `FILE` is opaque, so `llama_model_load_from_file_ptr` takes `?*c.FILE`.
- **`_SC_PAGESIZE` was Darwin's 29; glibc's is 30.** `sysconf(29)` returned
  some other setting, and the first `unmap_fragment` aborted on
  `last % page_size == 0`. `PATH_MAX` was Darwin's 1024 too.
- **`mmap.zig` had ported only the Darwin arms of `llama-mmap.cpp`.** The
  `__linux__` ones are now in: direct I/O through an `O_DIRECT` descriptor,
  with aligned-chunk reads, the `EINTR` retry, the buffered fallback on
  `EFAULT`/`EINVAL` and zero-filled padding at EOF; `posix_fadvise` and
  `MAP_POPULATE` on map; and the `RLIMIT_MEMLOCK` suggestion when `mlock`
  fails. Zig 0.17 declares no Linux `struct stat`, so `init_fd`'s `fstat`
  goes through `statx` for the same two fields.

One fault ran the other way. v0.0.18's `ggmlAbort` passed a `*[2048]u8`
where `[*:0]const u8` is wanted. Only targets where Zig defines the
variadic compile that function, so 0.0.18 built on Linux and **did not
build on macOS**.

Checked in a Debian 13 container (glibc 2.41) under podman, against
`libllama-ref.a` built for the same target. The CLI's greedy output equals
macOS's CPU-only run token for token. A driver loading under each of `dio`,
`mmap`, `none` and `mlock`, from a bind mount and from the container's own
filesystem, gives a log and output byte-identical to the C++'s in all eight
runs. Under `dio` neither side logs the buffered fallback, so the
`O_DIRECT` path really ran. Under `mlock` both print the Linux hint. This
needs glibc 2.36 or later: the sampler's `arc4random`, which was already
there in 0.0.18, is missing from older glibc.

## Stage 6 — the CUDA host layer, ported and swapped (GB10 machine)

Everything in this section ran on the second machine (Decision 43): Linux
aarch64, NVIDIA GB10 (compute 12.1, `sm_121a`), CUDA 13.0. Nothing here ran
on the Mac; `PLAN.md`'s "Mac pickup" lists what the Mac has to re-run.

`ggml-cuda.cu`'s host code and every `.cu` file's host half are Zig under
`src/ggml/cuda/` — 78 files, ~18,800 lines, 553 citations. The kernels
are upstream's `.cu` files compiled by `nvcc -fatbin` with the reference's
flags and loaded through the driver API (Decisions 44–46); CUB's device-wide
algorithms sit behind `src/ggml/cuda/cub_shim.cu` (Decision 47). **The
swap is done**: `-Dcuda-port` defaults to true, `isPorted` is gone, and
`supports_op` is the C's predicate alone. `-Dcuda-port=false` still builds
upstream's host layer, which only the gates use.

### What the gates say at the swap

| Gate | Result |
|---|---|
| `scripts/cuda-ops-diff` | **14,581 / 14,581** supported cases bit-identical; `supports_op` asked 188,935 times, 0 claimed beyond the reference, 0 refused that the reference accepts; 10 cases `NONDET` (below) |
| `scripts/cuda-node-diff` | Qwen3.5-2B: **2,750** nodes identical at 1 and 4 threads; the **fused** pass, 33 decodes' full logits identical at 1 and 4 threads |
| `scripts/cuda-backend-diff` | **2,805** vtable answers identical on the physical device, **8,387** with `GGML_CUDA_DEVICES=3` |
| `scripts/cuda-parity-cli` | the shipped `llama-cli` against the reference driver, greedy: **6/6** prompts identical |

Generation runs at ~136 t/s, as the reference build did (~137).

### The ops gate, and three holes found in it

`cuda-ops-diff` is upstream's `test-backend-ops` with
`ggml_backend_compare_graph_backend` renamed to a harness hook that runs the
case on the port and on the **reference CUDA backend** (upstream's objects,
`ggml-cuda.h` exports renamed `ref_*`) and compares every node on bits. Its
first version had three holes, each found by an injection or a contradiction
rather than by reading:

- **The reference could never fuse.** `ggml_backend_graph_copy` fills the
  copy's `nodes[]` directly and builds no use counts (ggml-backend.cpp:2202),
  so the second backend of every whole-graph comparison runs unfused —
  harmless upstream, where it is the CPU. Measured: with fusion ported, every
  fusion family "failed"; with the port's fusion switched off, they all
  passed — the gate was comparing a correct fused port against an unfused
  reference. The harness now re-expands the copy node by node into a fresh
  graph (same nodes, same order — asserted — real use counts). Two details
  cost a run each: test-backend-ops appends op-`NONE` sentinel nodes, which
  expanding files as leafs, and an op-`NONE` *parameter* is a node to the
  expansion already. After the fix, an unfused port fails every family
  (SILU_MUL 9/14, RMS_NORM_MUL_ADD 2/30, TOPK_MOE 209/320, RMS_NORM_MUL_ROPE
  63/144, MUL_MAT_VEC_FUSION 180/192, SNAKE_FUSE 6/16).
- **Shapes upstream does not test.** Three paths had no case at all, each
  found because an injection escaped. The harness now owns `main`
  (`test-backend-ops.cpp` is compiled `-Dmain=test_backend_ops_main`) and
  runs extra cases on fresh backends first:
  - `MEAN` sends one row longer than 65536 to `cub::DeviceReduce::Sum`;
    upstream's longest single row is 32769. A `divide_by_count` fault passed
    7/7 until `MEAN([65537])` and `MEAN([262144])` were added — and
    `MEAN([65536])` passed under the same fault, which pins the threshold.
  - No upstream case builds the gated-delta-net snapshot `CPY` that
    `ggml_cuda_try_gdn_cache_fusion` matches. `slot_stride = 0` passed
    36/36 until a K = 3 case with a padded cache was added.
  - Upstream's test-mode `MUL_MAT_ID` batches are 1, 4, 5, 17, 32 and 129;
    the MMVQ/MMQ boundary is at 8 on this GPU. Moving it by one passed
    869/869 until batches 8 and 9 were added on q4_0 and q4_K; both 8s fail.
- **The reference reads out of bounds.** `flash_attn_mask_to_KV_max` reads
  up to `ncols1 − 1` mask rows past the mask when `Q->ne[1] % ncols1 != 0`.
  Where the graph copy puts the mask last in its buffer, the reference
  faults — 62 of 432 such cases, varying with what is mapped past the end.
  The harness interposes `cudaMalloc` and gives every allocation 1 MiB of
  slack on both sides; all 432 now run, and all pass. The rows read feed
  rows nobody keeps, so no result changes.

### Where the reference's own answer is unspecified

`CUDA_DIFF_SELF=1` puts a second reference backend where the port goes. A
case that differs is re-run that way; if the reference disagrees with
itself it is `NONDET` and not failed. Two sources, both measured:

- **CUB's float `DeviceScan::InclusiveSum`**, which `CUMSUM` uses on long
  rows, is nondeterministic run to run even in a standalone program: of 49
  consecutive pairs, 0 differ at 20,481 elements, 23 at 242,004, 43 at
  375,960.
- **The fused MMVQ for `MUL_MAT_ID` with lane scale, `ADD_ID` bias and a
  gate, on NVFP4.** Unfused (`GGML_CUDA_DISABLE_FUSION=1`) the reference
  agrees with itself; fused it does not, and where it differs moves between
  runs. 8 cases. The port's fused output for that configuration is
  therefore not gated.

### `node-diff` could not see fusion or CUDA graphs

`harness/node_dump.c` hashes every node by answering the scheduler's `ask`
with true, which splits the graph at **every node**. Neither side can fuse
or capture a CUDA graph in that pass. Measured: switching off the port's
RMS_NORM + MUL fusion left all 2,750 nodes identical. `node_dump` gained a
`fused <steps>` mode with no callback, hashing the full logits of the prompt
and each greedy step, and `cuda-node-diff` runs it after the per-node pass.
It is deterministic (three reference runs, identical).

Proving it needed care, because **turning a fusion off is inert on
Qwen3.5**: the reference's own `GGML_CUDA_DISABLE_FUSION=1` gives identical
logits, so every fused kernel the model reaches matches its unfused route
bit for bit. An injection that switches fusion off is therefore not
evidence. A wrong GLU op in the fused MMVQ gate path is — the per-node
passes stay identical, the fused pass differs from the first single-token
step, and `cuda-parity-cli` fails 6/6. A trace of the port's fusion on this
decode: RMS_NORM+MUL 237, MUL_MAT+bias 60, gate/up+GLU 48, GDN+CPY 54,
SSM_CONV+SILU 54, SILU+MUL 54, SOFTPLUS+MUL 36, SIGMOID+MUL 18.

### Fault injections

Each built (library timestamp checked), run, reverted, and rebuilt.

| Where | Fault | Caught by |
|---|---|---|
| `scale.zig` | bias dropped | ops-diff |
| `kernels.zig` ABI check | an `i32` passed where the kernel takes `i64` | the launch's size check, before running |
| `mmvq.zig` | rows per block 4 | ops-diff, 20 BITDIFF |
| `mmvq.zig` | small-k off; rows per block 1 | **inert** — zero partial sums, over-coverage |
| `mmvf.zig` | block size 32 | ops-diff |
| `fusion.zig` | port unfused, reference fused | ops-diff, every fusion family |
| `fusion.zig` | topk-moe `clamp` dropped | ops-diff, TOPK_MOE 264/320 |
| `fusion.zig` | GDN cache `slot_stride = 0` | ops-diff **extra case only** |
| `fusion.zig` | GLU op wrong in the fused MMVQ gate path | node-diff fused pass, parity-cli 6/6 |
| `sum.zig` | `ne − 1` into `DeviceReduce::Sum` | ops-diff, SUM 0/7, CROSS_ENTROPY_LOSS 0/2 |
| `mean.zig` | `ncols + 1` into `divide_by_count` | ops-diff **extra cases only** |
| `mulmat.zig` `mulMatId` | row map without `iex % ne11` | ops-diff, 10 cases |
| `mulmat.zig` `mulMatId` | MMVQ/MMQ boundary `<=` to `<` | ops-diff **extra cases only** |
| `lightning_indexer.zig` | `n_kv − 1` | ops-diff, 0/144 |
| `lightning_indexer.zig` | wrong `K_VECS_PER_BLOCK` | the kernel table: the key does not exist, compile error |
| data movement (worker C) | `getrows`, `set_rows`, `concat`, `cumsum`, `cpy`, `set`, `argsort`, `top_k`, `cub_shim` — one each | ops-diff, 9/9 (a first `cub_shim` fault, `end_bit = 31`, was inert: test rows never vary the sign bit) |
| flash attention (worker E) | 10 across `fattn_mma`, `fattn_tile`, `fattn_vec`, `fattn_common`, `fattn` | ops-diff 8/10; two inert — a tile `nbatch_fa + 1` only bounds `parallel_blocks`, and the quantized Ada vec rule has no `nb = 2, kv % 256 == 0` case |
| `fattn.zig` `getAllocSize` | scratch end + 128 | backend-diff |
| `fattn.zig` `getAllocSize` | scratch end − 128 | backend-diff — by an assertion abort (the size underflows), not by the comparison |
| `backend.zig` `offload_op` | `>=` threshold to `>` | backend-diff |
| `backend.zig` `get_alloc_size` | row padding 256 instead of 512 | backend-diff |

### `cuda-backend-diff`

`harness/cuda_backend_diff.c` asks both registries — port and `ref_*` — the
same questions in one process: registry name, count, proc addresses and
features; per device, props, memory total (free memory moves, so it is not
compared), both buffer types, `supports_buft`; `offload_op` across the
threshold of 32 for every arm of `get_op_batch_size`; `get_alloc_size` for
every type around the 512-element row padding, and for 240 flash-attention
nodes at five data addresses each. 120 of those shapes give a different size
at a different address, which is why the address varies.

### CUB, and what is C++ in the shipped binary

`cub_shim.cu` is the only host code the port writes in C++: eight
`extern "C"` wrappers — six sorts, `InclusiveSum<float>`, and
`DeviceReduce::Sum` (one wrapper serves `sum.cu` and `mean.cu`, whose call
sites have identical argument types). Measured with `cuobjdump -sass`:
its kernels are exactly the reference's CUB kernels by mangled name, with
identical SASS. `llama-cli` contains no upstream host function
(`nm -C | grep ggml_cuda_op_` is empty) and 136 `cub::` symbols, which are
the shim's.

### Left unported

- **Multi-GPU collectives**: `allreduce.cu` and the internal all-reduce
  pipeline. `comm_init` falls back to the butterfly with the C's warning, as
  upstream does when the pipeline is unavailable. `GGML_CUDA_DEVICES` gives
  virtual devices on one GPU, which reaches the registry and vtables
  (`cuda-backend-diff`) but not a meaningful all-reduce.
- **AMD and MUSA arms**, and the AMD flash-attention config tables.
- **Arms this GPU cannot reach**: Volta/Turing/Ada-only dispatch, `top_k`'s
  `cub::DeviceTopK` (needs CCCL ≥ 3.2, not compiled by the reference either).
- **`cuda-struct-layout`** is not built. The by-value kernel argument
  structs are covered indirectly: every launch's first call checks each
  argument's size against `cuKernelGetParamInfo`, and every op gate passes
  on bits. Field offsets inside a struct are not checked separately.

## Mac pickup: the Mac gates after Stage 6

The CUDA work was done on the GB10 machine, which cannot run a Metal gate
and, by choice, ran no git command. `PLAN.md`'s "Mac pickup" listed what
only the Mac could finish. Everything below ran on the Mac (Apple M5 Max).

### What the gates say

| Gate | Result |
|---|---|
| `make validate` | green after the two fixes below; `zig build test` 459/461 (2 skipped) |
| `make node-diff ARGS=--gpu` | 2,750 nodes identical at 1 and 4 threads, **and** the new fused pass: 33 decodes' full logits identical at 1 and 4 threads |
| `make metal-diff` | 65 vtable answers identical |
| `make sched-diff` | 16 assignments identical |
| `make graph-diff` | 131 nodes identical |
| `make parity-cli` | 6/6 prompts identical |
| `make backend-ops` | 21,093 op configurations, 2/2 backends |

`harness/node_dump.c`'s four-argument mode gives the same 2,750 nodes it
always has, so the per-node pass is unchanged by the `fused` mode beside it.

### Two gates the CUDA machine could not have run, both broken

- **`scripts/port-coverage` compiled nothing.** A bare `zig build-obj`
  needs a stand-in for the `config` module the build graph supplies, and the
  script's comment says every option `build.zig` defines has to appear in
  it. The CUDA build added `use_cuda`, `cuda_port` and `cuda_arch`;
  `src/ggml/backend_reg.zig` and `src/ggml/module.zig` read two of them, so
  `ported.zig` stopped compiling and `make validate` stopped there. The
  stand-in now defines all three, off.
- **`scripts/port-links` had never resolved a CUDA citation.** It collects
  reference files by extension and knew `.c`, `.h`, `.cpp` and `.hpp`, so
  every one of the 589 citations of a `.cu` or `.cuh` read "no such file
  under llama.cpp (or ambiguous)". The CUDA machine checked its 553
  citations with a script over the working tree, which is how they passed
  there. With `.cu` and `.cuh` added, 34 real problems were left:
  - **20 C++ member names the checker cannot parse**: it takes plain
    identifiers in backticks, so `ggml_backend_cuda_context::stream(int,
    int)`, `ggml_cuda_pool_alloc<T>` and `~ggml_backend_cuda_context` named
    no symbol and were never checked. Rewritten the way libllama's
    citations already were, ``ggml_backend_cuda_context`'s `stream` ``.
  - **Seven line numbers pointed at the wrong line**, now right: the
    `bin_bcast_cuda` functor (cited at a `}`, it is at 389), `pool(int)`
    (cited at the body of `pool()`), the C++ `ggml_can_fuse_subgraph`
    overload (cited at a `return` in another function), the `ggml_cuda_lock`
    trio (three symbols, one number), `MMQ_DP4A_TXS_*` (cited under the
    function name), `GGML_CUDA_CC_*`, and a citation in `graph.zig` that
    had been written twice with two numbers.
  - **Three ranges** (`convert.cu:273 to 376`, `ggml-common.h:95 to 172`)
    that the checker reads as malformed, and **three citations with no
    commit**, one in `cli/session.zig`.

  4,381 citations in 249 files pass. Negative-tested: drifting one CUDA
  citation by two lines fails it.

### Metal's fusion had never been compared, and now is

`node-diff`'s per-node pass answers the scheduler's `ask` with true, which
splits the graph at every node, so neither side ever fused. The CUDA work
added a `fused <steps>` mode to `harness/node_dump.c` and ran it from
`cuda-node-diff`; `scripts/node-diff` now runs it after the per-node pass
on both devices. Measured on Metal: swapping the two fused-operand buffer
bindings of the norm encoder (`ops.zig`, slots 2 and 3, which only a fused
`RMS_NORM + MUL` reads) leaves **all 2,750 per-node comparisons
identical**, and the fused pass differs from the prompt's logits on. Same
separation as on CUDA. Reverted by reverse replacement; the gate passes on
the reverted tree.

One thing this run did not explain: the `libggml.a` built by the first
`make node-diff` hashed differently from every later build of the same
tree, while three later builds, one from a `touch`, agree with each other.
Nothing under `src/ggml/` was edited between the first build and the
later ones except the injection and its exact revert.

### `zig build test` passes here

On the GB10 it fails through the runner's `--listen=-` mode, and only for
tests that make ggml log to stderr. Here the same runner prints those
tests' stderr under a `failed command:` banner -- which reads like a
failure -- and still reports the step green, 426 pass and 2 skip, with
`Build Summary: 57/57 steps succeeded`. Shared code is not what differs.

### What else changed

- `make validate` formats `src/ggml/cuda/*.zig`, the one directory of Zig
  its list missed.
- `make cuda-ops-diff`, `cuda-node-diff`, `cuda-backend-diff` and
  `cuda-parity-cli` build what each script's `Requires:` line names and
  run it; on anything but Linux they refuse.
- No shared script gained a Linux branch: the CUDA work wrote separate
  `cuda-*` scripts and left the Mac's alone, so the "run each changed
  script once on the Mac" item had nothing in it.

## The other architectures (Decision 36's first follow-on)

Qwen3.5 was the only architecture with a graph builder. Upstream has 145
model classes in `llama_model_mapping` (`clip` and `gptj` have no usable
implementation, upstream says so, and are left out); this milestone ports
the rest.

### The gate: upstream's own fixtures, `context-diff` across architectures

There are no real models of 140 architectures on this machine, and there
did not need to be. `llama.cpp/tests/test-llama-archs.cpp` builds a small
random model of every architecture from one recipe (`get_gguf_ctx`: n_embd
256, two layers, a 128-token `no_vocab` vocabulary, every hyperparameter
key any architecture reads), and its `-o <dir>` mode writes each as a GGUF.
`scripts/gen-arch-fixtures` compiles that file **unmodified** against
`libllama-ref.a`, with `harness/arch_fixtures_shim.cpp` standing in for the
six `common` functions it calls. So a fixture is upstream's definition of a
model of that architecture: a port that reads a key or a tensor the C++
does not write fails to load, rather than passing on a fixture written to
suit it.

`scripts/arch-model-diff` runs each fixture through
`harness/context_diff.zig` -- the harness behind `make context-diff`,
unchanged in what it compares: everything logged, every graph node (op,
shape, strides, `op_params`, sources, data hash), every output row's bits,
the state API -- on the GPU and the CPU, 22 cases each. `context-diff` is
Qwen3.5 in depth; this is every architecture in breadth.

What it took to make the fixture set complete:

- **`-o` skips 44 architectures.** `llama_model_saver` cannot rebuild their
  metadata from a loaded model, or `arch_supported` marks them FIXME, or the
  loops skip them. `harness/arch_fixtures_extra.cpp` includes
  `test-llama-archs.cpp` textually with `main` renamed, so its static
  `get_gguf_ctx` is the recipe here too, builds the model through the C++ on
  the CPU, and writes the metadata as built plus each tensor once by name --
  the saver's `add_tensor` rule. One process per architecture: the C++
  asserts on some, and an abort ends the process.
- **The BERT family's "TODO vocab" is one missing key.** Their
  `load_arch_tensors` sizes the token-type embedding from
  `tokenizer.ggml.token_type_count`, which `llama_vocab::impl::load` reads
  before its `no_vocab` return; no vocabulary is needed. Found by the worker
  porting them.
- **RWKV needs head size 64 on Metal.** The WKV kernels assert `C / H == 64`
  and the recipe gives 128, so every GPU case aborted in ggml before RWKV's
  code ran. An `hs64` variant sets it; RWKV6-Qwen2 also needs
  `token_shift_count` 1 (the C++ asserts `n_embd == n_embd_r()` otherwise)
  and must not carry the optional `time_mix_first`, which sends upstream
  down a path that reads `time_mix_ln` it never loads and segfaults.
  RWKV-7 and ARWKV-7 read four LoRA ranks the recipe never writes -- the
  whole reason the C++ could not build them; `hs64nogate` is ARWKV-7's
  gateless form.

- **Granite-Switch, eagle3 and DFlash need more than keys.** Their loaders
  want adapter token maps, target-layer lists and, through
  `get_tensor_meta`, tensors the user-data path never shows them.
  `harness/arch_fixtures_draft.cpp` (from the worker that ported them) adds
  exactly those, writes the file, and reloads and decodes it through the
  C++ before it counts: eight fixtures, among them DFlash's iSWA, DSpark
  and DeepSeek-V4 forms.

159 fixtures. Two architectures have none: deepseek2-ocr, which the C++
cannot build from the recipe, and the Gemma 4 assistant, which needs a
target model's context and is gated by `make gemma4-assistant-diff`
instead -- a target and an assistant built from metadata alone, dense and
MoE, on both devices.

### What the gates say

| Gate | Result |
|---|---|
| `make arch-model-diff` | **157 / 159 fixtures pass**, 0 fail, 0 unported; the other 2 are `REF-ABORT` -- PLM (upstream's bug) and RWKV6-Qwen2's unmodified recipe (the C++ asserts; its `hs64` variant passes 22/22). Cases where the C++ and the port abort on the same assertion: 66, counting each device, across 13 passing fixtures; one documented upstream nondeterminism (T5, q8_0 cache, Metal) |
| `make gemma4-assistant-diff` | dense and MoE targets, GPU and CPU: identical |
| `make validate` | green: unit tests 459/461 (2 skipped) and 300/300; 5,483 citations in 402 files; `loader-diff` 253,154 checks; `api-diff` 85.8M |
| `make context-diff` | 2,417,100 checks, 0 failures (Qwen3.5, every configuration) |
| `make node-diff`, `ARGS=--gpu` | every node and every fused decode identical, CPU and Metal |
| `make parity-cli` | 6/6 |
| `make sampler-diff` | 40,289 records identical |
| `make adapter-diff` | 278,212 checks, 0 failures |
| `scripts/quant-diff` | 120,329,357 checks, 0 failures |
| `scripts/opt-llama-diff` | 50,856 transcript lines and 6 abort cases identical |

### What the gate learned about itself

- **A crash is split case by case.** When a model configuration crashes,
  each case runs alone, and a crashing case runs again with each side alone
  (`CONTEXT_DIFF_SIDE`). The C++ and the port aborting on the same assertion
  is a match (`+1abort`); a C++ segfault, with no assertion to compare, is
  listed as `+1refcrash` and not counted a match -- the first version
  compared two empty strings and called them equal. A port-only abort fails;
  negative-tested. A fixture every case of which the C++ refuses is
  `REF-ABORT` -- PLM at v0.3.0, which upstream's own test skips as "TODO
  tensor shapes".
- **The port-only side must not run the C++.** The harness's warm-up
  context went through the C++ even for `CONTEXT_DIFF_SIDE=port`, so a
  fixture the C++ aborts on would have read as the port aborting.
- **The draft models were compared not at all, by that fix.** eagle3 and
  DFlash have encoders too, which read the target model's features as
  embeddings: given the prompt's tokens, both sides segfaulted, and the gate
  read that as `+refcrash` and PASS while real injections went uncaught.
  Found by the worker porting them, who drove them as
  `common/speculative.cpp` does -- an embedding batch of features first,
  eagle3's g-embeddings beside its tokens, DFlash's K/V injection -- and the
  harness now does the same.
- **An encoder-decoder was compared not at all.** The harness drove
  `llama_decode` only; for T5 the decoder asserts "llama_encode must be
  called first" on both sides, so T5 "passed" on loading alone -- a decoder
  ADD turned SUB passed. Each round now encodes first for a model with both
  halves, as upstream's test does.
- **Two cases are the C++'s preconditions for a model without memory**
  (encoders, embedding and diffusion models decode by encoding):
  `n_ubatch 7` and `n_outputs_max 5`. Skipped for those, and counted.
- **One case is upstream's nondeterminism.** T5 with a q8_0 KV cache on
  Metal: the quantized V transpose falls back to the CPU, and the split it
  adds lets the host `pos_bucket` input be overwritten before it reaches the
  GPU -- the decoder's GET_ROWS on `attn_rel_b` reads values around ±2.1e9,
  different in every process. Shown to be upstream's by running pure
  upstream (llama.cpp.zmake) alone with a probe; the logits agree because
  the cells read are masked. Listed in the script's `KNOWN_NONDET` and
  counted as `+1nondet`.

### How the work was done

Eighteen workers, in three waves, each in a private copy of `src/llama` gated
with `ARCH_DIFF_PORT`/`ARCH_DIFF_LIBS`, so a half-written file never broke
anyone's build -- the failure the last parallel session recorded. Each
copied a snapshot; the merge copied new files, merged the registry, and
applied shared-file changes as diffs against that snapshot. Every
architecture had a fault injected and caught, or a recorded reason the
injection was inert.

Two merge faults, both found by checking rather than trusting:

- **Three architectures were ported, gated, and silently left out of the
  registry.** The merge recognised one-line arms only, and three were
  multi-line `Entry` literals. Every worker's arms are now checked against
  the tree's.
- **Three workers wrote Llama 4's attention-temperature input** and two the
  same libc++ `unordered_set`. One of each was kept: the temperature input
  without the fusion one copy named, since no fixture reaches a non-zero
  logarithm and the project names a fusion only when something can say it
  is wrong; the set measured against real libc++.

### Assertion lines

A multi-line `GGML_ASSERT` reports the line of its closing parenthesis --
measured with `zig c++`, through a nested macro too. A checker over every
`impl.assert`, `impl.abort` and `ggml_abort` in `src/llama` found 8
multi-line assertions cited at their first or a middle line, and, matching
each assertion's condition text to the nearest upstream one, **44 cited a
line or two off** -- most in code ported long before this milestone
(`kv_cache.zig`, `memory_recurrent.zig`, `context.zig`, the vocabulary). The
reference's own abort at `llama-context.cpp:2189` confirmed one: the port
said 2190. These lines appear only in abort messages, and no gate compared
those death paths. All fixed.

### Two shared structures that silently knew only Qwen3.5

Generating `Layer` with all of `llama_layer`'s members broke two places
that had listed the old 75 by hand, and neither was reached by the fixture
gate:

- **Training offered the wrong tensors.** `context_opt.zig` walked a
  hand-ordered list of `Layer`'s members to hand them to the parameter
  filter in the C++'s order; a comptime check refused the build once
  `Layer` grew, which is what it was for. It now walks `Layer`'s fields as
  declared -- generated in `llama_layer`'s order -- recursing into the
  nested structs, as the C++ walks the struct as an array of pointers. The
  model-level half offered eight of the C++'s fourteen tensors, on the
  grounds that no ported architecture created the other six; GPT-2's
  position embeddings and BLOOM's token norm now do.
- **`loader-diff` compared 75 members of each layer.** Its C++ side read
  them through a hand-written `offsetof` table. `scripts/gen-layer` now
  writes `harness/layer_fields.inc` and `harness/model_fields.inc` beside
  the Zig, and the harness walks every member generically: 253,154 checks
  where there were 13,719, and no architecture left loading vocab-only.
  Negative-tested by putting a Qwen3.5 tensor in the wrong field.

### What no fixture reaches

Ported as the C++ is, ungated here, from the workers' reports:

- **MTP / NextN heads**, everywhere: the recipe sets no
  `nextn_predict_layers`, and setting it makes the C++ refuse the two-layer
  fixtures. That is glm4-moe, hy-v3, bailingmoe2/3, deepseek2/32/4, glm-dsa,
  mimo2, cohere2moe, step35, qwen35moe, qwen3next and nemotron-h-moe's
  `graph_mtp`, and their MTP-context arms of `create_memory`.
- **Deepstack** (qwen3vl, qwen3vlmoe, granite, granite-swa): the recipe's
  `n_deepstack_layers` is 0.
- **Llama 4's attention temperature** (llama4, mistral3, deepseek2): every
  fixture position gives a logarithm of 0, so its arithmetic -- and whether
  clang fuses it -- is untested.
- **Gemma 4's non-SWA layers and KV reuse on its fixture**: the recipe's
  scalar pattern 5 makes every layer SWA. The assistant harness's target
  covers both.
- **DeepSeek 4's unfused hyper-connection paths**: every configuration
  resolves the fused ops, so an injection there passed, inert.
- **Dense fixtures of hybrid MoE architectures**: jamba's, granite-hybrid's
  and nemotron-h's MoE branches.
- Smaller arms each worker listed: qwen3next's legacy `ssm_in`, the
  Kimi models' `wkv_b` fallback, WavTokenizer's posnet attention, smollm3's
  no-RoPE layer, deci's attention-free layers.

### Upstream's own bugs, reproduced and matched

The C++ and the port abort identically on each: PLM's flattened `v_states`
(every case); Chameleon's image-logit write into an empty output; PLaMo-2's
`kv_head * n_seqs` state offset; DeepSeek 4's `state_write` calling
`seq_pos_max` without its range check, and its `n_embd_out = hc * n_embd`
overrunning the embedding buffer; T5's encoder positions under
`split_equal`; RWKV6-Qwen2's `time_mix_first` path and its `n_embd_r`.

## CUDA pickup: the Mac's list, run on the GB10

`PLAN.md`'s "CUDA pickup" had four items. All are done; two needed fixes in
the harnesses, and one turned up a script that cannot run here as written.

### What the gates say

| Gate | Result on the GB10 |
|---|---|
| `make cuda-ops-diff` | every supported case bit-identical to the reference CUDA backend |
| `make cuda-node-diff` | every node and every fused decode identical |
| `make cuda-backend-diff` | 2,805 + 8,387 vtable answers identical |
| `make cuda-parity-cli` | 6/6 |
| `zig build test` | green: 428/430 (2 skipped), 394/394 steps |
| `make validate`, the parts that run here | formatting (with `src/ggml/cuda/`), the scaffold, all four test suites, `libcxx-diff` (452,904), `port-links` (5,483 citations in 402 files) |
| `make arch-model-diff` | **157 / 159 pass**, 0 fail, 0 unported; the same 2 `REF-ABORT` as the Mac. 49 abort matches across 12 fixtures, each in a `gpu`/`cpu` pair apart from T5's CUDA case below |
| `make gemma4-assistant-diff` | dense and MoE, CUDA and CPU: identical |

`zig build test` was not a regression in shared code: it prints the
runner's `failed command:` banner over the stderr-logging tests, exactly as
on the Mac, and exits 0.

### The arch scripts linked Metal, and now link per platform

`gen-arch-fixtures`, `arch-model-diff` and `gemma4-assistant-diff` called
`xcrun` and linked four frameworks. `scripts/platform-libs`, sourced, sets
the link line for either machine: `platform_libs cc` for a `zig cc`/`zig
c++` link, `platform_libs exe` for `zig build-exe`. The two spell an rpath
differently (`-Wl,-rpath,…` against `-rpath …`), and `build-exe` refuses
the first. The fixtures come out byte-identical to the first Linux batch,
all 159.

### Two C++ runtimes in one binary, and every exception aborted

The cuda-* scripts link the system `libstdc++.so` for the nvcc objects.
Doing the same here made **every C++ exception in the reference
uncatchable**. `llama_context`'s constructor throws by design when a Gemma 4
assistant has no target, and `llama_init_from_model` catches it; on the
GB10 the process aborted in `__gxx_personality_v0`.

Zig appends its own `libc++abi.a` *after* every input, so a `libstdc++.so`
earlier on the line wins every runtime symbol nothing in `libc++.a` has
already pulled in. Measured with `nm`: `__cxa_throw` came from Zig's archive
(hidden, `t`), and `__gxx_personality_v0` from libstdc++ (`U`). So the
binary threw a libc++abi exception and unwound it with libstdc++'s
personality routine, which cannot handle it.

The fix is to leave `libstdc++.so` off. The only GNU-ABI C++ in the port's
`libggml.a` is `cub_shim.o`, and Zig's libc++ supplies all of it:
`std::runtime_error` keeps the GNU mangling in libc++, outside its `__1`
namespace. libcublas still loads libstdc++ at run time through its own
`DT_NEEDED`. The `rwkv7`/`arwkv7` lines in `gen-arch-fixtures` changed with
it, from a bare "aborted" to "the C++ cannot build the model", which is
upstream's exception now being caught as on the Mac.

**This does not carry to a `-Dcuda-port=false` build.** Upstream's host
objects need `std::__throw_length_error`, `std::condition_variable::wait`
and other libstdc++-only symbols, so a harness over that library has to
link both runtimes and cannot catch an exception. None of the four cuda-*
gates throws.

### CUDA's free-memory log line

On a unified-memory device, `ggml_backend_cuda_get_available_uma_memory`
reads `/proc/meminfo` and logs `final available_memory_kb: N` at debug
level. N moves between any two reads. `context_diff.zig` and
`gemma4_assistant_diff.zig` now normalize it next to the `MiB free` lines;
`api_diff`, `adapter_diff` and `loader_diff` have the same `normalize` and
will need the same line when they first run here.

### T5's `q8_0` case is upstream's abort on CUDA, not nondeterminism

`KNOWN_NONDET`'s T5 entry was keyed on `gpu`, so CUDA skipped it too. Run
without the skip, the case aborts identically on every run, on both sides:
`ggml_cuda_cpy: unsupported type combination (q8_0 to q8_0)`.
Upstream's own host layer (`zig-out/cuda-upstream`) aborts at the same
node, at `cpy.cu:609`. Its `supports_op` returns `true` for every `CONT`,
and `ggml_cuda_cpy` has no non-contiguous quantized same-type path, so the
quantized V transpose is claimed by CUDA and then refused. The port matches
it.

So the entry is Darwin-only now, and `abort_of` recognizes a formatted
`ggml_abort`, `<file>:<line>: <message>`, when no assertion is found.
Without that second change the case read `+1refcrash`: both sides printed
the same message, and the gate saw none. T5 now reads `gpu:17+4abort`.
Checked on synthetic logs: an assertion still takes priority, two
different messages stay different, and an ordinary log line is not taken
for an abort.

### What cannot run here

- **`scripts/port-coverage`** measures the macOS contract
  (`-DGGML_USE_METAL -DGGML_USE_ACCELERATE`, `use_metal = true`), and it
  strips one leading underscore as the Mach-O C prefix. On ELF that turns
  `_ZN…` into `ZN…`, which slips past the `^_Z` C++ filter. It fails here on
  `M_PI` in `ggml.c` before measuring anything, and fixing the defines
  alone would produce wrong counts, not just different ones.
- **The in-process diffs in `make validate`** (`regex`, `unicode`,
  `vocab`, `arch`, `loader`, `batch`, `chat`, `api`) still link the Metal
  frameworks, so `make validate` stops at `regex-diff`. `platform-libs` is
  what they would use.

## Stage 5: the measurement

`PLAN.md`'s Stage 5 section was three sentences. This is how it was
measured, on the Mac, before any Stage 5 code was written -- and the one
item the CUDA pickup sent back here.

### The Mac re-run of the arch gates

The CUDA pickup moved `gen-arch-fixtures`, `arch-model-diff` and
`gemma4-assistant-diff` onto `scripts/platform-libs` and taught `abort_of`
the formatted `ggml_abort`. Both changes were exercised only on the GB10.
Here, from no fixtures:

| Gate | Result |
|---|---|
| `make arch-fixtures` | 159 fixtures: 109 through `test-llama-archs`, 50 through `arch_fixtures_extra` |
| `make arch-model-diff` | **157 pass**, 0 fail, 0 unported, 2 `REF-ABORT` (`plm`, `rwkv6qwen2`, both at `n_seq_max 300`); T5 `gpu:17+3abort+1nondet cpu:18+3abort` |
| `make gemma4-assistant-diff` | dense and MoE, Metal and CPU: identical |

Unchanged from before the CUDA pickup, which is the expected result.

### Cross-builds

`zig build cli -Dcuda=false --release=fast` into a scratch prefix:

- **`aarch64-linux-gnu`** fails to link on one symbol, `arc4random`, from
  `sampler/core.zig`'s `getRngSeed`. Zig's default glibc for the target is
  older than 2.36, where `arc4random` arrived; the GB10 builds natively
  against its own 2.39 and never saw it. `aarch64-linux-gnu.2.39` links.
  The call is libc++'s Apple arm of `std::random_device` in any case: on
  Linux libc++ is `_LIBCPP_USING_DEV_RANDOM` (`__config:150`) and reads
  `/dev/urandom`, after `entropy()`'s `RNDGETENTCNT` ioctl decides whether
  `get_rng_seed` takes the device or the clock.
- **`x86_64-linux-gnu.2.39`** fails to compile at
  `repack/arm/q4_0_gemm.zig:519`, an `nzcv` clobber that does not exist on
  x86 -- the first of the NEON arms the CPU backend compiles on every
  target.

### Live lines per target

A script in the scratchpad ran `zig cc -E` / `zig c++ -E` over every CPU
translation unit with the build's defines, attributed each non-blank output
line to its source line through the linemarkers, and kept the set per
upstream file -- the method `CLAUDE.md` names for `arch/arm/quants.c`, as a
set rather than a count so two targets can be subtracted. Targets:
`aarch64-macos` at `baseline` and `apple_m3` (together, what is ported),
`aarch64-linux-gnu` at its baseline, and `x86_64-linux-gnu` at `x86_64`,
`x86_64_v2`, `_v3` and `_v4`. `PLAN.md` has the table.

What the subtraction cannot see, and why `PLAN.md` says the table
undercounts: `simd-mappings.h` contributes no live lines of its own -- its
`#define`s vanish in preprocessing -- so a `GGML_F32_VEC_FMA` line in
`ops.cpp` counts as already ported on x86 although it expands to AVX.

libllama measured the same way, every `src/llama*.cpp` and `unicode.cpp`
for `aarch64-macos` against `aarch64-linux-gnu`: Linux adds 32 lines of
`llama-mmap.cpp` (direct I/O, which `mmap.zig` ports) and one of
`llama-quant.cpp`. Nothing else.

### `GATE_CPU=generic` is Linux arm64's baseline

`zig cc -dM -E` for `aarch64-linux-gnu -mcpu=baseline` and for
`aarch64-macos -mcpu=generic` give the same `__ARM_*` set, apart from the
Apple-only `__ARM_NEON__` and `__ARM64_ARCH_8__` spellings: NEON, FMA,
`__ARM_FP16_FORMAT_IEEE`, and **no `DOTPROD`, no
`FP16_VECTOR_ARITHMETIC`**. So the existing `GATE_CPU` mechanism runs the
GB10's CPU configuration on the Mac, and `scripts/gate-cpu` checked both
libraries reported `dotprod=0 matmul_int8=0 sve=0 llamafile=1`.

| Gate | `generic` | `baseline`, same tree |
|---|---|---|
| `make ops-diff` | **86 of 632 differ** | 650 / 650 |
| `make repack-diff` | **26 of 60 differ** | -- |
| `make node-diff` (CPU) | **first divergence node 27, `SSM_CONV`**, threads 1 and 4; fused decodes from step 0 | -- |
| `make parity-port-cpu` | **2 of 6 prompts differ** (`Explain gravity in one sentence:` and `def fibonacci(n):`) | -- |

Both libraries were rebuilt at `baseline` afterwards; `ops-diff` passing
there is the check that they were.

Four classes, read from which ops differ and from the C's `#if`s:

- **The F16 arm.** Without `FP16_VECTOR_ARITHMETIC`, `simd-mappings.h:303`
  widens `f16` to `f32x4` and computes in `f32`, `GGML_F16_STEP` 16. The
  port has the `float16x8_t` arm only. f16 `mul_mat` and every f16-K/V
  `flash_attn_ext` case differ.
- **sgemm's guards.** `tinyBLAS_Q0_ARM` is `#if __ARM_FEATURE_DOTPROD`
  (sgemm.cpp:1214) and the f16 `tinyBLAS<8>` is behind
  `FP16_VECTOR_ARITHMETIC`; `generic` gets `tinyBLAS<4, float32x4_t>` over
  f16 A and f32 B. `sgemm.zig` dispatches as on the Mac. `q4_0` and `q8_0`
  `mul_mat` differ.
- **The `dotprod` `#else` arms.** `q4_1`, `q5_0`, `q5_1`, `q1_0` and
  `q2_0` differ too, which sgemm does not explain. The C's no-`dotprod`
  arms are integer (`ggml_vdotq_s32`'s fallback, `vmlal_s8` into
  `int16x8_t`), so the difference is likely in an epilogue; not yet
  localised. In `repack.cpp` every `dotprod` kernel falls to its
  `_generic` body at `generic` (`ggml_gemv_q4_K_8x4_q8_K` →
  repack.cpp:706) where the port runs the NEON one: that is the gross
  `q4_K`/`q5_K`/`q6_K` class in `repack-diff`. None of those is selected
  by the dispatcher at `generic`, on either side.
- **The vectorizer's cost model.** `ssm_conv`, `ssm_scan`,
  `conv_2d_direct` and `conv_3d_direct` are `f32` and touch no `#if`
  above. `ops.cpp` compiled at `-O3` for each CPU and disassembled:
  `ggml_compute_forward_ssm_conv`'s reduction loop has four `fmul.4s` at
  `apple_m1` and two at `generic`, the same one fused `fmadd` tail. Clang
  interleaves the strict in-order reduction twice as wide for the M1, so
  16 elements an iteration against 8, and a different remainder is fused.
  `strictLoopVectorized` encodes the M1's width.

### What the GB10 gates could not see

`context-diff`, `arch-model-diff` and `gemma4-assistant-diff` put
`libllama-ref.a` beside the port's libllama in one process **over the
port's `libggml.a`**: ggml is not a side of the comparison, so they are
blind to the CPU kernels by construction. The four `cuda-*` gates compare
CUDA backends. `ops-diff`, `repack-diff`, CPU `node-diff` and
`parity-port --cpu` never ran there. So the GB10's 157 / 159 said nothing
about Linux arm64 CPU numerics, and nothing claimed it did -- but nothing
said it did not either.

### Linux host arms the port does not have

From the live-line subtraction, then read:

- `threading.zig` ports `ggml_thread_apply_affinity` and
  `ggml_thread_apply_priority` from the `__APPLE__` arm (ggml-cpu.c:2579,
  2585) with macOS's `SCHED_OTHER = 1`, `SCHED_FIFO = 4` and its 8-byte
  `sched_param`. The Linux arm (2614, 2643) pins with
  `pthread_setaffinity_np` and maps `PRIO_LOW` to `SCHED_BATCH`; Linux's
  `SCHED_OTHER` and `SCHED_FIFO` are 0 and 1.
- NUMA -- `ggml_numa_init` (636), `set_numa_thread_affinity` (2148) and
  `g_state`'s `cpuset` -- is ported from the non-Linux `#else`.
- `ggml.c`'s Linux backtrace and `ggml-cpu.cpp`'s `/proc/cpuinfo`
  description were already ported.

### An x86 gate machine, looked for

- **Rosetta 2 is not installed.** A Zig-built `x86_64-macos` binary fails
  with "bad CPU type in executable"; `/Library/Apple/usr/libexec/oah` is
  empty and no Rosetta package is registered. macOS 27's own binaries have
  no x86 slice, so `arch -x86_64 /usr/bin/true` proves nothing either way.
- **Podman is installed**, with a libkrun VM (`podman-machine-default`).
  That is an aarch64 Linux guest; x86 in it would be QEMU user-mode.
- **`llama.cpp.zmake` refuses x86** at `build.zig:361` -- "the ggml source
  list is arm64 only" -- so there is no x86 reference library without
  either changing that repository or building one from `build/llamacpp.zig`.

Those three are `PLAN.md`'s open questions.

## Stage 5 step 2: the CPU path made exact at `generic`

Four changes, each chosen at comptime on `builtin.cpu` exactly as the C's
`#if` or the C compiler chooses. `ops-diff` at `GATE_CPU=generic` after
each: **86 → 54 → 8 → 0** of 632 -- so every class was seen failing before
it was seen passing, which is the negative test.

| change | where | `ops-diff` at `generic` after |
|---|---|---|
| the F16 arm without `fullfp16`, and sgemm's two guards | `vec.zig` (`dotF16Widened`), `ops/vecinline.zig` (`mad_f16`, `scale_f16`), `ops/sgemm.zig` | 54 |
| `ggml_vdotq_s32`'s own lane grouping | `quants/arm/neon.zig`, and every `ggml_vdotq_s32` site in `legacy.zig`, `k.zig`, `iq.zig` | 8 |
| the vectorizer's width as a property of the CPU model | `features.strict_loop`, `vec.vectorizedTail`, `ops/common.strictLoopVectorized` / `strictSplit`, `ops/ssm.zig` | 0 |
| the 19 `dotprod`-guarded repack kernels falling to `_generic` | `repack/arm/{q4_0,q4_0_gemm,nibble4,q4_k,q5_k,q6_k,q8_0}.zig` | -- (`repack-diff` 26 → 0 of 60) |

### Two new feature constants

`neon.has_dotprod` and `neon.has_fp16_va` are `__ARM_FEATURE_DOTPROD` and
`__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`, beside the existing `has_i8mm`.
Clang defines the second for `fullfp16`.

### The F16 arm, and sgemm

Without `FP16_VECTOR_ARITHMETIC`, `simd-mappings.h:303` defines the `F16`
macros over `float32x4_t`: load is `vcvt_f32_f16(vld1_f16)`, store
`vst1_f16(vcvt_f16_f32)`, FMA `vfmaq_f32`, `GGML_F16_STEP` 16, and the
reduce is `GGML_F32x4_REDUCE`, whose halving count is `GGML_F32_ARR` --
also 4. For `mad_f16` and `scale_f16` that makes the vector body and the
scalar tail the same function of each element (`f16(fma(x, v, y))`,
`f16(y * v)`), so at `generic` each is one loop. `ggml_vec_dot_f16`
accumulates in four `f32x4`.

sgemm's `tinyBLAS_Q0_ARM` is `#if __ARM_FEATURE_DOTPROD` (sgemm.cpp:4052,
4089) and its f16 `tinyBLAS<8>` is behind `FP16_VECTOR_ARITHMETIC`
(sgemm.cpp:3974); without them q8_0 and q4_0 return false and f16 takes
`tinyBLAS<4, float32x4_t>` over **f16 A and f32 B** (sgemm.cpp:3984), which
needs the third `load` specialisation the port had called dead. `TinyBLAS`
now takes `TA` and `TB` separately.

### `ggml_vdotq_s32` is not `vdotq_s32` without `dotprod`

`ggml-cpu-impl.h:310`: two `vmull_s8` halves, `vpaddlq_s16`, added. Lane
`k` gets elements `2k, 2k+1, 2k+8, 2k+9`; `sdot` gives `4k..4k+3`. Upstream
says so in a comment. The integer total is the same, so the K-quant and IQ
kernels -- which reduce lanes in integer before converting -- were never
affected; the legacy ones convert each lane to `f32` and multiply by a
scale first, and all seven differed. The port had one `dotq_s32` for both
spellings. Now `neon.ggml_vdotq_s32` is the C's macro, and `dotq_s32`
stays for the sites the C spells `vdotq_s32` inside `#if
__ARM_FEATURE_DOTPROD` -- `nvfp4`, `tq1_0` and `tq2_0`, whose `#else` arms
give the same per-lane integers and were never among the failures.

### The loop vectorizer reads the CPU model

The IR of `ggml_compute_forward_ssm_scan` at `-O3`, both CPUs, both
strict loops: `apple_m1` checks the trip count against 4 and then 16 and
the store-load distance against 64; `generic` against 8, and 32. The
disassembly of `ssm_conv` agrees -- four `fmul.4s` and a one-vector
epilogue for the M1, two and no epilogue at `generic`. LLVM's AArch64
tuning gives Apple cores an interleave factor of 4 and the generic model 2.
So the rounded/fused split is `t - t % G` with `G` 4 or 8, and the guard
is `trip >= G` with a footprint of `16 G` bytes.

`features.strict_loop` holds `{group, footprint}` per CPU model: `apple_*`
and `generic`. **Any other model is a compile error**, deliberately: these
sites are bit-exact or wrong, and a build for an unmeasured CPU -- the
GB10's `-Dcpu=native` would be one -- should say so rather than produce
bits that differ from the reference. To add a model: emit `ops.cpp`'s IR at
`-O3` for it, read the `icmp ult` constants in `ssm_scan`, and confirm with
`ops-diff` and `node-diff` at that `GATE_CPU`.

### Repack

19 exports in `repack/arm/` begin `if (comptime !neon.has_dotprod) return
…_generic(…)`, the C's trailing call after its `#if`. Unreachable through
the dispatcher at `generic` -- it selects no repack type without `dotprod`
-- but exported, and now equal to the reference's.

### The goldens have a `generic` set

The unit tests build at whatever `-Dcpu` says, and at `generic` 11 golden
tests failed: the goldens are the reference's bits **at the CPU they were
captured for**, and at `generic` the reference computes different bits.
`scripts/vecdot-golden` and `scripts/vec-golden` now take `GOLDEN_CPU`
(default `baseline`, which reproduces the committed files byte for byte);
`generic` writes `quants/arm/golden_generic.zig` and
`vec_golden_generic.zig`, 18 and 20 lines different. The contract-off
goldens of the `_generic` kernels, `quants/golden.zig`, came out identical
and have no second set. `features.golden_generic` picks the set; the Apple
cores share the baseline's, and `apple_m3` passes on it.

One test asserted the Apple arm outright -- `mad_f16` and `scale_f16`
rounding the first 32 elements in half precision -- and now expects the
`f32` result at `generic`. `zig build test` passes at `generic`,
`baseline` and `apple_m3`. Negative-tested at `generic`: forcing
`ggml_vdotq_s32` to `sdot`'s grouping fails the seven legacy kernels;
setting `generic`'s strict-loop group to 4 fails the float dot products'
short lengths.

### Still unmeasured off Darwin

`ops/common.sinCos` calls `__sincosf_stret` on Darwin because that is what
clang folds a `sinf`/`cosf` pair into there, and assumes `sincosf`
elsewhere. Rope and `timestep_embedding` depend on it. The Mac's
`GATE_CPU=generic` builds for Darwin, so it cannot see this; the GB10 can.
It is on the GB10 pickup list.

## Stage 5 step 3: the Linux host arms

Written beside the Apple arms, each chosen at comptime as the C's
preprocessor chooses. They compile for `aarch64-linux-gnu` at Zig's default
glibc and at 2.39, and nothing on the Mac runs them, so they are
**ungated** until the GB10 pickup runs.

- **`threading.zig`.** `ggml_thread_apply_affinity` and
  `ggml_thread_apply_priority` under `__linux__`: glibc's 1024-bit
  `cpu_set_t`, `pthread_setaffinity_np`, Linux's `SCHED_OTHER` 0, `FIFO` 1,
  `BATCH` 3 and its one-`int` `sched_param`. NUMA under `__gnu_linux__`:
  `ggml_numa_init`'s `/sys` walk, `getcpu` (glibc's wrapper from 2.34, the
  raw syscall before, as the C selects), the `numa_balancing` warning, and
  `set_`/`clear_numa_thread_affinity`. `CPU_ALLOC` becomes a full-size
  stack set sized with `CPU_ALLOC_SIZE`, since `GGML_NUMA_MAX_CPUS` bounds
  it. `stat` is `statx`, as in `llama/mmap.zig`.
- **The Apple arm's warning text was wrong**, and is fixed in passing: the
  port printed `warn: failed to set thread priority %d : %d`, the C prints
  `%d : %s (%d)` with `strerror`. Both arms print the C's now.
- **`get_rng_seed`.** libc++ is `_LIBCPP_USING_ARC4_RANDOM` on Apple and
  the BSDs and `_LIBCPP_USING_DEV_RANDOM` elsewhere. The second opens
  `/dev/urandom`, takes `entropy()` from the `RNDGETENTCNT` ioctl clamped
  to 32, and reads four bytes; an `entropy()` of 0 sends `get_rng_seed` to
  `system_clock`, which libc++ counts in microseconds. The function-local
  `static` is an atomic tri-state. A failure to open or read terminates,
  through `ggml_abort`, with libc++'s `std::system_error` text.
- **The glibc floor.** `arc4random` was the only symbol that kept the
  default `aarch64-linux-gnu` cross-build from linking. It now links, and
  the highest version it references is `GLIBC_2.29`.

`port-links` caught six of the new citations, all mine: two at the wrong
line (`g_state` is ggml-cpu.c:573, `ggml_get_numa_affinity` 622) and four
naming one symbol against two line numbers. On the Mac, after the change:
`zig build test`, `ops-diff` 650 / 650, CPU `node-diff`, `parity-cli` 6 / 6.

## Stage 5 step 4: `make validate` off macOS

- **The eight in-process diffs** -- `regex`, `unicode`, `vocab`, `arch`,
  `loader`, `batch`, `chat`, `api` -- linked the four Metal frameworks and
  read `xcrun`. They now source `scripts/platform-libs` (`platform_libs
  exe`), as the arch scripts did after the CUDA pickup. `libcxx-diff`
  assigned an unused `SDK` from `xcrun`, which on Linux printed an error
  and carried on; the line is gone.
- **`platform-libs` linked CUDA unconditionally on Linux**, so a CPU-only
  Linux machine could not link a harness. The CUDA libraries are now added
  only where `$CUDA_HOME/lib64/libcudart.so` exists.
- **`port-coverage --target <triple>`.** The script measured the macOS
  contract only (NOTES.md, "What cannot run here"): Metal and Accelerate
  defines, a Mach-O underscore stripped, `use_metal = true`. Read off the
  target now -- the host's by default -- an ELF target gets the
  reference's Linux define, `_GNU_SOURCE` (without which `ggml.c` has no
  `M_PI`), keeps names as written, and reports the Metal units as not part
  of it. Two things the first Linux run reported missing were not:
  `DW.ref.__gxx_personality_v0`, ELF's counterpart of the already-filtered
  `__clang_call_terminate`, and `ggml_abort`, `ggml_log_internal` and
  `ggml_format_name`, which `src/ggml/varargs.c` defines on aarch64 outside
  Apple and Windows because Zig 0.17 cannot -- so that object joins the
  measured side where `needsVarargsShim` would compile it. Cross-compiled
  from the Mac, `aarch64-linux-gnu` reads 100% on every unit; native is
  byte-identical to before.
- **`validate` swallowed a `sched-diff` failure.** Its line was
  `./scripts/sched-diff || { [ -f <ref> ] || echo skipped; }`: with the
  reference present and the diff failing, the `[ -f ]` succeeds and the
  group returns 0. Demonstrated with `false` in the script's place. It is
  now `if [ -f <ref> ]; then ./scripts/sched-diff; else echo skipped; fi`,
  `graph-diff`'s form, which exits 1 in the same case. Both diffs keep the
  Apple-clang CMake reference; on Linux they print that they were skipped.

## Decision 52: `llama.cpp.zmake` builds x86_64

Granted explicitly against Decision 18, and kept to what x86 needs. In
`llama.cpp.zmake/build.zig`, the two `arch/arm` entries left the shared
ggml source lists for an `archSources(arch)` pair -- `arch/arm` for
aarch64, `arch/x86/quants.c` and `arch/x86/repack.cpp` for x86_64, as
upstream's `ggml-cpu/CMakeLists.txt:240` lists them -- and the arm64-only
guard now admits x86_64 on Linux and still refuses everything else,
x86_64-macos included. CMake's `GGML_AVX*` definitions are not added:
measured, no source reads them but `arch/x86/cpu-feats.cpp`, which only
`GGML_BACKEND_DL` builds. `-Dcpu` sets the compiler's own `__AVX2__` and
friends, which are what the kernels test. Its Makefile gained
`linux-x86_64`, `X86_CPU` defaulting to `x86_64_v3` (Decision 53), and its
README and CLAUDE.md say so.

Checked by building, since nothing here runs x86: `x86_64-linux-gnu` at
`x86_64_v3` gives a `libggml.a` with 959 `vpmaddubsw` and no `zmm`
register, at `x86_64_v4` 1,675 and 9,410 -- the AVX2 and AVX-512 kernels
are in. `aarch64-linux-gnu` still builds, and on the Mac `ops-diff`
rebuilt the macOS reference and passed 650 / 650 against it.

Not changed, though it is wrong: zmake's CLAUDE.md and README say Zig's
aarch64 baseline "already implies `DOTPROD` and `FP16_VECTOR_ARITHMETIC` on
macOS and Linux". On Linux it is `generic`, which has neither ("Stage 5:
the measurement"). Correcting another repository's prose is outside the
grant.

## Stage 5 step 5: the SIMD layer

`src/ggml/cpu/simd.zig` is `simd-mappings.h`'s `GGML_SIMD` block. `F32`
is `GGML_F32_VEC` and `F16` is `GGML_F16_VEC`: `Vec`, `step`, `epr`,
`arr`, `zero`, `load`, `fma`, `add`, `reduce` and the rest, chosen at
comptime on `builtin.cpu.arch`. NEON's `F16` is two different things,
`GGML_F16x8` computing in half precision under
`__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` and `GGML_F32Cx4` over widened
halves without it, and `F16.half_precision` says which one this build has.
The vectorizer's strict-loop shape moved in beside them: `strict_loop`
from `features.zig`, `vectorizedTail` from `vec.zig` (now `strictDot`),
and `strictLoopVectorized` and `strictSplit` from `ops/common.zig`. Any
architecture but aarch64 is a compile error naming step 6.

**The plan undercounted, as plans here tend to.** It named five direct
`neon.zig` imports. There were six (`vec.zig`, `ops/common.zig`,
`vecinline.zig`, `recurrent.zig`, `ssm.zig`, `sgemm.zig`), and two more
files, `attention.zig` and `sort.zig`, had `GGML_F32_EPR` written as a
literal 4 -- which x86 makes 8 or 16 -- with no import to find them by.
They were found by grepping for `simd-mappings.h` citations, not imports.

What changed in the kernels:

- `ggml_vec_dot_f32`, `ggml_vec_dot_f16`, `rwkv_wkv7` and `ssm_scan`'s
  Mamba-2 body are written against `F32`/`F16` the way the C writes them
  against the macros: `[F32.arr]F32.Vec` accumulators, `F32.step`,
  `F32.reduce`. `ggml_vec_dot_f16`'s two arms, which step 2 had added as a
  second function (`dotF16Widened`), are one body again, as they are in
  `vec.cpp`.
- `mad_f16` and `scale_f16` ask `simd.F16.half_precision` and
  `simd.F16.step` rather than `neon.has_fp16_va` and a local 32.
- The elementwise kernels that collapse body and tail into one fused loop
  (`mad_f32`, `mad1_f32`, `conv_2d_dw`, `rwkv_wkv6`, `gla`) did not change:
  they are width-independent as long as `GGML_F32_VEC_FMA` fuses, which it
  does on NEON and on every x86 level Decision 53 targets (`__FMA__` is in
  v3).
- `vec.cpp`'s SiLU, softmax and `ggml_v_expf` arms and `sgemm.cpp`'s
  `tinyBLAS` are written in NEON intrinsics in the C, not through the
  macros, so they reach `neon.zig` through `simd.neon` and get their x86
  arms beside the NEON ones in step 6.

**The gate was the machine code itself.** A refactor meant to be a no-op
should compile to the same instructions, which is a stronger claim than
any number of passing diffs. The port's `libggml.a` was built
`--release=fast` at `baseline`, `apple_m3` and `generic` before and after,
and every object disassembled and compared. All 26 objects in all three
builds are identical except two functions: `ggml_vec_dot_f32` at all three
and `ggml_vec_dot_f16` at `baseline` and `apple_m3`. In each, the vector
loop's two pointer registers trade names, `x10` and `x11`, with every load,
`fmla` and reduction step unchanged and in the same order. That comes from
the new `ax[j]`/`ay[j]` arrays the C has and the old code did not.
`ggml_vec_dot_f16` at `generic` is byte-identical, though its source
changed most.

The first rename check was itself wrong: `sed s/x10/.../` also rewrote the
`x10` inside immediates like `#0x10`, and reported differences that were
not there. It was redone with word boundaries before the conclusion above
was drawn.

Also run: `zig build test` at all three CPUs (the new `simd.zig` tests
included, confirmed by the count moving under a filter) and `make validate`
(5,518 citations). Negative test: an ordered `@reduce` in `F32.reduce`
fails `simd.zig`'s own test and both float-dot golden tests.

## Preparing the GB10 pickup

The GB10 list was meant to be run, not developed, on the visit. Getting it
there found two real Linux differences and one bug on every platform.

**`-fmath-errno` is the Linux difference `GATE_CPU=generic` cannot show.**
`GATE_CPU=generic` gives the Mac Linux's *CPU features*, but it still
builds for Darwin, and clang's libm defaults differ by OS: Linux assumes a
libm call may set `errno`, Darwin does not. The `aarch64-linux-gnu`
reference, cross-built here and read with `nm` and `objdump`, shows the two
consequences, and a five-line C file at both settings on both OSes
confirms the cause:

- **No `sinf`/`cosf` fold.** Linux's `ops.o` imports `sinf` and `cosf` and
  no `sincosf`; Darwin's imports `__sincosf_stret` and neither. The port
  had assumed `sincosf` off Darwin. `ops/common.sinCos` now calls the two
  separately there.
- **No vectorizing around a libm call.** `ssm_scan`'s Mamba-1 loop calls
  `expf`. Darwin's `ssm_scan` at `generic` has twelve `expf` call sites,
  eight of them one per lane of the vectorized loop; Linux's has four, and
  six fewer `fmul.4s`. So on Linux the loop is scalar and every
  `sumf += state * C` fused, where the port applied Darwin's rounded groups.
  `simd.math_errno` now carries both.

The other Linux-only libm imports were checked and move no bit: `sqrtf`
beside an `fsqrt` for the `errno` path, `lroundf` where Darwin inlines
`fcvtas`.

Which of the two mattered was measured, natively, in a Linux arm64
container on this Mac -- the podman VM with the local `debian:stable-slim`,
glibc 2.41, both sides cross-built here. `ops-diff` passes there, 650 /
650 at `generic`. With both changes reverted it fails on exactly the four
Mamba-1 results whose `d_state` reaches the 8-wide group (16 and 10, at one
thread and three). **The `sinCos` change moved nothing on these inputs**:
glibc's `sincosf` agrees with its `sinf` and `cosf` here. It is kept
because it is what the reference calls, but it is a fidelity change, not an
observed fix.

**The port sent twelve of the C's direct `stderr` writes through ggml's
logger.** The C writes some warnings with `fprintf(stderr, ...)`, which a
log callback does not intercept: the NUMA "already initialized" and
`pthread_setaffinity_np` warnings, the affinity and priority failures,
`ggml_get_n_tasks`' "op not implemented", `ggml_compute_backward`'s
unsupported unary op, two of `unary-ops.cpp`'s unsupported types, and
`binbcast.cu`'s. The port used `impl.logError` for all of them -- the same
text on `stderr` under the default handler, and on the *callback* under
`llama_log_set`'s, which every llama.cpp tool installs. `unary_ops.zig`'s
doc comment even called the substitution equivalent. They now go through
`impl.fprintfStderr`. Found by building the harness below with a callback
installed, which is the only way to see it.

**`make host-arms-diff`** is the gate step 3 lacked. `harness/host_arms.c`
runs a probe op on every worker of a real threadpool and reads back the
thread's policy, priority and affinity from the OS; eighteen scenarios --
five priorities, CPU masks loose, strict, strict and paused, with low
priority, and invalid (CPU 511, loose and strict), every NUMA strategy and
a second init, and the default seed's source -- each in its own process,
because the C applies a pool's priority and affinity to the calling thread
and initializes NUMA once. stdout is compared exactly and stderr sorted,
since workers warn in whatever order they run; the affinity warning prints
the *address* of the mask, the C's own oddity, so that one hex value is
normalised. On macOS 18 / 18, covering the Apple arms. In the container
18 / 18, covering the Linux ones for real: `SCHED_BATCH` for low priority,
`EPERM` warnings for `SCHED_FIFO`, per-thread strict masks, `EINVAL` for
the invalid mask, the `/sys` NUMA walk and `getcpu` finding one node, a
`random_device` seed. Negative-tested 3/3: the Darwin `HIGH` priority 80 to
70 on the Mac, and on Linux `SCHED_BATCH` as `SCHED_IDLE` and the priority
warning back on the logger. The container has one NUMA node, as the GB10
does, so `set_numa_thread_affinity`'s body is still unreached.

**The gate scripts assumed macOS.** `ops-diff`, `repack-diff`, `node-diff`
and `parity-port` linked against the SDK's frameworks through `xcrun`, and
`scripts/gate-cpu` derived the expected CPU features with `-target
aarch64-macos`, where `baseline` has dotprod -- on the GB10, where
`baseline` is `generic`, every native run would have failed its CPU check
before comparing anything. All five take the link line from
`scripts/platform-libs` now, and `gate-cpu` asks the host. `repack-diff`
also needed `harness/repack_diff_stub.cpp`: the renamed reference
`traits.cpp` references the C++-linkage
`ggml_backend_cpu_get_extra_buffer_types()` from two functions the harness
never calls, which ld64 dead-strips and lld reports anyway. With it,
`repack-diff` passed in the container, 60 / 60.

## `sdot`

`neon.dotq_s32` calls LLVM's `llvm.aarch64.neon.sdot.v4i32.v16i8` when the
build has dotprod, as `mmlaq_s32` already called `smmla`, and keeps the
widening form as `dotq_s32_portable` for `generic` and for a test that
holds the two together over 100,000 random inputs and the wrap case. The
arithmetic is integer, so the swap is bit-identical, and it is everywhere at
once: 99 call sites in eleven files -- `quants/arm`, `repack/arm` and
sgemm's `tinyBLAS_Q0_ARM` -- reach it directly, through `ggml_vdotq_s32`
or through `dotq_laneq_s32`.

In the port's `libggml.a` the `sdot` count went from 337 to 1,234 at
`baseline` and 177 to 1,074 at `apple_m3` -- LLVM had already recognised
the widening pattern in some kernels -- and the object lost about 10,600
instructions at each. `generic` is byte-identical, as it should be. Every
exact CPU gate passed at `baseline` and `apple_m3`: `ops-diff` 650,
`repack-diff` 60, `node-diff` 2,750 nodes and 33 fused decodes at 1 and 4
threads, `parity-port --cpu` 6 / 6. `zig build test` passes at all three
CPUs, the new test skipping at `generic`.

What it bought, on the CPU device alone -- Qwen3.5 2B `Q4_K_M`, `llama-cli
-ngl 0`, three runs each, the generated text unchanged:

| threads | prompt t/s | generation t/s |
|---|---|---|
| 1, before | 7.0 | 4.9 |
| 1, after  | 55.5 | 28.1 |
| 8, before | 40.0 | 26.8 |
| 8, after  | 224 | 105 |

Against the C, `make perf-cpu` (five alternating rounds a side, 512 greedy
tokens, the same tokens decoded in every run, `-Dcpu=baseline`) at the
commit before and the tree after:

| port | pp tok/s, port / C | tg tok/s, port / C |
|---|---|---|
| `9fc4cf8`, portable `dotq_s32` | 28.1 / 232.3 = 0.12 | 17.2 / 82.0 = 0.21 |
| with `sdot` | 217.1 / 234.0 = 0.93 | 77.5 / 82.2 = 0.94 |

The remaining six to seven percent is not attributed yet.

`CLAUDE.md` had called the portable form's cost "none detectable". That was
measured on Metal, where the CPU kernels do not carry the model; on the CPU
device it was most of the time -- the port's CPU path ran at a fifth of the
C's speed, and nothing that checks bits could say so.

## Stage 7: the measurement

Decisions 54–56 put Vulkan on the GB10, against `llama.cpp.zmake -Dvulkan`,
starting now. Everything below was measured on this Mac from the pinned
source, with Khronos's header-only `Vulkan-Headers` and `SPIRV-Headers` at
SDK tag 1.4.363 fetched into scratch -- this Mac has no `glslc`, no Vulkan
headers and no Vulkan driver. `PLAN.md`, "Stage 7 — Vulkan", has the
numbers; this is how they were taken and what went wrong on the way.

**The contract** was read the Stage 4 way: `ggml-vulkan.cpp` compiled for
`aarch64-linux-gnu` (about 40 s, one object of 38 MB) and its symbols sorted.
Nine C exports -- `ggml-vulkan.h`'s eight and `vk_conv_block_sizes`, an
internal table missing its `static` -- and no C++-linkage symbol either
way. It calls only four Vulkan entry points directly; `vulkan.hpp`'s
dynamic dispatcher reaches the rest through `vkGetInstanceProcAddr`. So
anything linking it needs `-lvulkan`, and nothing else from the SDK.

**The shader inventory** came from upstream's generator, built natively and
run in its header pass, which lists every variant without compiling any:
1,440 with no optional extension, 1,980 with all seven. **The first
measurement said 1,656 and was wrong.** The seven `-D` flags were in a zsh
variable, which zsh does not word-split, so the "all" generator was built
with one malformed define -- and so was the first contract compile. It
surfaced only because the zmake build, which passes the flags as a list,
generated 1,980 variants and the counts would not reconcile; a direct loop
over the shader files, run with the same broken generator, then showed four
files compiling nothing. Both were redone in bash. The contract did not
change; the shader-symbol count did, and the one recorded is the second.
It is not the first time zsh's word splitting has bitten a measurement
here: Stage 5's did the same with a list of flags.

**What decides what compiles** is new with this backend: the toolchain.
Seven `glslc` probes set seven macros guarding 64 `#if` sites;
`VK_HEADER_VERSION` and the header's extension macros guard more; and most
variants are compiled with `glslc -O`, so the SPIR-V bytes are `glslc`'s.
Hence Decision 55: one `glslc` and one header set per comparison, and the
port embedding the reference generator's output rather than its own.

**The zmake build** (`-Dvulkan`, `make … VULKAN=1`) reproduces CMake's
three parts:

- The probes run at configure time through `sh -c`, because `glslc`
  reports a refused extension on stderr *and* exits non-zero, and Zig's
  `runFallible` returns stdout only on success. CMake's rule is kept
  exactly: supported unless the stderr says `extension not supported:
  <name>`. Locating `glslc` poisons the configuration cache, so a changed
  `glslc` is always re-asked.
- `vulkan-shaders-gen` is built for the host with the probe's macros and run
  once for the header and once per `.comp`. Each run is a cached step with
  every `.comp` and `.glsl` as inputs -- `glslc` resolves the includes, so
  Zig cannot see them otherwise.
- `ggml-vulkan.cpp` gets the headers with `-idirafter`, not `-I`: the
  default is `/usr/include`, which must not shadow the glibc headers Zig
  supplies. `libvulkan` is linked by `llama-cli`, not the archive: Zig
  resolves a system library even for a static archive, which failed the
  first attempt here, where there is none.

Checked with a stand-in `glslc` that accepts everything, writes a 20-byte
SPIR-V header per variant and a dependency file: every probe, both passes,
all 1,980 variants plus the dispatch tables defined in the archive,
`ggml_backend_vk_reg` defined and called by `ggml-backend-reg.cpp`,
natively and cross-built for `aarch64-linux-gnu`. Without `-Dvulkan` the
default archive is byte-identical to one built before the change. A second
build with the stand-in made only the seven probe calls, so the per-shader
steps cache. What the stand-in cannot check -- real SPIR-V, real probe
answers, a device -- is the GB10's pickup.

**`scripts/vk-repro`** is the first GB10 measurement: `node_dump` linked
against the Vulkan reference, run per node and fused several times on the
GPU -- the only GPU device that build registers -- and every run diffed
against the first, with the `glslc` and header versions printed beside the
answer. Its loop was exercised here against the Metal reference with the
Vulkan guard removed (2,750 nodes and 33 fused decodes identical across two
runs); the guard refuses a reference without the backend.
`scripts/platform-libs` now adds `-lvulkan` when an archive it is given
imports `vkGetInstanceProcAddr`, and only then.

## Stage 7: the GB10 pickup

**The machine had a driver and nothing to build with.** NVIDIA's ICD
(`libnvidia-gl-580`, 580.173.02) and the loader (`libvulkan1` 1.4.328) were
installed; `glslc`, the Vulkan and SPIR-V headers and `vulkaninfo` were
not, and `sudo` asks for a password. Ubuntu 24.04's `glslc` is shaderc
2023.8, which predates `GL_NV_cooperative_matrix2` and `GL_EXT_bfloat16`,
so taking it would have compiled NVIDIA's fastest arm out of the reference
-- the case the pickup warned about. So the toolchain is pinned instead,
Decision 59: **`scripts/vk-toolchain`** (`make vk-toolchain`) builds it into
`.tools/vulkan/` with no root, in about 45 s on 20 cores:

| piece | source |
|---|---|
| `include/vulkan`, `include/vk_video` | Khronos `Vulkan-Headers`, `vulkan-sdk-1.4.363.0` (`VK_HEADER_VERSION` 363) |
| `include/spirv` | Khronos `SPIRV-Headers`, `vulkan-sdk-1.4.363.0` |
| `bin/glslc` | google/shaderc v2026.4, whose `DEPS` pin glslang `e1b562a8` and SPIRV-Tools `ef96ed76` -- exactly the `vulkan-sdk-1.4.363.0` commits -- built against those tags |
| `lib/libvulkan.so` | a link to the system loader; `libvulkan-dev`, which carries the unversioned name, is not installed |
| `bin/vulkaninfo` | the distro's `vulkan-tools`, unpacked with `apt-get download` and `dpkg -x` |

1.4.363 is the tag the Mac's measurement was taken at, so the counts above
and the build here describe the same headers. `make vk-repro` passes the
toolchain to the zmake build itself; `scripts/platform-libs` adds
`-L.tools/vulkan/lib` beside `-lvulkan` where the link exists.

**The device**, by `vulkaninfo`: NVIDIA GB10, `deviceID` 0x2e12, API
1.4.312, integrated, with `VK_KHR_cooperative_matrix`,
`VK_NV_cooperative_matrix2`, `VK_KHR_shader_integer_dot_product`,
`VK_KHR_shader_bfloat16` and `VK_NV_cooperative_vector`. Mesa's llvmpipe is
enumerated too; ggml-vulkan skips CPU-type devices and reports one device.
ggml's own line: `uma: 1 | fp16: 1 | bf16: 1 | fp4: 0 | warp size: 32 |
shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2`.

**The probes**: all seven passed, and the generator declared 1,980
variants, the all-extensions count the Mac measured.

**A probe hole, found by a typo.** The first build was given a `-Dglslc`
path that did not exist (`llama.cpp.zmake` is a symlink to a sibling
directory, so `$PWD/..` was not this repository). Configuration succeeded
with all seven probes "passing": the probe counts anything but
`extension not supported: <name>` as support, as CMake does, and `sh: not
found` is not a refusal. Only the per-shader compiles failed. The zmake
build now runs `glslc --version` before probing and refuses one that does
not run. CMake cannot hit this, because `find_program` fails first.

**`make vk-repro` passes on all four of the GB10's arms**, three runs each,
2,750 nodes per node and 33 fused decode steps:

| arm | `matrix cores` / `int dot` | per node | fused |
|---|---|---|---|
| default | `NV_coopmat2` / 1 | identical | identical |
| `GGML_VK_DISABLE_COOPMAT2=1` | `KHR_coopmat` / 1 | identical | identical |
| `+ GGML_VK_DISABLE_COOPMAT=1` | none / 1 | identical | identical |
| `+ GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` | none / 0 | identical | identical |

The compute nodes all ran on `Vulkan0`; only the token embedding's
`GET_ROWS` reads the host buffer. The hashes are not degenerate: 1,630
distinct over 2,750 nodes. Negative-tested by making runs after the first
take `GGML_VK_DISABLE_COOPMAT2=1`: per node, the first divergent node is the
first `FLASH_ATTN_EXT` (node 216), and all 33 fused lines differ. So Vulkan,
like Metal and CUDA, is an exact oracle, and each arm is a distinct
reference that the gates can select by environment.

**`glslc -O` is not reproducible, on one shader.** Four complete
generations were compared, 1,980 variants each: the `make vk-repro` build,
and three fresh-cache builds (one plain, two under `MALLOC_PERTURB_`). 1,979
variants are byte-identical in all four. **`tri_f16` is not**: two bytes,
one `OpConstant f32` whose value is `0x7f800000` (+inf) in the first build
and `0x00000000` in the other three. Standalone, the same command gives +inf
200 times out of 200, and under `MALLOC_PERTURB_` = 0, 1, 85, 170 and 255 it
gives four different binaries. So `spirv-opt` reads uninitialized memory,
and its output follows heap state, which argv and environment also move.
The unoptimized module has the +inf constant. In the optimized one,
`FConvert` of it is stored by the `else` arm, `data_d[…] = D_TYPE(0)`. So
**0.0 is right and the +inf build is a miscompile**: that `tri_f16` writes
+inf over every masked element. It is dead code: `ggml_tri` asserts an f32
source (ggml.c:5309), so no graph built through the API reaches the f16
pipeline that `supports_op` and `pipeline_tri[dst->type == F16]` offer, and
upstream's `test-backend-ops` builds `TRI` at f32 only.

What it changes is Decision 55's "identical SPIR-V by construction".
That holds only if **both sides embed one generation's files**. Two runs of
the same generator with the same `glslc` are not the same reference, and a
gate that compared the port's embedded blobs with the reference archive's
would fail on `tri_f16` for no porting reason. Hence Decision 59's second
half. Not root-caused further; it is upstream SPIRV-Tools' bug, not ours.

## Stage 7 step 4: the build, the first gates, the registry

**This repository builds Vulkan now**, beside zmake: `-Dvulkan` on
`zig build`, the toolchain defaulting to `.tools/vulkan` (Decision 59),
the seven probes run as zmake runs them, and `ggml-backend-reg`'s Vulkan arm
in `backend_reg.zig`, behind `config.use_vulkan`, with the C++'s
`GGML_DISABLE_VULKAN` check. Three artifacts:

- **reference mode** (`-Dvulkan`): upstream's `ggml-vulkan.cpp` and the
  generated shader `.cpp` files compiled into `libggml.a`, under our Zig
  ggml and libllama.
- **`vk-ref`**: the same C++ with its nine exports renamed `ref_*`, as
  `libggml-vulkan-ref.a`, for in-process gates (`addCudaRef`'s method).
- **port mode** (`-Dvulkan-port`): `src/ggml/vulkan/` instead of the C++.
  The shaders reach it through `build/vulkan_shaders.zig`, which reads the
  **same generated `.cpp` files** the reference compiles and writes one
  51.7 MB blob plus a Zig index under the C names (`abs_f16_data`,
  `add_data[2][2][2]`, `arr_dmmv_*`): 1,980 variants and every dispatch
  table, no C++ in the port. `vulkanShaders` memoises the generation per
  build graph, so a build's reference objects and table are one
  generation's bytes.

**One generation per build graph, not per checkout.** The generator binary
built with `-Dcuda` (the Linux default) differs from the one built with
`-Dcuda=false`, 8,573,240 bytes against 8,592,816, so the two
configurations generate separately. Here the two generations agreed,
`tri_f16` included, but they need not. Every gate takes both sides from
one configuration (the in-process ones from one invocation;
`vk-node-diff`'s from two `-Dcuda=false` builds), so this does not reach a
comparison. Why the generator differs is not known yet.

**Our Zig ggml and libllama are exact under a Vulkan device**, measured
before any host code was ported: `scripts/vk-node-diff` with
`VK_REF_DIR=llama.cpp.zmake/zig-out/lib` against reference mode, so both
sides run upstream's `ggml-vulkan.cpp`. All four arms, 2,750 nodes per node
and 33 fused decodes, identical. Negative-tested through `VK_PORT_ENV`,
which adds environment to one side only: `GGML_VK_FORCE_MMVQ=1` fails both
passes, and `GGML_VK_DISABLE_FUSION=1` fails the fused pass alone. That is
the per-node pass's blindness to fusion, shown rather than assumed.

**The host port starts outside-in**, as CUDA's did:

- `common.zig` holds `vulkan.hpp`'s behaviour without the C++: the
  dynamic dispatcher as a struct of `PFN_`s loaded through
  `vkGetInstanceProcAddr`, a thrown `vk::SystemError` as an error union,
  and the `VK_INCOMPLETE` retry of every `enumerate*`.
- `instance.zig` holds `ggml_vk_instance_init`: device selection,
  `GGML_VK_VISIBLE_DEVICES` read as a `stringstream` reads it, the
  UUID/LUID dedup with its driver priorities, the architecture probe,
  `print_gpu_info`'s device line, and memory, type and PCI id.
- `backend.zig` holds the registry, the device vtable's descriptive answers
  and `offload_op`, and the nine exports, `vk_conv_block_sizes` included.

Everything else aborts naming the C++ it stands in for. `supports_op`
answers false, so no op reaches an unported path. `translate-c` handles
`vulkan_core.h` at SDK 363 cleanly, 21k lines. The header has every
extension the C++ carries fallback structs for, so the port needs none and
says so at comptime.

**`scripts/vk-backend-diff`** asks one side per process, port or ref,
because the instance initialises once per process and writes to the log
and to `stderr` while it does. It runs both sides under 13 environments:
the visible-device list (the GPU, llvmpipe, both orders, the first index
past the end, one far past it, a malformed list), the offload minimum, and
every switch in the device line. It diffs the whole transcript, log lines
and exit status included. All identical. Negative-tested 3/3, after closing
one escape: `>` for `>=` in the visible-device range check passed until
the scenario `GGML_VK_VISIBLE_DEVICES=2` was added. The GB10 has two
physical devices, and index 7 is out of range under either comparison.
That is the input-distribution lesson again, now for an environment.

## Stage 7 step 5: the logical device, the pipeline walk, the buffers

**The second layer is in**: `ggml_vk_get_device`, `ggml_vk_load_shaders`,
`ggml_vk_create_pipeline_func` and its three SPIR-V passes, buffers and
their memory-type choice, pinned host memory, both buffer types, command
buffers and submission, and the synchronous transfers. In files:

- `device.zig`: `vk_device_struct` with every pipeline slot,
  `ggml_vk_get_device` (the extension `else if` chain, both property and
  feature `pNext` chains, queue families, the coopmat and coopmat2 shape
  queries, `createDevice`, the per-vendor tile switches, the descriptor set
  layout), queues and command pools. `std::recursive_mutex` is a small
  recursive mutex over a plain `pthread_mutex_t`, because its attribute
  type and constant differ between C libraries.
- `shaders.zig`: the walk. The C++ pastes shader names together in
  macros (`NAMELC ## F16ACC ## _cm2_data`); the port pastes the same names
  at comptime and reads them from the generated table with `@field`, so a
  misspelt name is a compile error. Runs of macro invocations became row
  tables walked in the C++'s order. **The order matters**: a slot two calls
  name is claimed by the first, with that call's arguments.
- `pipeline.zig`: `vk_pipeline_struct`, the compile with its float-controls
  patch, the coopmat2 decode-vector strip and the Asahi BK-loop roll (the
  last two are unreachable here and kept), statistics and register counts.
- `push.zig`: the 46 push-constant structs, **generated** by
  `scripts/gen-vk-push` with a layout check. They live inside the `.cpp`,
  not a header, so the generator computes every size and offset and writes
  them twice: as comptime assertions in the Zig, and as `static_assert`s in
  `harness/vk_push_layout.cpp`, which `#include`s the real
  `ggml-vulkan.cpp`. `zig build vk-push-layout -Dvulkan` builds it, and
  building it is the check: 1,266 facts. Negative-tested by moving one
  offset, which the C++ compiler rejected.
- `buffer.zig` and `submit.zig`: `vk_buffer` as a hand-counted reference
  (the C++'s `shared_ptr` frees in its destructor), the memory-type
  fallbacks, the memory logger, pinned memory, the device and host buffer
  types, `vk_context`, submissions with timeline semaphores, and the 2-D
  read and write paths.

**`vk-backend-diff` now creates the device on both sides** and asks the
buffer types and buffers: name, alignment, max size, `alloc_size`,
`is_host`, `supports_buft` (also ported), and a buffer round-tripped through
whole, offset and 2-D `set`/`get`, `memset_tensor`, `clear` and
`cpy_tensor`, every byte hashed; the host buffer type, and a transfer whose
host side is pinned. Seven more environments for the memory switches
(`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM`, with and without sysmem fallback,
`PREFER_HOST_MEMORY`, the forced maximum buffer and allocation sizes, the
suballocation block). 20 scenarios, identical. Negative-tested: a pinned
read with its pitches swapped fails on llvmpipe, and a halved
suballocation block fails everywhere.

**One injection escaped, and it is a hole in this machine, not the gate.**
A wrong destination offset in the staging *write* path passed every
scenario. That path runs only when a device buffer is not host-visible,
and on the GB10 none is: its device-local-only memory types (0 and 1) take
no buffers, so even `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM` lands on a
mapped type, and llvmpipe's memory is all mapped. The staging *read* path
is reached, through llvmpipe, which is not UMA. A discrete GPU would close
it; until one is in reach the write path is ported and ungated.

**Then the backend object.** `context.zig` holds `ggml_backend_vk_context`,
`ggml_vk_init`, the compute and transfer contexts, `ggml_vk_synchronize`
with its fence spin, graph cleanup, descriptor-set pools and
`ggml_vk_dispatch_pipeline`. `iface.zig` holds `ggml_backend_vk_init`, the
async 2-D transfers and their staging fallback, `cpy_tensor_async`,
events, and `buffer_from_host_ptr`. `perf.zig` is the
`GGML_VK_PERF_LOGGER` table. The C++'s weak `vk_context_ref`s become plain
pointers nulled in graph cleanup, the one place their contexts die.
`graph_compute` and `graph_optimize` abort, and `supports_op` still answers
false.

`vk-backend-diff` now builds a backend on both sides and runs async sets
and gets (whole, offset, 2-D, through malloc'd and through pinned memory),
`cpy_tensor_async` from a device and from a pinned host tensor, an event
recorded, waited on and synchronized, and a host pointer imported, aligned
and misaligned. Five more environments: the transfer queue
(`GGML_VK_ASYNC_USE_TRANSFER_QUEUE`, on the GPU and llvmpipe), serialized
submissions, `GGML_VK_DISABLE_ASYNC`, and both loggers. 25 scenarios,
identical.

**Two findings from it.** The memory logger turned up a missing
`VK_LOG_MEMORY` in both `ensure_sync_staging_buffer`s; its handle and
pointer values are masked before the diff, since they move run to run.
And **the harness escaped once, the familiar way**. Copying a half-size
pinned tensor into a full-size one trips ggml's own layout assertion, and
both sides aborted at the same place. The diff read that as agreement, and
everything after the abort went unasked. The script now fails any
scenario whose reference exits non-zero. Negative-tested: an async copy
offset by four bytes fails everywhere. An async 2-D read with its pitches
swapped escaped until the reads went through pinned memory, the only way
an async read reaches its pinned branch.

## Stage 7 step 6: graph compute, the first ops, `vk-ops-diff`

**The port computes graphs now.** `graph.zig` holds
`ggml_backend_vk_graph_compute`: the flop- and node-count batching, the
almost-ready fence, the perf logger's query pool, and every fusion test in
the C++'s order -- multi-add, the matmul and matmul_id fusions,
rms_norm+mul(+rope(+set_rows)), ssm_conv, rope+set_rows, snake, the five
topk_moe patterns and their trailing scale -- with the overlap check that
turns a fusion off. `ggml_can_fuse_subgraph` and `ggml_check_edges` moved
from `cuda/fusion.zig` to `graph.zig`, where both backends use them.
`nodes.zig` holds `ggml_vk_build_graph`'s synchronisation (a barrier when a
node overlaps an unsynchronised one), `ggml_vk_compute_forward` and
`ggml_vk_preallocate_buffers`. `op_pipeline.zig` is
`ggml_vk_op_get_pipeline` and the copy pipelines; `ops.zig` holds
`ggml_vk_op_f32` with the per-struct fastdiv and misalignment
specialisations, and the node switch; `ops_basic.zig` the 40 ops that are
one `op_f32` call, plus multi-add and the snake fusion; `supports.zig` the
reference's `supports_op`, which the device answers AND-ed with
`ops.ported`. `push.zig`'s derived structs became distinct types, because
`init_pushconst_tensor_offsets` overloads on concat's.

**`vk-ops-diff`** is `cuda-ops-diff` for Vulkan: upstream's
`test-backend-ops` with the reference Vulkan backend where the CPU goes,
every node compared on bits, and every `supports_op` question asked of both.
**2,136 cases bit-identical, none differing**; 15,937 questions refused as not
ported. `VK_DIFF_SELF=1` passes too. Negative-tested 3/3: a swapped
broadcast dimension (MUL), clamp's upper bound (CLAMP), roll's packed shifts
(ROLL).

**Three things the CUDA harness got away with, fixed for this one.**

- Its NONDET retry, and `VK_DIFF_SELF`, run a reference backend on the
  test's own graph, whose tensors live in the port's buffers. CUDA's port
  mirrors the reference's buffer-context layout; Vulkan's does not, so the
  reference read the port's context as its own and died of `SIGBUS`. A
  stand-in reference now gets its own copy of the graph.
- `ggml_backend_is_vk` matches on the GUID, which port and reference share,
  so it cannot say which backend is the port; the harness passes that
  explicitly.
- **That copy hid a real difference, once.** A plain
  `ggml_backend_graph_copy` has no use counts and cannot fuse. With it, the
  ten `SNAKE_FUSE` cases -- the reference fusing the chain, the port not yet
  -- re-ran unfused against fused, the reference "disagreed with itself", and
  all ten were excused as `NONDET`. Re-expanding the copy as backend 2's
  already was made them fail, which they should have: the port lacked the
  snake fusion while every one of its five ops was ported. It has it now.
  **Porting the ops of a fused pattern ports the fusion too**, or the port
  computes the chain unfused and differs.

## Stage 7 step 7: every op, `graph_optimize`, the decode exact

**Every op `supports_op` answers yes to is ported.** `ops.ported` no longer
refuses anything: rms_norm and its rope and set_rows fusions, softmax,
RoPE, the sorts and top-k, the router (`topk_moe`, five patterns), the
sums and scans, the recurrent and state-space kernels, the optimizer
steps (`ops_reduce.zig`, `ops_more.zig`); the convolution family
(`ops_conv.zig`); matrix multiplication -- the pipeline getters, tile and
split-k choice, the matrix-matrix path with its dequantize, contiguous
and q8_1 staging, the three vector kernels, FWHT (`ops_matmul.zig`);
MUL_MAT_ID (`ops_matmul_id.zig`); flash attention with its tuning, split-KV
reduction, mask prepass and K/V dequantized into scratch
(`ops_flash_attn.zig`); and `ggml_vk_graph_optimize` (`graph.zig`).

`vk-ops-diff`, coopmat2 arm: every family bit-identical -- among them
MUL_MAT 1,009 cases, MUL_MAT_ID 872, MUL_MAT_VEC_FUSION 432, FLASH_ATTN_EXT
5,150, ROPE 466, IM2COL_3D 2,051, CONV_2D 1,962. `vk-node-diff`: all four
arms, 2,750 nodes per node and the fused decode, identical; negative-tested
2/2 against the port (fusion off, reorder off, port side only).

**`graph_optimize` had a bug no op-level gate could see, and `vk-node-diff`
found it.** `test-backend-ops` computes graphs directly and never reorders
them. The port called `ggml_vk_is_empty`, which counts a zero-element
tensor as empty; the reorder's own `is_empty` lambda looks at the op alone.
A single-token decode of Qwen3.5 has a zero-row `GET_ROWS`, which the port
then moved with the view nodes: the prompt matched and every later step
diverged, on every arm. With `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` on both
sides the whole decode was already exact, which isolated it in one run.

**`vk-backend-diff` now asks `graph_optimize` directly**: both backends
reorder the same graphs -- two decoder layers with QK-norm, a MoE router
with expert matmuls, a snake chain, ssm_conv + add + silu, four seeded
random DAGs -- and the new order is printed as original indices. The
reorder moves 26 to 84 nodes of each, so agreement is not vacuous. Its
first version escaped three of six injections, each a shape it lacked: no
rope sat far from its MUL, so shrinking the search window changed nothing;
no independent node waited behind a snake chain, so not keeping it whole
changed nothing; no node read the cache through its own view, so dropping
the implicit view dependency changed nothing. And it had no zero-element
node, which is why it passed the bug above. With those four shapes added:
7/7, the bug included.

**Three of `vk-ops-diff`'s holes, closed with extra cases.**

- **A misaligned set_rows destination.** Upstream's ROPE_SET_ROWS and
  RMS_NORM_MUL_ROPE write a fresh tensor at offset 0, so the fused
  dispatch's rounded-down binding and `d_offset` never ran; zeroing
  `d_offset` passed 144/144. The case that closes it has to use a *leaf*
  placed off alignment: a VIEW node there overlaps set_rows' output and
  the overlap check turns the fusion off on both sides. ggml-alloc never
  places a leaf off alignment, so no model graph reaches this arm.
- **Matrix-matrix shapes.** Upstream keeps K small and M, N small, so
  split-k never split and coopmat2's large-tile choice was never made;
  changing either passed. Cases at K = 4096 and 8192, and 1024 x 1024.
- **The second fused bias in `src[0]`.** The C++ takes it from whichever
  operand is not the first ADD; upstream always puts it in `src[1]`.

**A harness that cannot tell a failed build from a passing gate.** Two
injections in this step did not compile, and the run tested the previous
library and passed. The injection helper now refuses to test when the
build fails. One injection (p021's z workgroups) hung the run instead of
failing it: the run normally takes 1m47 and was killed at 17 minutes.

## Stage 7 step 8: `vk-parity-cli` and the swap

**`vk-parity-cli`** is `cuda-parity-cli` for Vulkan: our `llama-cli` against
`harness/raw_completion.c` linked to upstream's libllama and upstream's
Vulkan host layer (`zig-out/vk-upstream`, `-Dvulkan-port=false`), greedy,
six prompts. **6/6 identical.** Negative-tested 3/3: RoPE rotating half
its dimensions (1 of 6 prompts changed -- token parity is coarse, as
`parity-port` measured), flash attention's scale doubled (5 of 6), and a
CLI built with upstream's host layer, which the script refuses by `nm`.

**Its first version checked the wrong thing for "ran on the GPU".** Each
side's log had to mention `Vulkan0`, and every prompt failed: our CLI
installs a quiet log callback (`cli/main.zig`), so libllama's device lines
never reach stderr. The check is now the Vulkan backend's own perf logger,
which writes its timings with a direct `fprintf(stderr, ...)` and only from
the Vulkan graph compute -- in one separate run per side, so the compared
runs are the binaries' defaults. Without it, a side that fell back to the
CPU could agree on tokens and pass while comparing nothing.

**The swap**: `-Dvulkan-port` defaults to true, so `-Dvulkan` builds
`src/ggml/vulkan/`; `-Dvulkan-port=false` builds upstream's
`ggml-vulkan.cpp` for the gates. `supports_op` is the C++'s predicate
alone: the `ops.ported` filter went, after checking it was a superset of
`ggml_vk_build_graph`'s cases (the five view ops beyond them never reach
dispatch), so no answer changed. `vk-ops-diff` now fails on a refusal the
reference does not make, where before the swap it only counted one.
`make vk-builds` names `-Dvulkan-port=false` for the upstream side
explicitly; every gate's Requires lines say which side they build.

**After the swap, re-run on the default builds**: `vk-backend-diff` 25/25
environments, `vk-node-diff` identical on all four arms, `vk-ops-diff`
17,015/17,015 on the coopmat2 arm with `supports_op` agreeing both ways,
`vk-parity-cli` 6/6. On the KHR coopmat arm every family is bit-identical
but FLASH_ATTN_EXT, which **stalls -- and the reference stalls on its own**:
with `VK_DIFF_SELF=1` (reference against reference) the run stops at the
same case, `hsk=256, kv=16384, type_K=q8_0`, after the same 5,135 OKs (96 s
in, against the port's 131 s). Each such case run alone finishes in a
second on both sides, so it is state built up over the family, in
upstream's host layer or the driver, not a port difference. The KHR arm's
FA cases before it are bit-identical; the rest are unexamined.

## Stage 5 step 6: x86 begins

**QEMU stands in for the laptop at AVX2.** The podman VM registers QEMU
10.1's user-mode `qemu-x86_64-static` through `binfmt_misc`, so x86 binaries
run in an amd64 container (`debian:stable-slim-amd64`, pulled for this and
tagged apart from the arm64 image). The x86-64-v3 reference ran `ops_dump`
twice there with identical output, 113 s a pass at three threads. QEMU's
TCG computes IEEE arithmetic in software, exactly, so the one way it could
disagree with silicon is an instruction whose result is the hardware's own
approximation -- `rcpps`, `rsqrtps` -- and neither the x86 source nor the
compiled reference uses one. QEMU does not implement AVX-512, so v4 still
needs hardware.

**The first x86 build compiled and was wrong.** The ARM kernels are portable
`@Vector` code and the dotprod guards from step 2 route them to their
generic bodies, so with `simd.zig`'s arm in place the whole library compiled
for x86 and computed ARM's arithmetic where the reference runs `arch/x86`.
It segfaulted under QEMU before printing anything. So the arch dispatch came
first: `quants/` and `repack/` choose `arm/` or `x86/`, and every x86
kernel not yet ported is a **compile error** naming itself.
`-Dx86-incomplete`, for development only, stands its generic port in
instead -- the gates then run, and every stand-in fails visibly. ARM's eight
`arch-fallback.h` renames moved from the generic repack files into
`repack/arm/fallback.zig`, since x86 defines some of the same names itself;
ARM's 1,223 exports are unchanged.

**The inventory of everything else was measured, not remembered.** Every CPU
translation unit was preprocessed for `aarch64` and for `x86_64_v3`, split
into functions, and the bodies diffed with the one spelling that differs
harmlessly -- `GGML_CPU_FP16_TO_FP32`'s table lookup against NEON's cast --
normalised. 42 functions differ. Each was read in both forms:

- **Width only, element-wise or through `simd.F32`**: `ops.cpp`'s
  `conv_2d_dw`, `fwht`, `gla`, `wkv6`, `wkv7`, `ssm_scan`, `simd_gemm`,
  flash attention's `f32_epr`; `vec.h`'s `add_f32`, `mad_f32`, `mad1_f32`,
  `scale_f32`, `mad_f32_unroll`. Nothing to change beyond step 5.
- **A real x86 arm, ported**: the eleven feature predicates; the F16C row
  conversions; `vec.cpp`'s SiLU, SwiGLU, softmax and `cvar` (8 lanes, and a
  horizontal sum `(h0+h2)+(h1+h3)` where NEON's is `(v0+v1)+(v2+v3)`) and
  `dot_bf16` (unfused, its own reduction); `vec.h`'s `mad_f16`/`scale_f16`;
  `sgemm`'s `tinyBLAS<8, __m256>`, with `VECTOR_REGISTERS == 16`'s tiles.
  The AVX2 `ggml_v_expf` turned out to be the NEON one term for term, so
  `vec.zig` writes it once over the arm's width.
- **NaN payloads.** F16C's `vcvtps2ph` keeps a NaN's payload where ggml's
  software narrowing does not, and `vcvtph2ps` quiets a signalling NaN where
  the lookup table does not, so the C's vector bodies and scalar tails
  disagree on NaNs, and the port calls the instructions -- LLVM's
  `vcvtps2ph` intrinsic -- where the C's vector code does. (NEON's
  `GGML_F32Cx4` store at `generic` has the same NaN-only difference from its
  tail and the port does not reproduce it; recorded, not changed.)
- **Unused**: `ggml_vec_dot_f16_unroll` has no caller upstream.

**Three toolchain facts, each found by the build failing:**

- **The `getcpu` syscall number is per architecture** (168 on aarch64, 309
  on x86_64) and the C's old-glibc path hard-codes neither; the port's
  `std.os.linux.getcpu` was already right.
- **Zig 0.17 compiles x86_64 Debug with its own backend**, which emits an
  LLVM intrinsic called by name as an undefined external symbol. The ggml
  library and its tests now set `use_llvm`.
- **`va_list` is a pointer on x86_64 SysV.** It is a one-element array of
  `__va_list_tag` there, so `vsnprintf` takes a pointer; the port declared
  it by value, which is right on AArch64 and crashed `ggml_format_name` on
  the first x86 graph. `impl.CVaList` and `impl.cVaList` now give each ABI
  what it expects. Found with a `SIGSEGV` handler printing `backtrace()`,
  since the container has no debugger.

**Where it stands**, `ops-diff` under QEMU at x86-64-v3 with
`-Dx86-incomplete`: **492 of 650 results identical**, and every one of the
158 that differ is a quantized `mul_mat` or a flash attention over `Q8_0`
-- the paths the 39 stand-ins feed. None of the non-quantized ones differ.
`make x86-qemu-ops-diff X86_INCOMPLETE=1` (`scripts/x86-qemu-ops-diff`)
reproduces the whole run: both cross-builds into `zig-out/x86-qemu/`, the
container, the comparison.

## Stage 5: Windows builds (Decision 57)

**Asked for in so many words.** The x86 plan had been a Linux laptop; the
machine that turned up was a 2015 MacBook Pro running Windows 11 under Boot
Camp, and the answer to "can we gate AVX on it" was that Windows was always
meant to be a target. Its firmware leaves VT-x off on a cold boot, so WSL2
would not have started anyway.

**The measurement, as Stage 5's.** Every ggml and libllama translation unit
was preprocessed for `x86_64-linux-gnu` and `x86_64-windows-gnu` at
x86-64-v3 with each platform's zmake defines, every live line attributed
through the linemarkers, and the two sets subtracted per upstream file: 400
Windows-only lines in six files, 578 Linux-only. `PLAN.md`, "Windows
x86_64", has the table. Everything else in ggml and libllama is
platform-neutral C and C++ -- which is not the same as meaning the same
thing, and four places do not: `std::mutex` is an `SRWLOCK`,
`std::random_device` reports zero entropy, `wchar_t` is 16 bits, and
`stdout`/`stderr` are function calls. Each was found by reading what the
line compiles to, and each is ported.

**The order the build failed in**, which is the order the work went:

1. **The reference**: zmake had no Windows platform. Adding one compiled
   everything but `tools/mtmd/mtmd-audio.cpp`, which lost `M_PI` to
   MinGW's strict-ANSI guard. CMake passes `-std=gnu11`/`gnu++17` -- it
   leaves `C_EXTENSIONS` on -- and zmake's strict dialects never showed the
   difference on Apple or glibc. The Windows platform takes CMake's
   dialects; the others keep the ones their gates were calibrated against.
2. **`std.lang.VaList`** is a compile error on x86_64 Windows under LLVM, as
   on aarch64 Linux; `varargs.c` covers it, and `impl.zig_varargs`,
   `impl.CVaList` and `needsVarargsShim` learned the target.
3. **pthreads**: `std.c.pthread_mutex_t` is `void` on Windows. The
   threadpool's lock is `ggml_mutex_t` and now has both arms -- including
   `ggml_mutex_lock_shared`, which the port had folded into the plain lock
   because they are one call on pthreads, and which is a reader's hold on an
   SRW lock. The seven `std::mutex` translations became `impl.CxxMutex`.
4. **LLVM aborted**: "Intrinsic has incorrect argument type!" on
   `llvm.x86.vcvtps2ph.256`. The Win64 C convention passes a 256-bit (and a
   128-bit) vector by reference, so an intrinsic declared `extern fn` with
   the default `.c` convention reached LLVM with a pointer. `.x86_64_sysv`
   passes it in a register, as the intrinsic takes it.
5. **libllama**: `mmap.zig`'s three `_WIN32` arms, the sampler seed, the
   unicode lock.
6. **The link**: `sysconf` and a `stderr` symbol, from the CPU device's
   memory query and five copies of a `stderr` accessor. The device's
   registry and `GlobalMemoryStatusEx` arms were ported, and the accessors
   became one, `impl.cStderr`.

**What it builds.** `zig build reference` and `zig build cli` for
`x86_64-windows-gnu -Dcpu=x86_64_v3 -Dx86-incomplete`: `ggml.lib`,
`llama.lib`, `llama-ref.lib` and a 6.7 MB `llama-cli.exe`. macOS, Linux
arm64 and Linux x86_64 still build. `scripts/port-links` read the MinGW header
references in `win32.zig` as citations of upstream files; they are written
as prose now, and every new citation resolves.

**First run on Windows.** Our `llama-cli.exe` ran on the laptop and
generated from Qwen3.5 2B `Q4_K_M`. That exercised the Win32 arms a decode
reaches: `CreateFileMapping`/`MapViewOfFile`, the SRW-lock thread pool, the
QPC timer, the UCRT streams. Model loading and the pool are covered; most
of the other arms have not been measured. Against the zmake reference, with
both sides cross-built here and pushed:

- `raw_completion.c`, "The capital of France is", 24 tokens: identical.
- The six `parity-port` prompts at 48 tokens: 4 / 6 identical.
- `node_dump.c`, CPU, at 1 and 4 threads: 2,750 nodes on both sides, the
  first 13 identical. The first divergence is node 13, a `MUL_MAT` over
  `q5_K` weights. `ggml_vec_dot_q5_K_q8_K` is one of step 6's pending
  kernels, which `-Dx86-incomplete` stands in with the generic one.

So nothing has diverged that is not a stand-in. These become exact gates
once step 6 is done. One trap: `push --run` and `tty` stream the console
through ConPTY, which re-renders the screen and draws blank lines as cursor
positioning. Text read back that way loses them, which at first looked like
the CLI dropping `\n\n`. A driver that writes its output to a file avoids
it.

## Stage 5 step 6: the AVX2 kernels

**Every `arch/x86` kernel is ported, and the CPU path is bit-exact on the
Windows laptop.** That is `arch/x86/quants.c`'s 27 exports, `arch/x86/repack.cpp`'s
12 and `sgemm.cpp`'s `tinyBLAS_Q0_AVX`, over `quants/x86/avx2.zig`, an
intrinsic layer named after Intel's intrinsics. Where an instruction's lanes
are ordinary, it is `@Vector` arithmetic. Where they are not, it calls LLVM's
x86 intrinsic by name, under the SysV convention as `cpu/convert.zig`
explains: `pmaddubsw` saturates, `pshufb` stays inside 128-bit halves,
`psign` zeroes, `phadd` pairs neighbours, and the packs saturate.
`-Dx86-incomplete` and its stand-ins are gone, and so is
`scripts/x86-qemu-ops-diff`: the gates run on the laptop.

**The order came from `node-diff`.** At `-Dx86-incomplete` the first
divergent node of the Qwen3.5 decode was a `q5_K` `MUL_MAT`, so `q5_K` went
first, then the other K-quants Q4_K_M reaches, then the rest.

### The gates, all on the laptop

- **`x86-quants-diff`**: 28 kernels, 2,400 random trials each over eight
  lengths, on bits. The reference translation units are compiled with every
  export renamed to `ref_*`, the `repack-diff` technique, and linked into one
  process with the port. **The apparatus was checked before anything was
  trusted**: all 27 renamed functions disassemble identically to the
  `quants.o` in zmake's own Windows `ggml.lib`. The `q8` operand runs three
  modes: the full byte range with -128, the quantizers' [-127, 127], and
  all-extreme values where saturation would show.
- **`x86-repack-diff`**: `harness/repack_diff.zig` unchanged in what it
  calls, reference `repack.cpp` + `arch/x86/repack.cpp` + `traits.cpp`.
  68 checks.
- **`x86-ops-diff`**: `ops_dump` through `harness/win_ops_dump.c`, the
  reference writing quantized blocks and the port replaying them, at 1 and
  3 threads. 712 results, about 90 s where QEMU took two minutes a pass. It
  is `tinyBLAS_Q0_AVX`'s oracle.
- **`x86-node-diff`**: the Qwen3.5 2B `Q4_K_M` CPU decode, which the
  machine holds on its Desktop. 2,750 nodes at 1 and 4 threads, plus the
  fused pass, which compares full logits.

The four together take 2m17s (`make x86-gates`). `scripts/win-run` is the
shared step: push, run, and clean the ConPTY stream. The comparisons happen
on the laptop through `harness/win_lines_diff.c`, since `push` brings no
files back. The six `parity-port` prompts are 6 / 6 at 64 tokens, where the
stand-ins gave 4 / 6.

### What they caught

- **A hole in `repack_diff.zig`**: it never called `ggml_cpu_init`. On ARM
  `GGML_CPU_FP16_TO_FP32` is a cast; on x86 it is a table, and the table
  was all zeros. Every scale through it was zero on both sides, and the
  comparison passed by computing nothing: the generic stood in for the
  AVX2 `gemv_q4_0_8x8_q8_0` and "passed". Found when the ported gemm
  returned all zeros, as the stand-in had.
- **A second hole in the same harness**: its `quantize_mat` inputs never
  reached a rounding tie. The AVX2 quantizers round ties to even and the
  generic rounds them away, so the generic `quantize_mat_q8_0_4x8` passed.
  It now runs a tie pattern and an all-zero block as well: 68 checks, up
  from 60. Both holes are closed for ARM too, and `make repack-diff` passes
  there.
- **The wrong `#if` arm**: `GGML_F32Cx8_LOAD` has an F16C arm,
  `_mm256_cvtph_ps`, and a table-driven `#else`. The `#else` was ported
  first. x86-64-v3 has F16C, and it has BMI2 too, which is why `iq1_s` and
  `iq1_m` take their `_pdep_u64` arms.
- **A live generic body**: `arch-fallback.h:86` makes `quants.c`'s
  generic `q2_0` the real x86 kernel. clang ships it with both scalar
  statements fused, but the shared port is unfused to match goldens
  captured with contraction off, where the generic is dead. `x86-ops-diff`
  found it as the one op of 356 still differing. `quants/x86/legacy.zig`
  now exports a fused copy.

**Faithful oddities kept**: `q3_K`'s AVX2 arm never advances `is`; the
gemvs store at `y * nr`, not `y * bs`; the `q4_K`/`q2_K` gemvs subtract an
unpermuted minima accumulator from a permuted one; the `q8_K` quantizer
keeps the sign of the largest element through a lane mask.

**The fusions were read from the reference's disassembly**: `q5_K`'s
`summs += dmin * x` is `vfnmadd231ss`, `q4_1`/`q5_1`'s `summs += m * s` and
every odd-block tail are fused, the ternary kernels' `add(mul())` is not,
and the `iq1` finals are fused.

### Injected faults

| Fault | Gate | Result |
|---|---|---|
| `q5_K` `summs` unfused | x86-quants-diff | 141 / 2400 differ |
| `q5_K` ordered reduction for `hsum_float_8` | x86-quants-diff | 1161 / 2400 |
| `quantize_mat_q8_0_4x8` rounding ties away | x86-repack-diff | ties pattern, 32 bytes |
| LUT gemm: wrong row-scale lane | x86-repack-diff | all three formats' gemms |
| LUT gemv: column scales in the gemm's order | x86-repack-diff | all three formats' gemvs |
| `q2_K` gemm: scale masks swapped | x86-repack-diff | gemm at 4 and 20 rows |
| `q4_K` gemv: wrong bsums lane for the mins | x86-repack-diff | 16 / 16 lanes |
| `q2_0` unfused | x86-quants-diff | 588 / 2400 |
| `tinyBLAS_Q0_AVX` `gemm4xN` unfused | x86-ops-diff | 9 results per pass |

### Speed, which the gates cannot see

**Bit-exact was not fast.** With every gate green, our `llama-cli.exe`
generated at 3.3 t/s against the reference's 14.5. A timing harness on the
laptop, ours and the reference's kernels in one process, named two causes:

- **Loads written as an array `@bitCast` to a vector** compiled to 32
  `movzbl`/`vpinsrb` per 32-byte load. The K-quants ran 10-15x slower than
  the C's. Loading through a vector pointer is one `vmovdqu`.
- **Shifts by a run-time count** -- `q3_K` and `q5_K` shift by `bit++` --
  went through a branch and a splatted 4-bit count, 2-3x slower. LLVM's
  `psrli`/`pslli` take the count as a scalar and apply Intel's rule (past
  the width is zero) in the one instruction.

After both: generation 14.3 against 14.5 t/s, prompt about 0.9x. Every
kernel Q4_K_M uses is at the C's speed. `tq1_0` (1.6x), `iq1_m` (1.4x)
and the generic `q2_0` (1.6x) are still slower, and exact.

---

---

## The GB10 pickup, run

`PLAN.md`'s "GB10 pickup" ran natively on the GB10 (Ubuntu 24.04, glibc
2.39), at its baseline CPU -- `generic`: no dotprod, no fullfp16. The
checkout there is a sibling of `llama.cpp.zmake/`, not its parent, so a
symlink `llama.cpp.zmake -> ../llama.cpp.zmake` (gitignored) makes the
scripts' relative paths hold.

| Gate | Result on the GB10 |
|---|---|
| `make repack-diff` | 68 / 68 |
| `make host-arms-diff` | 18 / 18, on real hardware: `SCHED_BATCH`, the `EPERM` warnings, strict masks, the `/sys` NUMA walk, a `random_device` seed |
| `make node-diff` (CPU) | 2,750 nodes at 1 and 4 threads, and 33 fused decodes at each |
| `make parity-port-cpu` | 6 / 6 |
| `make ops-diff` | **failed** on `log`, `sigmoid`, `exp`, `softplus` -- then 650 / 650 after the fix below |
| `make validate` | **failed** in `vocab-diff`, 780 of 55,010,074 -- then 55,016,050 identical after the fix below. Past it, `loader-diff` (227,244) and `api-diff` (70,224,318) failed only on a log line the harnesses had not normalised, below; then every step passes: the four test suites, `libcxx-diff`, `regex-diff`, `unicode-diff`, `vocab-diff`, `arch-diff` (124,290), `loader-diff`, `batch-diff` (180,654), `chat-diff` (779,803; three locale sweeps skipped, the locales are not installed there), `api-diff`, `port-coverage` (21 units, 100%), `port-links` (5,700 citations). `sched-diff` and `graph-diff` report themselves skipped, as designed |
| `make cuda-node-diff`, `make cuda-parity-cli` | re-run after the `@panic` fix: 2,750 nodes and 33 fused decodes at 1 and 4 threads; 6 / 6 |

Two real bugs, both invisible to the container that had passed the same
gates, and both now fixed.

### Zig's `compiler_rt` shadowed glibc's `expf` and `logf` in every CUDA build

`ops-diff` failed by one ULP on a few elements of exactly the ops that call
`expf` or `logf`; `expm1f` and `tanhf` were fine. `LD_DEBUG=bindings` said
why: in the port's process `expf` and `logf` never bound to libm. The
executable defined them itself, as `compiler_rt.exp.expf` and
`compiler_rt.log.logf`.

The chain, each link measured:

- Zig links `compiler_rt` as **one object**. Whatever a link needs from it
  brings all of it, and it defines `expf` and `logf` -- weakly, but a
  regular object's weak definition still beats a shared library's.
  `compiler_rt` has no `expm1f` or `tanhf`, which is why those agreed.
- What the link needed from it was `__divti3`, and the only reference was
  `ggml_zcu.o` -- our own Zig. `objdump` put every call in
  `std.Io.Threaded`: `sleep`, `batchAwaitConcurrent` and the timestamp
  setters, dividing `i96` nanoseconds.
- `std.Io.Threaded` was there because **the CUDA host layer called
  `@panic("OOM")` at 28 sites**. In ReleaseFast `@panic` still reaches
  Zig's default panic handler, which writes through the debug `Io`. The
  CPU-only library has no reachable `@panic`, which is why a `-Dcuda=false`
  build on the same machine was 650 / 650, and why the container, with no
  CUDA, never saw it.

So **every CUDA build of the port computed `expf` and `logf` with Zig's
implementations instead of glibc's**, in the shipped `llama-cli` as much as
in the gates. Upstream's CUDA build, linked by gcc, calls glibc's. The CUDA
gates could not see it: their reference (`-Dcuda-port=false`) has no
`@panic` either, but Qwen3.5's CPU-side ops are multiples of the vector
width and never reach the scalar tail that calls `expf`.

**The fix is the project's own convention**: `catch impl.abort("ggml-cuda:
out of memory")`, the port of `GGML_ABORT`, which every other ported file
already used. `__divti3` is gone from `libggml.a` and `ops-diff` passes
with CUDA built in.

**And a check, so it cannot pass silently again**: `libm_check` in
`scripts/platform-libs`, called by `ops-diff`, `node-diff` and
`parity-port` right after they link, fails unless both sides take their
libm functions from the same place, and says so in those words rather than
as a kernel mismatch. Negative-tested with a program defining its own
`expf`. Any `@panic`, `std.debug.print` or `std.log` call reachable from
the library would bring it back.

### `char` is unsigned on aarch64 Linux

`vocab-diff` failed on one synthetic vocabulary: RWKV piece 266, same
length, a different byte. `llama_unescape_rwkv_token` reads a hex digit as
`(c >= 'a') ? (c - 'a' + 10) : (c - '0')` on a plain `char`, and the
synthetic token puts a byte above 0x7f after `\x` -- on Darwin and x86 it
is negative and takes the second arm, on aarch64 Linux it is not. The port
had written Darwin's answer as `i8`. **Zig's `c_char` carries the target's
signedness**, so that is the type now, here and in `chat.zig`'s `upper`
(which gave the same bytes either way, but said otherwise). `isspace` in
the "C" locale answers the same for either signedness, so `vocab.zig`'s
`isSpace` stands.

The general lesson is the `-fmath-errno` one again: a Mac configured to
look like another platform is not that platform. `GATE_CPU=generic` gives
the Mac Linux's CPU features and nothing of its ABI.

### Two harnesses compared a live memory reading

CUDA on a unified-memory device logs `ggml_backend_cuda_get_available_uma_memory:
final available_memory_kb: N`, read from `/proc/meminfo` at that moment,
and the two sides load one after the other. `context_diff.zig` and
`gemma4_assistant_diff.zig` already masked it; `loader_diff.zig` and
`api_diff.zig` did not, because they had never run on the GB10. Both mask
it the same way now. Nothing about the port moved.

---

---

## `jinja-diff`: the chat templates against upstream's engine

`PLAN.md` carried "a `chat-parity` gate" as an open gap: nothing checked
that zigjinja and upstream's `common/jinja` agree on the templates models
ship. `scripts/jinja-diff` (`make jinja-diff`) compiles upstream's engine
unmodified beside `cli/chat.zig` -- `harness/jinja_ref.cpp` renders as
`common_chat_template` and `common_chat_template_direct_apply_impl` do --
and renders every template in `llama.cpp/models/templates/`, 68 of them,
over seven conversations: a lone user turn, a system prompt, four turns,
no generation prompt, control-token and Jinja text in a message,
surrounding whitespace and non-ASCII, and two user turns in a row. Both
sides bind what `chat.zig` binds. It compares the text, which side raises,
and trust: **no byte upstream marks as input may be trusted by ours**, with
the one designed exception that `chat.zig` re-trusts an exact
`bos_token`/`eos_token` segment where upstream marks every bound variable.
Without that exception every template that opens with `{{ bos_token }}`
read as a violation at byte 0.

It fails, and what it measured is the point. At the pin, with `TZ=UTC`:

| | renders (of 476) |
|---|---:|
| identical | 191 |
| both raise | 7 |
| text differs | 83 |
| a byte upstream marks as input is trusted by ours | 6 |
| ours raises, upstream renders | 189 |

26 of 68 templates agree on every conversation, among them Qwen2.5, QwQ,
Qwen3-Coder, Llama 3.1/3.2/3.3, the Granite family, gpt-oss and the R1
distills.

**zigjinja cannot parse 20 of the 68 templates** (138 renders,
`SyntaxError`), and raises at render time on 51 more renders
(`AttributeError` 21, `RuntimeError` 23, `TypeError` 7) -- Gemma 4, Kimi,
GLM-4.6, the Mistral and Hermes tool-use templates, Nemotron, Cohere among
them. `chat.zig` folds every engine error into `TemplateFailed`, so the
harness repeats its steps against zigjinja to name the phase.

**zigjinja ignores `trim_blocks` and `lstrip_blocks`.** The `Environment`
has both fields and nothing reads them -- the lexer never looks. Upstream's
lexer defaults both on (lexer.cpp:112), as Hugging Face's
`apply_chat_template` does, and templates written without `{%-` rely on
it: SmolLM3, MiniMax-M1/M2, Nemotron-3 and the DeepSeek V3.2/V4 templates
render with extra newlines. **Setting the two fields in `chat.zig` looked
like a fix** -- the gate's identical count rose by 25 -- and was not: the
rise was the local date rolling over to match zigjinja's UTC one (below),
and a unit test asserting the trimmed output failed. The change was
reverted. A gate's aggregate moved for a reason that had nothing to do
with the change, and only the narrow test told them apart.

**`strftime_now` formats local time upstream and UTC in zigjinja**, so a
template that prints today's date (Llama 3.x's `Today Date:`) agrees only
while the two dates coincide. The script runs under `TZ=UTC` so its
verdict does not depend on the hour.

### The trust scheme is wrong in both directions

`chat.zig` marks every expression's output untrusted and leaves template
literal text trusted, so that a message cannot forge control tokens. The
gate shows two ways that diverges from where the input really is.

- **Input left trusted.** `mark` leaves a string that already contains a
  sentinel alone, so a macro's output passing through `{{ }}` a second time
  is not double-wrapped. Qwen3.5 writes a user turn as
  `'<|im_start|>' + message.role + '\n' + content + '<|im_end|>'`, where
  `content` is `render_content(...)`'s output and already marked -- so the
  whole concatenation passes through, and **`message.role`, which is
  input, comes out trusted**. That is the six violations, all Qwen3.5. The
  role comes from our own CLI today, but any input concatenated with a
  macro's output escapes the same way.
- **Template control tokens left untrusted.** The same concatenation
  *without* a macro -- Qwen2.5's `'<|im_start|>' + message['role'] + '\n' +
  message['content'] + '<|im_end|>' + '\n'`, Llama 3's
  `'<|start_header_id|>' + message['role'] + ...`, Qwen3, gpt-oss -- is
  wrapped whole, so `<|im_start|>` is in an untrusted run, and
  `Session.tokenizeSegments` tokenizes it with `parse_special` off. **For
  those templates `-cnv --jinja` feeds the model `<|im_start|>` as plain
  text pieces instead of the control token.** The gate counts it: 192
  renders across 29 templates have template text upstream takes as the
  template's that ours leaves untrusted. It is reported, not failed, since
  ours is coarser by design; what was not known is that the coarseness
  reaches the control tokens.

Neither is fixable at the output boundary, because `Environment.finalize`
sees only the finished string. What fixes both is upstream's design --
provenance carried per string part through every transformation -- which
`NOTES.md`'s chat-template section measured at ~360 sites in the engine.
Recorded as a question in `PLAN.md`.

Negative-tested: removing `env.finalize = mark` takes trust violations from
6 to 198 and identical renders from 191 to 0.

At this pin upstream computes the marking but its CLI path flattens it
(`parts->as_string().str()`, chat.cpp:984), so upstream's own prompt does
not use it either; the trust check compares ours with upstream's notion of
input, not with what its CLI tokenizes.

---

---

## The Windows pickup's host gates, built

`PLAN.md`'s Windows pickup lists `regex-diff`, `unicode-diff`,
`libcxx-diff` and `host-arms-diff` built for Windows. All four now
cross-build for `x86_64-windows-gnu` at x86-64-v3 and run through
`scripts/win-run`: `make win-host-diffs`, or `WIN_X86=1` on any of the four
scripts. They have not run: the laptop's agent was not up.

- **`scripts/win-target`**, sourced by the three in-process scripts, holds
  the target, the port archive, the system libraries and how to run the
  harness -- in place on the host, or kept under `zig-out/win-host/` and
  pushed under `WIN_X86=1`. The Mac's compile lines are unchanged, and so
  are its counts: 452,904, 43,537,552, 7,070,454.
- **`unicode_diff.zig` takes its inputs compiled in**, through an `inputs`
  module the script generates -- `push` carries a program but no files.
- **`host_arms.c` has Windows arms**: thread priority from
  `GetThreadPriority`, the power-throttling state from
  `GetThreadInformation` in the policy slot (the C opts every thread but a
  LOW one out of execution-speed throttling), affinity through
  `SetThreadAffinityMask`, which returns the previous mask, and the seed
  window from `GetSystemTimePreciseAsFileTime`, which is what libc++'s
  `system_clock` reads there -- MinGW's `gettimeofday` ticks too coarsely
  to bracket a seed. `harness/win_host_arms.c` does the script's half on
  the laptop: one child process per scenario, the same transcript, and
  `win_lines_diff.c` compares the two sides there.
- **`zig translate-c -target x86_64-windows-gnu -lc` never finishes**
  (Zig 0.17.0): it spun at 100% CPU for minutes. Without `-lc` it takes
  under a second and `llama.h` needs nothing more.

## zigjinja against `jinja-diff`

Decision 58, carried out on 2026-10-22: `../zigjinja` fixed until upstream's
68 model templates, over `jinja-diff`'s seven conversations, render
byte-identically to upstream's `common/jinja` -- the text, which side raises,
and every byte's input mark, both ways. It started at 26 of 68 and 191 of
476 renders; it ends at 68 of 68, 469 identical and 7 where both raise, and
the clock agrees under `TZ=UTC` and `TZ=Australia/Lord_Howe`. zigjinja's
own suite went from 733 to 763 tests, all passing, and `bench-check` still
passes against its checked-in Python reference.

**The engine gaps, in the order the gate surfaced them.** Most were
silent -- a successful render of the wrong text -- which is why a
differential gate found them and the fixture suite had not:

| gap | measured as |
|---|---|
| no dict or tuple literals | 13 templates failed to parse; `{{ {'a': 1} }}` rendered as raw text |
| `trim_blocks`/`lstrip_blocks` were fields the lexer never read | stray newlines on most templates (`T`) |
| statement keywords were keywords everywhere | `{% for call in ... %}`, `{% set block = ... %}` failed to parse |
| `x if c` with no `else` | parse error |
| `and`/`or` returned booleans, not an operand | `message.get('name') or message['role']` was `true` |
| no Python methods: `.get`, `.items`, `.append`, `.pop`; `.strip()` and the rest ran Jinja2 filters | `RuntimeError`, and marks lost |
| **an empty `{% else %}` swallowed the rest of the template** | Cohere2MoE rendered 535 bytes and stopped, successfully |
| **a statement inside `{% set %}`/`{% filter %}`/`{% with %}` blocks was emitted as text** | firefunction printed `filter trim` into its prompt |
| the optimizer folded `{% if false %}` past its `elif`s | found while isolating the empty-`else` bug |
| `default` replaced every falsy value, `true` included | poolside's generation prompt vanished: `add_generation_prompt \| default(false)` |
| `selectattr` ignored its test | Apriel's system turn never rendered |
| `dictsort` returned `{key, value}` dicts | `for k, v in d \| dictsort` raised |
| attributes of strings raised | `content[0].type` on string content |
| dicts iterated in hash order | -- (fixed with the literals, before it showed) |
| `-#}` did not trim | GigaChat's prompt began with two newlines |
| `select`/`map` on none | `none \| selectattr(...) \| list \| length` was 1 |
| `strftime_now` formatted UTC | the clock check, under Lord Howe |

Where upstream departs from Jinja2 the engine follows Jinja2 unless a
shipped template depends on the difference: `default` treats none as
missing, as upstream does; `split()` with no separator splits on
whitespace runs, as Python does -- upstream's single space is a FIXME in
its source, and no template it ships calls `split()` without one.

**Trust is now upstream's scheme, carried by address.** `zigjinja`'s
`Value.string` is a plain slice, and giving every string a parts list would
have touched ~580 sites. Instead `src/provenance.zig` keeps a marked
string's bytes in its own arena, never reused during a render, and looks
its parts up by address; `Value.deepCopy` shares such a slice and
`Value.deinit` leaves it alone, so a mark survives every variable read.
An operation that builds a new string is unmarked unless it says
otherwise -- upstream's default too -- so a missed propagation shows up
in `jinja-diff` as a trust difference rather than passing silently. In
rendered text the marks travel in band, between sentinel bytes, and are
decoded where text becomes a value again (a macro's result, a `set`
block) and at the end. Three things that follow from matching upstream
exactly, and are upstream's behaviour as much as ours:

- `tojson` and `join` produce unmarked -- trusted -- strings, so message
  content passed through either reaches the tokenizer with
  `parse_special` on. Upstream marks neither.
- `split` marks only its last piece; `content.split('</think>')[0]` is
  trusted.
- A part-wise `capitalize` capitalizes each part: `('a' + content) |
  capitalize` is `A` + `Content`.

`bos_token` and `eos_token` are bound as provenance *tokens*: input for
every propagation rule, re-trusted at the end by their own runs. The old
`chat.zig` re-trusted any run spelled like one, so a message reading
exactly `</s>` was honoured as EOS; `chat.zig` has a test for that now.

**The VM.** `Runtime.renderString` uses the bytecode VM whenever it can,
and it had its own `and`/`or`, its own `default` fast path, and compiled a
method call to a filter. `and`/`or` and `default` were fixed there too;
method calls and dict literals now send a template to the AST path.
`chat.zig` was already on the AST path, because only `visitOutput` writes
the marks.

**Negative tests, 5/5**, each injected into `../zigjinja`, run, and
reverted:

| injection | `jinja-diff` |
|---|---|
| `strip` drops its marks | 23 templates fail |
| `~` drops its marks | 4 fail |
| `lstrip_blocks` off | 9 fail |
| every decoded render marked input (over-marking) | 68 fail, 440 on trust |
| `strftime_now` offset forced to UTC | passes under `TZ=UTC`, fails the clock and 7 templates under Lord Howe |

**What is not covered.** On Windows zigjinja reads the system zone
(`GetTimeZoneInformation`) and ignores `TZ`; `jinja-diff` there runs once
as configured. A `TZ` value is honoured only where zigjinja can reach the
environment, which is through libc -- every llamazig build links it. The
VM still writes no marks; a caller who wants them must compile for the AST
path, as `chat.zig` does.

## `cnv-diff`: the conversation loop against a reference

`PLAN.md` carried "no gate covers the conversation loop": `parity-cli` runs
one-shot prompts, and multi-turn behaviour, the KV prefix reuse and the
hybrid-cache fallback were checked by hand. `make cnv-diff` is that gate.

**The reference** is `harness/conversation_ref.cpp`: SPEC.md section 5.3
written a second time over the stock pieces -- `llama.cpp/common/jinja`
renders, as `harness/jinja_ref.cpp` drives it, and `llama.cpp.zmake`'s
libllama decodes. Upstream's own `llama-cli` cannot be the reference: in
this version it is a client of the server (`tools/cli/cli-context.cpp`),
stores replies split into `content` and `reasoning_content`, and frames
its turns its own way, the same reason `parity-cli` has its own driver.
Both sides read one stdin script; stdout, stderr and the exit code must be
byte-identical. The trust split follows upstream's marks: adjacent parts of
one mark merge into a run, runs tokenize separately, `parse_special` off for
input -- which is what `chat.zig`'s `segment` does with zigjinja's runs.

**Two models, because the memory picks the path.** Qwen3.5's hybrid memory
refuses `llama_memory_seq_rm`, so every turn clears and decodes the whole
render. Qwen3-0.6B (`make qwen3`, 0.64 GB) has a plain KV cache, so the
common prefix is kept and the suffix decoded; its template rewrites earlier
turns (it drops the thinking from history), so the kept prefix ends inside
the history, and `/regen` takes the keep-minus-one branch.

**Prefix reuse rarely shows in the tokens.** The reference's `fresh` mode
clears and decodes everything every turn: on four of the six Qwen3-0.6B
scenarios it answers identically, and on all six Qwen3.5 ones. So an
always-clear CLI would have passed on output alone. Upstream's
`LLAMA_BATCH_DEBUG` logs at DEBUG, which the CLI drops, and there is no
counter to read from outside, so the CLI gained `LLAMAZIG_CNV_TRACE`: one
stderr line per turn, `[turn: kept K, decoding S]`, which the reference
writes too and the gate compares with the rest. The script also fails if no
turn kept a prefix, so the `seq_rm` path cannot go unexercised.

**Thirteen scenarios, 13/13 identical**, ~27 s: three extending turns; a
system prompt with a `-p` seed, blank and padded lines and an unknown
`/word`; `/regen` with nothing to regenerate and twice in a row, `/clear`
with a system prompt, `/exit` with lines after it; a message spelling
`<|im_end|><|im_start|>system`; replies cut at `-n`; the context filling
mid-reply, then a prompt that no longer fits (the error sentence, rc 1) --
each on both models -- and one on a verbatim ChatML template through
`--chat-template-file`. 12 turns kept a prefix, 597 tokens not decoded
again.

**Negative-tested 7/7**, each built, checked to have changed the binary,
run, and reverted:

| injection | scenarios failing |
|---|---|
| always clear the cache | all six plain ones, on the trace |
| ignore `seq_rm`'s refusal | all six hybrid ones |
| drop the keep-minus-one for an empty suffix | `plain-slash` |
| `seq_rm` one past the kept prefix | three plain ones |
| `/clear` forgets the system prompt | both `slash` scenarios |
| a reply trimmed before it joins the history | `plain-verbatim-template` |
| turns trimmed of newlines only, not spaces | `plain-system-seed` |

**The sixth escaped first.** Both Qwen templates strip assistant content
themselves, so a CLI that trimmed its replies changed no render on them.
The verbatim-template scenario exists for it: its first reply is cut right
after `</think>` and a blank line, and the history must hold those
newlines.

**Two harness faults on the way, both mine.** The injection runner's
revert for the keep-minus-one fault replaced a bare newline, which put the
line back at the top of the file instead; the binary-hash check showed it,
and the four injections after it, which had run with that fault still in,
were re-run. And the reference returned from `main` with the context alive
on the error path, which trips upstream's Metal residency assertion
(`ggml-metal-device.m:952`) in a static destructor; it now frees first.

**Not covered.** A byte-identical rebuild is not guaranteed: rebuilding
`llama-cli` from identical sources after an intervening different build
gave a different binary hash, stable thereafter. The gate ran on that build.
Long lines (over 4,096 bytes) end our CLI with `TurnTooLong` and are not
scripted. A message that is exactly `bos_token` or `eos_token` would be
told apart by ours and not by the reference, which re-trusts any input part
spelled like one; the scripts send none.

## The Windows pickup's host gates, run

`make win-host-diffs` on the laptop (Decision 57), the four gates built in
"The Windows pickup's host gates, built":

| gate | Windows result |
|---|---|
| `host-arms-diff` | 168 / 168 lines identical: priority, power throttling, affinity, the seed window |
| `libcxx-diff` | 452,904 identical |
| `unicode-diff` | 7,070,454 identical, the 16-bit `wregex` fallback included |
| `regex-diff` | 43,537,552 identical, after one fix in `regex.zig` and one in the harness |

**`PLAN.md`'s open question has its answer, and it was not the one it
feared.** The worry was that the UCRT's `iswctype` would classify
non-ASCII code points where Apple's does not. It does not: above 0x7f
nothing is classified, wide or narrow, exactly as on the Mac. The one
difference is `\t`: the UCRT's classification table sets `_BLANK` for the
space alone, so `[[:blank:]]` misses a tab on Windows. It does so in
`std::wregex` too, because libc++'s classic `ctype<wchar_t>::do_is` reads
`ctype<char>::classic_table()` for ASCII (`locale.cpp:694` in the libc++
the reference links), and on Windows that table is the UCRT's
(`__locale`'s `_LIBCPP_MSVCRT_LIKE` masks). `regex.zig`'s `ctypeOf` takes
the arm as `tab_is_blank`.

**My first reading of it was wrong, and the gate said so.** The first run
printed four narrow failures and capped each wide pattern at four lines,
so the fix went in for `std::regex` only, as a per-bracket flag. The
re-run left 17 failures: `[[:blank:]]` against U+0009 in the wide sweep,
and the sixteen code points that narrow to it. The arm moved into
`ctypeOf`, for both engines, and the flag went.

**The other 357,000 failures were the harness.** The reference stores each
swept code point with a `(wchar_t)` cast, which keeps 16 bits on Windows,
while the Zig side passed the full value: U+10041 is `A` to one and an
unclassified code point to the other. The port never sees such a value --
`unicode.zig` narrows to `wchar_t` before it calls `regex.WRegex` there,
which is what `unicode-diff` checks -- so `regex_diff.zig`'s `wchar` now
narrows the Zig side's inputs the same way.

**Negative tests**: the two failing runs before each fix, on exactly the
cases the fixes address; the Mac's run is unchanged at 43,537,552.

**Getting there**: the laptop's agent did not reach this Mac for an hour,
then did after a restart, then stopped again while an interactive
`zupdater` console was running on the Mac. Each agent appears to
attach to one host process at a time -- a GB10 `tty` also waited while a
Windows push held the link -- and the console was one. A push that waits
on an absent agent waits forever; `win-run` cannot tell that from a slow
gate.

## The xcframework after the swap: no C++ runtime, and an iOS build that had broken

`PLAN.md` noted that the xcframework still linked Zig's libc++ into each
slice, "which nothing needs since the swap". Measured first: the macOS
slice's `libggml.a` and `libllama.a` reference no C++ runtime symbol --
no `std::__1`, `__cxa_*`, `__gxx_personality`, `operator new` or vtable --
only `_Unwind_Resume`, which libSystem's libunwind supplies. And the
prelinked object built with `libc++.a` and `libc++abi.a` beside
compiler_rt is **byte-identical** to one built without them, before the
change and again after a full rebuild. So the probe is a plain C link that
fetches compiler_rt alone, the prelink names only that archive, and the
`noabi` negative test is gone: it injected "libc++abi left out of the
prelink", which since the swap changes nothing, so it would have reported
an escape.

**Rebuilding the slices found a real break.** The iOS and simulator slices
no longer compiled: `unicode.zig`'s `unicode_regex_split` error path
called `std.debug.print`, which reaches `std.Io.Threaded`, and Zig 0.17's
iOS `NullFile` has no `fd` (`Threaded.zig:15486`). The slices had last been
built on 2026-10-13, before libllama swapped, so nothing had compiled that
code for iOS since. The grammar parser had already met the same thing and
carried an iOS-only `write(2)` arm. Both now go through `impl.writeStderr`,
an `fwrite` to C's `stderr` -- the bytes and the stream of the C's
`fprintf(stderr, ...)` -- on every target, which is also what `CLAUDE.md`
asks of library code (no `std.debug.print`, which drags in compiler_rt's
`expf`). **`make validate` now compiles the libraries for iOS**, so the
next one fails there rather than at the next xcframework build; its
negative test is this failure.

Gates: `make -C xcfwk verify` on all three slices (1,438 exports, 0
private externs, 0 `std::__1`, compiler_rt math inside, stub links with no
libc++), `negative` 8/8 caught, `smoke-macos` 6/6 on the GPU and the CPU
plus app C++ beside the framework, `smoke-sim` 6/6 on the iOS simulator's
CPU; `zig build test-llama` 300/300,
`unicode-diff` and `sampler-diff` unchanged.

## The Vulkan arms, completed

`PLAN.md` left two Vulkan items open after the swap: the no-coopmat arm's
`vk-ops-diff`, and the KHR coopmat arm's FLASH_ATTN_EXT cases after the
point where the reference stalls against itself. Both are closed, on all
four arms, and `scripts/vk-ops-diff` now finishes every arm on its own.

**Every arm is bit-identical wherever the reference can compute.**

| arm | ops other than FLASH_ATTN_EXT | FLASH_ATTN_EXT |
|---|---|---|
| coopmat2 | 17,015 / 17,015 in all, FLASH_ATTN_EXT included (step 8) | 5,150 identical, in one process |
| KHR coopmat | 10,650 / 10,650 | 5,148 identical; 2 hang in the reference alone |
| none | 10,650 / 10,650, `supports_op` agreeing both ways | 5,150 identical |
| none, no integer dot | 10,650 / 10,650 | 5,150 identical |

The family is 5,158 cases on every arm; 8 of them, at `hsk=64` and
`hsk=128`, are unsupported on both sides. Outside FLASH_ATTN_EXT, 4,062
case lines are unsupported on both sides; `supports_op` agrees both ways
on every arm.

**The stall, characterised.** On the KHR and no-coopmat arms a single
process running the whole FLASH_ATTN_EXT family stops at `hsk=256,
kv=16384, q8_0` and never resumes. A backtrace, taken by preloading a
`prctl(PR_SET_PTRACER_ANY)` shim because the GB10's Yama scope forbids
attaching otherwise, has the main thread in the port's `waitForFence`
spin -- upstream's `ggml_vk_wait_for_fence`, which exits on a lost device
and otherwise polls -- while `nvidia-smi` reports 0% utilisation, the
208 MHz idle clock and 5 W. Nothing is running on the GPU; the host waits
on a fence for work the queue never ran. That case alone passes in 1.2 s.
Run head size by head size in fresh processes, the KHR arm still stalls
inside `hsk=576`, at case 256 of 259, and **the reference against itself
(`VK_DIFF_SELF=1`) prints byte-for-byte the same 54,201 bytes and stalls
on the same case.** It is upstream's host layer or the driver, not the
port.

**Two cases hang even alone, in the reference too:** the MLA decode shape
`hsk=576, hsv=512, nh=1, nr23=[20,1]`, q8_0, `nb=1`, V a view of K, at
`kv=113` and `kv=1024`, on the KHR arm only -- the same GPU-idle wait,
reference against reference, every time. No arm had ever computed them:
every earlier KHR run stopped at the `kv=16384` stall first.

**`scripts/vk-ops-diff` with no arguments now** runs the other ops in one
process and FLASH_ATTN_EXT one head size per process, from
test-backend-ops' own list of head sizes; a part that outlives 600 s has
the cases it finished counted from its log, and every case from the one it
stalled on runs in its own process (`support` mode lists the part's cases
without computing them; `-o` takes a whole case string as an exact name).
A case that hangs alone is run again reference against reference; if that
hangs too it is reported `REFHANG` and counted in the verdict, not failed
-- the counterpart of `NONDET` -- and if only the port hangs, it fails.
With arguments it is one run, as before. Run as committed on the KHR arm,
end to end: `PASS`, 15,798 cases bit-identical, the stall at case 256 of
`hsk=576` finished one case per process, and both MLA cases reported
`REFHANG`.

**Coverage is measured, not assumed.** The per-head-size parts must cover
the family: on coopmat2, which never stalls, one unchunked
FLASH_ATTN_EXT run prints 5,158 case lines, exactly the sum of the eleven
parts.

**Two mistakes of mine on the way**, both in the scratch runners, neither
in a result: a waiter `while pgrep -f vkchunk.sh` matched its own command
line and would have waited forever; and the first single-case fallback
read `$?` after a `!`, which is the negated status.

## The UNO Q, and `--fit`: what running on a 3.6 GB board found

**The board.** An Arduino UNO Q: Debian 13, kernel 7.0, glibc 2.41, four
Qualcomm Kryo cores (`CPU part 0x801`, A53-class) whose `/proc/cpuinfo`
features are `fp asimd aes pmull sha1 sha2 crc32` -- no `asimddp`, no
fp16 -- which is exactly Zig's `generic` aarch64 baseline, the CPU
`GATE_CPU=generic` reproduces here. 3,667 MiB of RAM, `/tmp` a 1.8 GB
tmpfs. It enumerated on the Mac's USB, not the Windows laptop's, and
`../zigadb`'s own CLI drove it natively (`zigadb -d push`, `zigadb -d shell
-n -T`). Our `llama-cli` cross-built as `aarch64-linux-gnu`; the reference
side is `harness/raw_completion.c` linked against `llama.cpp.zmake`'s
`zig build lib -Dtarget=aarch64-linux-gnu`, both pushed and run on the
board by a seven-prompt script, the hashes checked there.

**Parity: 7 / 7 on `plain-d6_base_005000.gguf`, and 7 / 7 on
Qwen3-0.6B Q8_0** at `-c 2048`. The comparison was made to fail first: the
same script with `--repeat-penalty 1.3` on our side only reports 7 / 7
differing. `plain-d6` runs at 232 t/s prompt and 80 t/s generation, Qwen3 at
13 and 9.

**Qwen3 at default flags was killed.** `-c 0`, the default, asked for the
trained context, 40,960 tokens: a 4.6 GB KV cache on a 3.6 GB board. On
Linux the allocation succeeds under overcommit; the kernel's OOM killer
took the process (anon-rss 2.77 GB) when the cache was first cleared --
and, the process being in `adbd.service`'s cgroup, the USB session with it.
`session.zig`'s comment had said that where upstream would shrink the
context ours "fails to create the context instead". Not on Linux.
Upstream's own `llama-cli`, pushed beside ours, ran: `common_fit_params`
fits the parameters left unset to free memory before anything loads, and
on the CPU alone measures against total system memory.

**`cli/fit.zig` ports `common/fit.cpp`, the whole file**, with the user's
answer to how the CLI reaches libllama's internals: as Zig. `fit.cpp`
includes `../src/llama-ext.h`, whose memory breakdown is a `std::map` with
no C signature, so `api/` gained the four `llama-ext.h` functions it calls
(unexported, like the other eight), and **the CLI now compiles libllama's
module into itself** instead of linking `libllama.a` -- one compilation, so
the `llama.h` calls and the Zig ones reach the same code. `-fit`, `-fitt`
and `-fitc` are parsed as upstream does, and an explicit `-c 0` forbids the
shrink, as upstream's `-c` handler does. The log is upstream's `common_log`
at `llama-cli`'s verbosity, `LOG_LEVEL_ERROR`, so only the error line can
reach the terminal; every line is formatted by `snprintf` with the C's
format string, because Zig's `{d:6}` prints a `+` on a signed integer and
`{d:.2}` rounds 0.125 to 0.13 where `printf` gives 0.12.

**`make fit-diff`** runs 68 cases through `fit.cpp` and `log.cpp`, compiled
unmodified over `libllama-ref.a`, and through `fit.zig` over the port, in
one process, and compares the status, the context size, `n_gpu_layers`,
where `tensor_split` and the overrides point and every entry of both
buffers, every line fit logs at trace level with its level letter, and
every libllama message through its log filter. The fit decides from what
devices report free, so `harness/fit_dev.c` adds up to four fake GPUs whose
memory the case sets -- each the CPU device relabelled, with a non-host
buffer type of its own pointing back at it, so the breakdown attributes to
it what lands there -- and memory is set relative to what each model was
measured to need: fractions of one device's use, the host margin as a
fraction of the way from the least context's use to the full one's. Plus
the real Metal device, and the CPU alone. **Every arm of `fit.cpp` must
appear in some reference transcript** (38 phrases), so a case list that
drifts out of an arm fails rather than passing by testing nothing -- it
caught exactly that twice while the cases were written: one-device sizes
measured at libllama's default 512-token context made every "fits" case
too small, and fixing that moved three refusal cases past the reduction
that returns before them. ~55 s.

**It found two bugs in the port, both where Zig and C part on undefined
behaviour**, and only the ReleaseSafe build of the harness saw them:

- `for (id_dense_start + 1..nd)`: the C's `for` runs no times once
  `id_dense_start` has reached `nd`; a Zig range starting past its end is
  illegal -- a panic with safety on, undefined in ReleaseFast.
- `assert(n_layer >= n_part)` in `n_full()` **fails in upstream too**, on
  three fake GPUs at 40/10/30% with the MoE fixture; upstream's Release
  build compiles it out (`NDEBUG`, which `zig c++ -O2` also defines) and the
  unsigned subtraction after it wraps. Ported as `std.debug.assert` and `-`,
  both are undefined in ReleaseFast. `fit.zig`'s `cAssert` compiles to
  nothing and the subtractions wrap, so the port does what the shipped C
  does, in every mode -- and the two transcripts agree after it.

**Negative-tested 12 / 12 effective**: the alignment of the shrunk context,
a log width, the false-position comparison, which buffer a partial layer
overflows to, the attention-overflow regex, the MTP extra model's weights,
an accelerator's 0/0 memory, the log filter's demotion, the streams under a
unified cache, the description-prefix strip, the GATE attempt, the MoE
surplus. Two first attempts did not compile -- removing a use left a
variable unused -- and are not counted. **One escaped**: `mem_test <=
target` as `<` changes nothing unless a measurement equals its target
exactly, which no case did. Four tie cases now set a device's free memory to
exactly what L layers measure, at a context the user set so step 3 runs,
and that injection fails exactly those four.

**Upstream's choice, measured on the board: 18,176 tokens; ours was
15,616.** The fit is gated identical, so the inputs differed:
`common_params` defaults `n_outputs_max_per_seq` to 1 and the server
`llama-cli` drives sets `n_outputs_max` from `server_output_limits` -- one
sequence, no draft, one output -- where `llama_context_default_params`
sizes the output buffer for the whole 2,048-token batch, 1.2 GB of logits
for a 151,936-token vocabulary. Our CLI only ever reads the last token's
logits, so it now takes upstream's limits: on the board it picks 18,176,
parity is 7 / 7 on both models at default flags, and the CLI no longer
holds 1.2 GB it never reads, on any machine.

**`cnv-diff` covers the wiring.** `fit-diff` gates what the fit decides,
not that the CLI hands it to libllama. Both sides of `cnv-diff` now report
`[ctx: N, gpu layers: G]` under the trace -- the reference runs
`common_fit_params` as `common_init_result` does, its log flushed before
the load -- and three Qwen3-0.6B scenarios use a margin beyond any
machine's memory, so the outcome is machine-independent: the context
shrinks to 4,096 with no layer on the GPU, an explicit `-c 0` keeps 40,960,
`--fit off` changes nothing. 16 / 16; an explicit `-c 0` that stopped
forbidding the shrink fails one scenario, and a fit on a copy of the model
parameters fails two.

**On the way, `upstream_flags.zig` was missing sixteen real flags.** Its
extraction regex expected `{"-x", "--y"}` and upstream also writes
`{ "-fit", "--fit" }`: `-fa`, `--flash-attn`, the whole `--fit` family and
eight more were refused as typos rather than as flags not yet supported.
Regenerated with the spaces allowed: 541 spellings, not 525.

## Vulkan on the UNO Q's Adreno 702

**The GPU.** The UNO Q's QCM2290 carries an Adreno 702, driven by Mesa's
Turnip (`unoqimg-mesa-vulkan-turnip` 26.1.6, Vulkan 1.4.354) through the
board's Khronos loader 1.4.309; the user is in `render`. Upstream's backend
describes it as `Turnip Adreno (TM) 702 | uma: 1 | fp16: 1 | bf16: 0 | fp4:
0 | warp size: 32 | shared memory: 16384 | int dot: 0 | matrix cores:
none` -- **a fifth arm**, beside the GB10's four: fp16 arithmetic, no
integer dot, no matrix cores, 16 KiB of shared memory, unified memory.
`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT` changes nothing here, so the GB10's
`none` and `nodot` arms collapse into one.

**Building for it from this Mac.** The pinned toolchain (Decision 59) had
only ever been built on the GB10; `scripts/vk-toolchain` now builds the
same shaderc v2026.4 and SDK 1.4.363 headers on macOS (the loader and
`vulkaninfo` steps are Linux-only and skipped), which is all the SPIR-V
generation needs. Linking takes the **board's own** `libvulkan.so.1`,
pulled into `.tools/unoq/lib/` and passed as `-Dvulkan-lib`. That loader
imports `__isoc23_sscanf@GLIBC_2.38` and `dlopen@GLIBC_2.34`, and lld will
not link against a shared library whose symbols the target glibc lacks, so
the target is `aarch64-linux-gnu.2.41`, the board's glibc. libc itself is
not optional here: the loader `dlopen`s Turnip, a C shared object, which a
static binary cannot. Four builds, each `--release=fast -Dvulkan
-Dcuda=false`: `zig build cli` (our `llama-cli`), `zig build reference
-Dvulkan-port=false` (upstream's `ggml-vulkan.cpp` and `libllama-ref.a`,
for `raw_completion.c`), `zig build reference vk-ref` (the port, plus the
renamed reference for `vk-ops-diff`), and `harness/node_dump.c` linked
against each side. Strip what is pushed: unstripped, the reference driver
is 121 MB and the push takes minutes.

**What ran, on the board:**

| Check | Model | Result |
|---|---|---|
| parity, our `llama-cli` vs `raw_completion.c` over upstream's libllama and Vulkan host layer, greedy, 48 tokens | plain-d6 | **7 / 7 identical**; each side first shown on Vulkan by `GGML_VK_PERF_LOGGER` |
| `vk-node-diff` per node | plain-d6 | **384 lines identical**, reference reproducible against itself |
| `vk-node-diff` fused, 32 steps | plain-d6 | **33 lines identical**, reference reproducible |
| `vk-node-diff` per node | Qwen3-0.6B Q8_0 | **1,972 lines identical**, reference reproducible -- the quantized `MUL_MAT` paths plain-d6 (all f32) never reaches |
| `vk-node-diff` fused | Qwen3-0.6B Q8_0 | both sides `VK_ERROR_DEVICE_LOST` at the default submit size; **9 lines identical** (prompt and 8 steps) at `GGML_VK_MAX_NODES_PER_SUBMIT` 16 and 4 -- below |

Negative-tested on this GPU, port side only: `GGML_VK_DISABLE_FUSION=1`
fails the fused pass from step 2, and `GGML_VK_DISABLE_F16=1` makes the
port take the f32 matmul, which aborts on "Shared memory size too small"
-- 16 KiB is below what that shader needs. A third, `LLAMA_ARG_FLASH_ATTN`,
is inert: `node_dump` never reads it, so it is not evidence either way.

**Qwen3's long submissions are killed by the kernel, on both sides.**
At the default submit size the fused graph loses the device in upstream's
host layer and ours alike (`vk::Device::waitForFences: ErrorDeviceLost`),
and `dmesg` says why: the msm driver's `hangcheck_handler` times the
submission out and `recover_worker` resets the GPU, naming the process.
With `GGML_VK_MAX_NODES_PER_SUBMIT=16` or `4` on both sides no submission
is long enough, and the fused pass is identical. The per-node pass splits
the graph at every node, so it never hit the limit. The same timeout is why
upstream's own reference, at the default, prints text like
`ynosakespeareakespeareegrate` at 0.3 t/s: work lost to resets. plain-d6
decodes at 8 t/s prompt and 5.4 generation on the Adreno against 230 and
80 on the CPU, so on this board the CPU is the device to use; `--fit` and
the CPU path, gated above, are what make Qwen3 usable there.

## llamazig on an iPhone

**The device.** An iPhone 15 (`iPhone15,4`, A16, iOS 27.0.1) on this
Mac's USB, registered in a development team by `xcodebuild
-allowProvisioningDeviceRegistration` -- it was in no team before. The A16
runs the xcframework's `ios` slice, built for `apple_a14`.

**How it runs.** iOS has no shell, so the device side is an app:
`xcfwk/test/device`, a two-file Xcode project -- `objectVersion` 77, whose
file-system-synchronized `App/` folder makes the project file short --
running `test/smoke.swift`'s loop over parity-port's six prompts on the CPU
and on Metal, and printing each output as a `SMOKE <mode> <n> <base64>`
line. `make -C xcfwk smoke-device` stages it into `build/device` with the
model linked into the bundle and the framework beside it, builds and signs
it with `xcodebuild`, installs and runs it with `xcrun devicectl device
process launch --console`, and `smoke.sh device` compares each output with
`test/driver.c` on the port's own `apple_a14` libraries here. A SwiftPM
test target would have been shorter and cannot run there: "Tool-hosted
testing is unavailable on device destinations".

**12 / 12 byte-identical**, CPU and Metal. The CPU half is expected to be:
the same kernels, compiled for the same CPU, linking the same
`compiler_rt` math. Metal was measured, not assumed: the phone's GPU gives
the same bytes as this Mac's on all six prompts. Negative-tested by
altering one byte of the device's output.

**Three kills, and only the first was memory:**

- **Jetsam, at the first CPU prompt**, without the increased-memory-limit
  entitlement -- the jetsam report lists the app as the largest process, at
  1.3 GiB. The CPU path holds the repacked weights in anonymous memory; the
  app now prints its footprint after each prompt (`SMOKE-MEM`), 1,338 MiB on
  the CPU and 128 MiB on Metal.
- **The launch watchdog, everywhere else.** With the entitlement the app
  still died with SIGKILL partway through -- at Metal prompt 3, 4 or 5 from
  one run to the next -- and the footprint said 126 MiB used, 3,969 MiB
  available. The termination report (`devicectl device info files
  --domain-type systemCrashLogs`) said `0x8BADF00D`, "process-launch
  watchdog transgression ... exhausted real (wall clock) time allowance of
  20.00 seconds": the work ran in `main()`, so the app never finished
  launching. Two fixes made on the memory theory first -- an autorelease
  pool per prompt, and one model load per device instead of per prompt --
  were not what fixed it. The work now runs on a thread started after
  launch. **Read the termination reason before reasoning from a signal 9.**
- The first launch after an install twice returned none of the app's output
  on the console though it ran to completion; the target launches again.

**One thing measured on the way**: the port's footprint grew with every
model loaded and freed in one process, where upstream's does not. Found and
fixed next, in "The reload footprint".

## The reload footprint

**What it was.** Loading Qwen3.5 2B on Metal, decoding once and freeing
it, twenty times in one process: the port's footprint grew a steady ~7.2 MB
a cycle (248 MB to 377), upstream's from `llama.cpp.zmake` levelled off at
226 MB. `leaks --atExit` found 0 leaks on both sides, and a `malloc`
interposer found none of our large blocks live after the frees. `vmmap`
put the growth in `Malloc Large (empty)` -- 71 MB over ten cycles, against
upstream's 9: freed large blocks malloc keeps resident.

**Finding it.** Not by size or count -- both sides allocate about the same
-- but by address. Logging every block of 64 KiB and over across six
cycles, nearly every size comes back at the same one or two addresses each
cycle; a chain of four in `Vocab.load`, 2.2, 3.3, 5.0 and 7.5 MB, came back
at six distinct addresses in six cycles, each final block in a fresh 8 MB
region. That is an `ArrayList` growing by half, and `atos -i` against a
`dsymutil` dSYM named it through the inlining: `libcxx.StringMap`'s node
array, appended to once per token as `readTokens` fills `token_to_id`. A
second, `text_buf`, the one buffer holding every token's text, grew the same
way. libc++ allocates a node and a `std::string` per token, small blocks
that are always reused; the port's two growing buffers are its own storage
choice, and on macOS their final blocks did not land where the last load's
had been freed.

**The fix.** `readTokens` measures the texts first and sizes both before
filling them: `text_buf` with `ensureTotalCapacityPrecise`, and
`token_to_id` with a new `StringMap.reserveStorage`, which reserves node and
key storage only -- the buckets, which decide iteration order, still grow
and rehash as `__rehash` does, and a unit test holds a reserved map's
bucket count, order and values to an unreserved one's over 5,000 keys and
proves the reserved storage never grows. Measured after: **0.0 MB a cycle**,
flat at 243-247 MB over twenty cycles; `text_buf` alone took it to 5.7.
Sizing `buildPieceCache`'s `piece_buf` too was tried and changed nothing,
so it was left as it was.

**A suspect that was not it.** The interposer's first find was
`std.Thread`: the CPU pool spawns its workers with `std.Thread.spawn`, whose
entry point attaches a thread-local 256 KiB signal stack
(`Thread.maybeAttachSignalStack`; three instances make the binary's
`__thread_bss` 0xC0010 bytes), so on macOS dyld `calloc`s and frees 768 KiB
for every worker of every pool, and each worker gets Zig's 16 MiB stack
where the C's null attributes take the platform's default. Replacing it with
the C's `pthread_create` and `CreateThread` removed those allocations and
**did not move the footprint** -- 7.2 MB a cycle with and without -- so it
was reverted: it fixes nothing, and the port keeps `std.Thread`.

## BLAS over Accelerate

**What was asked, and what it came to.** zDNN, ZenDNN and BLAS, "not
external libs". Measured, each is a thin backend around a vendor library:
BLAS 530 lines around `cblas_sgemm`, zDNN 924 around IBM's `libzdnn`
(s390x with NNPA, z16 or later), ZenDNN 838 around AMD's C++ `zendnnl.hpp`,
which upstream's CMake downloads and builds with oneDNN, libxsmm and AOCL.
Without the library there is nothing left to port, so zDNN and ZenDNN are
not built. BLAS on Apple is the exception: the port already links Accelerate
on every Darwin target and calls its vDSP from the CPU backend, so a BLAS
backend over it adds nothing (PLAN.md, Decision 60).

**The port.** `src/ggml/blas.zig`, the C++ line for line with the arms this
target compiles: Accelerate's description, no vendor thread-count call, no
OpenMP -- so the dequantization runs on the `std::async` arm, one thread per
task and one split per plane, joined after the last plane. Its contract is
four C exports, the same four the reference object has (the rest of that
object is libc++ template code). `ACCELERATE_NEW_LAPACK` and
`ACCELERATE_LAPACK_ILP64`, which upstream defines for Accelerate, bind
`cblas_sgemm` to `cblas_sgemm$NEWLAPACK$ILP64` with 64-bit integers, read
off the reference object with `nm`; the port calls that symbol by name.
`-Dblas` builds it, Apple only, off by default -- the reference configuration
has BLAS off, and on by default it would change which device computes every
large CPU `MUL_MAT` in every gate. The registry registers it where the C++
does, after Vulkan and before the CPU.

**`make blas-diff`**: upstream's translation unit compiled with its four
exports renamed `ref_*`, linked beside a `-Dblas` libggml, both asked the same
questions in one process. Every vtable answer; `supports_op` over every type
at `ne0`, `ne1` and `ne10` straddling 32, F32 and F16 `src1`, contiguous and
not, the Hadamard hint, `OUT_PROD`'s types, ranks and `src1` layouts, the
view ops and ops it must refuse (333 true, 1,641 false); and 87 computations
on bits -- every type with a `to_float`, broadcast in both batch dimensions,
at one thread and four, plus a row count four threads do not divide and a
`k` wide enough to give each thread one row, and `OUT_PROD` with a plain and
a transposed `src1`. A case counts only if the reference overwrote a
sentinel in every output and lands near the CPU backend by normalized mean
squared error, so a graph neither side computed cannot pass by agreeing.
2,547 checks.

Negative-tested on twelve faults: eleven caught, one provably inert (the
thread count, which only changes how rows split -- each row converts the same
way). One escaped at first: dropping `ggml_is_matrix(src1)` from `OUT_PROD`,
because the sweep's 3-d case made both operands 3-d and `src0`'s own check
refused it first. `ggml_out_prod` broadcasts a matrix `src0` over a 3-d
`src1`; with that case and a 4-d one added it is caught. The `lda` fault was
caught by Accelerate itself: "Parameter number 9 passed to cblas_sgemm had
an invalid value".

**`make blas-node-diff`**, and why its first two attempts compared nothing.
`node_dump` over upstream's BLAS and over ours, every node of a decode. On
Qwen3.5 every node was identical -- and identical to a build without BLAS
too: its quantized weights sit in the CPU's repack buffers, which are not
host buffers, and BLAS's `supports_buft` takes host buffers only, so the
scheduler kept every multiply on the CPU. On the f32 `plain-d6` the same
again, for a different reason: `node_dump`'s `cpu` mode names the CPU as the
model's device, so libllama's backend list is CPU, BLAS, CPU, and the
scheduler gives a weight's op to the first backend that takes it. Upstream's
own libllama does exactly the same in both cases -- measured by linking
`libllama-ref.a` -- so neither is a port difference; they are why a gate
must prove the backend ran. `node_dump` gained a `host` mode, as `-ngl 0
--no-kv-offload --no-op-offload` loads: default devices, BLAS ahead of the
CPU. There, on `plain-d6`, 384 nodes identical at one and four threads and
32 fused steps identical, and **366 of the 384 differ from the build without
BLAS**, which the script requires. A perturbed `alpha` in the port fails it
at the first `MUL_MAT`.

**One false check fixed on the way.** The scripts first matched the archive
member as `'/ggml-blas\.o:'`; `nm -A` on macOS prints `libggml.a:ggml-blas.o:`,
so "the port contains no upstream object" passed for any library. With
`[:/]` it refuses the upstream build, shown by pointing `blas-diff` at it.

## Stage 5 step 6: AVX-512

Decision 53's second level, `x86_64_v4`, and `tigerlake` above it, gated
on a second Windows machine, `JWHLPTOP` -- an Intel i5-1135G7 (Tiger Lake:
AVX-512 F, BW, DQ, VL, VNNI, VBMI; no BF16, no FP16), reached through
`zupdater` like the Haswell laptop, with `X86_AGENT=JWHLPTOP`.

### The measurement

Every CPU translation unit preprocessed at `x86_64_v3`, `x86_64_v4` and
`tigerlake`, live lines compared by linemarker: **+2,123 lines at v4**,
1,954 of them the three AVX-512 gemm arms in `arch/x86/repack.cpp`. The rest
is small and enumerable -- `vec.cpp` and `vec.h` (the `__m512` `ggml_v_expf`,
SiLU, softmax, `cvar`, the bf16 dot), `ggml-cpu.c` (16-wide f16 conversion
blocks), `sgemm.cpp` (`tinyBLAS<16, __m512>`, `VECTOR_REGISTERS == 32`),
`simd-gemm.h` (block sizes) and a dead `packNibbles` arm in `quants.c` (no
caller at this pin). `tigerlake` over v4 is **five sites**, all VNNI:
`quants.c:107`, `repack.cpp:125` and `153`, `sgemm.cpp:1757`, and two
feature reports. `simd-mappings.h` shows no line difference because it is
macros -- its `__AVX512F__` arm changes every `GGML_F32_VEC` loop through
`simd.F32` and `simd.F16`.

### What was ported

- **`simd.zig`**: `GGML_F32x16` and the `GGML_F32Cx16` fallback for `F16`.
  `_mm512_reduce_add_ps` lowers to a reassociable
  `llvm.vector.reduce.fadd`, so its order is the backend's; read off the v4
  reference's `ggml_vec_dot_f32`: fold 256, fold 128, `vshufpd`,
  `vmovshdup` (`simd.reduceAdd512`). `strict_loop` for both new models was
  measured the same way as v3's: never vectorized.
- **`vec.zig`**: the AVX-512 `ggml_v_expf` is **not** the AVX2 routine at
  sixteen lanes -- it ends in `_mm512_scalef_ps` (LLVM's
  `avx512.mask.scalef.ps.512`, rounding 4), its polynomial's last term is
  `fma(c, b, 1)` rather than a product, and its slow path only separates
  `|n| > 192`. Read against the reference's IR.
- **`convert.zig`**: the 16-wide `vcvtps2ph` block, through LLVM's own
  intrinsic for the NaN payloads, as the 8-wide one.
- **`sgemm.zig`**: `tinyBLAS<16, __m512>` for f32, bf16 and f16, and
  `tinyBLAS_Q0_AVX`'s 32-register `mnpack` cases (4x4, 4x3, 3x4, 3x3) --
  a schedule, not a result: every tile computes each element alike.
- **The repack gemms, by a new layer, `repack/x86/wide.zig`.** Each AVX-512
  arm is its AVX2 arm on two interleaved blocks at once: the same
  intrinsics in the same order (`q2_K`: 328 `shuffle_epi32`, 256
  `maddubs_epi16`, 192 `add_epi16`, 96 `madd_epi16` in *each* arm), every
  one confined to a 128-bit lane or its own element. Only how a register is
  filled differs: columns, scales and mins as
  `_mm512_inserti32x8(cast(block x), block x + 1, 1)`, activation rows and
  their scales repeated. So `q4_b32.zig`, `q4_k.zig` and `q2_k.zig` now run
  one body generic over `Wide(256)` and `Wide(512)`, with `join`/`rep` at
  the inputs, and each AVX-512 lane is computed exactly as the AVX2 arm
  computes that column. The 512-bit sign trick (`abs` and a masked
  `0 - y`) differs from `sign_epi8` only where `x == 0`, whose product is
  zero either way.
- **VNNI** (`tigerlake`): `vpdpbusd` in `mul_sum_us8_pairs_float`, both
  `mul_sum_us8_pairs_acc` widths and `sgemm`'s `updot`. `__AVXVNNI__`
  alone, `__AVXVNNIINT8__`, `__AVX512BF16__` and `__AVX512FP16__` are
  compile errors: no CPU here has them.

### Two things the gates would not have said

- **The v4 build compiled before any of the repack work.** The three gemm
  files said in prose that their AVX-512 arms were not ported, with no
  guard, so a v4 build took the AVX2 arm where the C takes the 512-bit one.
  It was also bit-identical -- the lane argument above -- so every gate
  passed it. Bit-exactness was never in question; 512-bit code was, and
  the disassembly is what shows it now (`zmm` in all three gemms,
  `vscalefps` in softmax).
- **`repack-diff` ran every gemm at sixteen columns only**, which reaches
  the AVX-512 loop and never its hand-off: `if (anc != nc) { xstart =
  anc/8; y = 0; }` sends the leftover columns to the AVX2 loop **from the
  first row again**. Twenty-four columns now run beside sixteen (60 checks
  to 116), and the injection that drops `y = 0` fails 2 of them.

### The gates

All on `JWHLPTOP`, both sides built for the level:

| gate | `x86_64_v4` | `tigerlake` |
|---|---|---|
| `x86-quants-diff` | 28 / 28 | 28 / 28 |
| `x86-repack-diff` | 116 / 116 | 116 / 116 |
| `x86-ops-diff` | 712 / 712 | 712 / 712 |
| `x86-node-diff` (Qwen3.5 2B `Q4_K_M`, 1 and 4 threads, fused) | 5,505 | 5,505 |

and on the Haswell laptop at v3 after the refactor, `x86-repack-diff`
116 / 116 and `x86-ops-diff` 712 / 712. The model is at
`%USERPROFILE%\llamazig\` there: `X86_MODEL='llamazig\Qwen3.5-2B-Q4_K_M.gguf'`.

### Injected faults

| fault | level | gate | result |
|---|---|---|---|
| the two blocks' column scales swapped (`q4_b32`) | v4 | repack | 12 / 116 |
| `y = 0` dropped after the 512-bit loop (`q4_K`) | v4 | repack | 2 / 116 |
| block x's mins in both halves (`q2_K`) | v4 | repack | 4 / 116 |
| `reduceAdd512` pairs `(0+1),(2+3)` | v4 | ops | 60 / 712 |
| `vExpf512`'s `fma(c, b, 1)` unfused | v4 | ops | 20 / 712 |
| `sgemm`'s 512-bit `hsum` ordered | v4 | ops | 12 / 712 |
| `vec_dot_bf16`'s two reductions merged | v4 | ops | 2 / 712 |
| `vpdpbusd` operands swapped, `quants` | tigerlake | quants | 5 / 28 |
| the same, 512-bit repack | tigerlake | repack | 12 / 116 |
| the same, `sgemm`'s `updot` | tigerlake | ops | 18 / 712 |

10 / 10. The VNNI ones matter because a VNNI arm and the `maddubs` arm it
replaces agree on every input these kernels see -- `|x| <= 128` against a
signed byte never saturates a pair -- so a pass alone could not show the
arm was reached.

## Upstream's other tools

Decision 61. Each tool is a Zig binary under `tools/`, `zig build tools`,
named as upstream names it; upstream's own build over `libllama-ref` and our
`libggml` is `zig build tools-ref` (`build/tools.zig`), so between a port and
its reference only the tool's code differs. `scripts/tools-diff <tool>` runs
both on every line of `harness/tools/<tool>.cases`, each side in its own
scratch directory and as `llama-<tool>` (argv[0] reaches the usage text),
and requires stdout, stderr, the exit code and the hash of every file in the
directory to be the same after every case. `!mask` lines drop what moves run
to run -- timings, Metal's pointer addresses, `common_log`'s timestamps; a
`~` case compares stderr's lines in any order, for the one place upstream is
not deterministic (below); `harness/tools/<tool>.setup` writes input files.

### `tools/common/`

What the tools share, ported from upstream's `common/`: `common_params_parse`
with its option table (`arg.zig`, `arg_options.zig`, `arg_options2.zig`) --
the 161 options the ported examples admit, out of 349, each with its
spellings, hints, help text built from the tool's defaults, examples and
environment variable; `common_log` (`log.zig`); `common_init`, the token and
batch helpers and the params converters (`common.zig`); the imatrix loader;
`common_cpu_get_num_math` (`cpu.zig`); `string_parse_kv_override`.

Three things about it were found by the gate, not by reading:

- **`set_env` appends `(env: NAME)` to the help text.** Every option with
  an environment variable shows it in `--help`; the first port left it out.
- **`llama_supports_rpc()` has a side effect.** `--rpc` is added only when an
  RPC backend exists, which it never does here, so the port first skipped
  the question. But asking initialises the backend registry: upstream's
  Metal device logs happen while the table is built, at the verbosity in
  force *before* the command line is read, and are filtered out; the port's
  happened later and printed at `-lv 4`.
- **`common_log` starts its clock in `common_init`**, so the first message
  already has a non-zero timestamp; started at the first message, the port
  printed none on it.

`common_log` writes on a worker thread upstream, so a `LOG_ERR` and a direct
`fprintf(stderr, ...)` after it reach stderr in either order -- upstream
differs from itself there. The port writes synchronously, in the order the
C issues the calls.

Refused, with a message: the model downloader and cache (`-hf`, `-mu`,
`-dr`, `-cl`), the INI config files, `--completion-bash`, JSON-schema
grammars (`-j`, `-jf`). Their help text is ported, so `--help` is upstream's.

### Results

| tool | cases | injected faults caught |
|---|---:|---|
| `gguf-split` | 29 | 2/2 -- a padding byte, the merge's metadata rewrite |
| `quantize` | 40 | 3/3 -- ftype by number, `--tensor-type` case, derived name |
| `tokenize` | 43 | 4/4 -- help wrapping, a `stoi` message, log verbosity, a section |
| `fit-params` | 13 | (the fitting is `fit-diff`'s) |
| `export-lora` | 18 | 2/2 -- `std::map` order, the LoRA scale |
| `batched-bench` | 11 | 1/1 -- the shared-prompt KV size |
| `cvector-generator` | 13 | 2/3 -- `minstd_rand0`'s multiplier, the mean's row count; the escape is below |
| `imatrix` | 35 | 4/4 -- the accumulation unfused, the legacy `ncall` floored, the statistics' tie-break, a per-expert count |
| `perplexity` | 31 | 4/5 -- the 16-bit log-prob offset, `uniform_int_distribution`'s bit count, Winogrande's trailing threshold, one output too many; the fifth is below the printed precision |
| `llama-bench` | 42 | 4/4 -- `\|` read as `+` in a range, a column width, `>> bool` on a non-number, the tensor-split precision |

`gguf-split` has one case left out: `--split-max-size xM` makes `sscanf`
convert nothing, and the C then reads an uninitialised `n`.

**`cvector-generator`'s PCA starts from `std::time(0)`**, through
`std::default_random_engine` -- libc++'s `minstd_rand`, multiplier 48271 --
and `uniform_real_distribution<float>`, which is `generate_canonical<float,
24>` over one draw divided by 2^31. Ported exactly, it would still differ
from upstream whenever a second boundary fell between the two runs, so
`harness/tools/fixed_time.c` pins `time` for both sides (a `__DATA,__interpose`
entry on macOS, a plain definition under `LD_PRELOAD` elsewhere) and
`tools-diff` preloads it on a `!preload` line. PCA then compares on bits,
the written vector included. It needs `common_init_from_params`
(`tools/common/init.zig`), the sampler chain's construction among it.

- **Every `GGML_ASSERT` in the tool is kept in ReleaseFast.** The first port
  wrote them as `std.debug.assert`, which compiles out: on Qwen3.5 upstream
  aborts on `diff_filtered.size() == n_layers - 1` -- a hybrid model's
  recurrent layers do not all name an `l_out` -- and the port carried on and
  wrote a file.
- **The arch fixtures have no tokenizer**, so the cases run on `$PLAIN`,
  Qwen3-0.6B and Qwen3.5-2B; one fixture case checks both sides abort alike.
- **Two cases are undefined upstream and left out.** With no prompt pairs
  -- files of blank lines, or files of different lengths, whose error
  `prepare_entries`'s caller ignores -- the diff has zero rows and
  `print_debug_tensor` reads past a `malloc(0)`. And a model that fails to
  load is dereferenced as null, which kills the process before the async
  logger writes what it was given.
- **The escape**: `filter_nonzero_rows` counts a row as zero unless an
  element is *greater than* 1e-6, so a row of negatives is dropped; testing
  `|x|` instead passes. It differs only on a hidden-state row with no
  element above 1e-6 and one below -1e-6. Rows from a shared prompt prefix
  diff to exactly zero, and real rows mix signs, so no real model input
  reaches it. Not provably inert, so it is recorded as an escape.

**`imatrix` walks a `std::unordered_map`** -- `m_stats` -- in both writers
and in `--show-statistics`, which pushes the per-tensor rows in its order and
then sorts them unstably. It is `libcxx.StringMap` and `libcxx.sort`, which
is why `src/llama/module.zig` now exports `libcxx`. What else the gate
found or needed:

- **No local model with a vocabulary has experts**, and the arch fixtures
  are `no_vocab`, so `MUL_MAT_ID`'s per-expert counts had no text to run
  on. `harness/tools/add_vocab.c` rewrites a fixture's 128 tokens as an SPM
  vocabulary -- printable ASCII, a newline byte token -- and the MoE cases
  read an ASCII-only text; one case keeps a non-ASCII text, where both
  sides throw from the vocabulary's byte fallback.
- **An empty `std::vector`'s `data()` is null**, and libllama logs the
  adapter pointer at `-lv 5`: `common_set_adapter_lora` passed `0x0`
  upstream and an empty slice's dangling `0x8` in the port.
- **An exception escaping libllama is reported differently, library-wide.**
  `impl.zig`'s `Exception.terminate` goes through `ggml_abort` -- a
  `llama-vocab.cpp:4418:` prefix and a backtrace -- where libc++abi prints
  `libc++abi: terminating due to uncaught exception of type ...` and no
  backtrace. The case masks the prefix; changing the convention is a
  decision about every gate that compares an abort, not about this tool.
- **`std::inner_product` over floats is the strict-loop shape** of
  `simd.zig` -- rounded products in whole groups, the rest fused -- in
  `compute_statistics` and `compute_cossim`.
- **The perplexity's partial sums add in scheduling order upstream**
  (`process_logits`); the port adds them in thread order, below the four
  decimals printed. A `--model` is required even to merge files, and a
  repeated `--in-file` keeps only the last: the merge cases use
  comma-separated lists.

**`perplexity` draws HellaSwag's tasks with
`std::uniform_int_distribution<size_t>` over `std::mt19937(1)`**, so the
port carries libc++'s algorithm for it -- `__independent_bits_engine`
drawing `w` bits and rejecting past the range, read from the bundled
`<__random/uniform_int_distribution.h>` -- as `uniformIndex`. Dropping the
bit count's round-up changed which tasks ran. Its other three modes read
files in formats of their own, which `harness/tools/perplexity.setup`
writes: six-line HellaSwag tasks, a Winogrande CSV with malformed rows and
a last line without its newline (which upstream drops), and the
multiple-choice binary layout. The KL divergence runs `$PLAIN`'s fine-tuned
sibling against the base logits `$PLAIN` saved, enough tokens for the
statistics block.

- **Winogrande's `skip_choice` threshold escaped at first**: every row had
  many tokens after the blank, so `> 3` and `> 2` agreed. Rows with the
  blank one to four words from the end take the other arm, and the
  injection fails.
- **The escape**: computing the KLD `percentile` interpolation unfused
  instead of fused moves a float by an ULP, printed to six decimals. The
  gate compares what the tool prints, so a difference below that is
  invisible to it by construction.
- **Every contraction is named**: `fma` where clang fuses a same-type
  `a*b + c` (`scale*q + min`, the Wilson interval, `sum2/n - f*f`, the
  uncertainties' `a*a + b*b - 2*cov`), none where a `float` product meets a
  `double` sum.

**`llama-bench` has a command line of its own**, not `common_params_parse`,
and three of its parsers are standard-library behaviour more than code.
`parse_int_range` is a backtracking `std::regex`, matched here by hand: each
attempt takes the longest form followed by `,` or the end, and the operator
class `[\+|\*]` admits `|`, which the loop then rejects as an invalid range
format. Booleans go through `string_split<bool>`, `std::istringstream >>
bool` without `boolalpha`: a `long`, where 0 is false and every other number
is true -- and so is anything that is no number at all, which the C++ reads
as 0 and so as false. `-ts` splits with `std::sregex_token_iterator` over
`[;/]+`. Its `-ot` loop breaks out of the group loop only, so an invalid
override lets the argument loop go on and the error names the last argument
read. What it measures is time, so the masks keep each row's parameters
and drop the speeds, the test time and the samples, in all five formats; it
decodes random tokens, so the arch fixtures run without a vocabulary. `-hf`
is not a case: upstream downloads, and on the reference build fails on its
missing HTTPS; no downloader is ported, and the port prints only the final
error.

## Stage 8: libmtmd and `llama-tts`

`tools/mtmd/` is upstream's multimodal library: the CLIP-style vision and
audio encoders (`clip.cpp` and 43 graph files under `models/`), the image
and audio preprocessors, the public `mtmd.h` API that turns a prompt with
media markers into text and embedding chunks, and the helpers -- loading a
bitmap from a file, evaluating chunks into a llama context, and the two
text-to-speech pipelines `llama-tts` drives. 17,399 lines of C++ outside
the graphs and 8,292 in them, plus the stb_image, miniaudio and SHA-256 code
`mtmd-helper.cpp` compiles in. The port is `src/mtmd/`: 16,262 lines, 8,847
of them graphs, and 6,085 in `vendor/`. `PLAN.md`, Decision 62.

### How it was split

`clip.cpp` is one 6,000-line file built around four `switch`es over the 60
projector types -- `load_hparams`, `load_tensors`, `clip_encode`'s inputs
and `clip_n_output_tokens` -- and a class hierarchy of graph builders. The
core (`clip.zig`: the context, the loader, encode) was written first with
each switch behind a stub of fixed signature, so the four switches
(`arms_hparams.zig`, `arms_tensors.zig`, `arms_inputs.zig`) and three
groups of graphs could be ported in parallel without touching one file
twice. The hierarchy flattens: `graph.Graph` is the `clip_graph` base, each
model file exposes `build(g, params)`, a virtual `build_mm` override is a
function pointer (`mm_hook`), and `support_batch` is a table.
`std::function` arguments are a function pointer and a context. A throw is
`clip.throwErr`, which records `what()` and returns `error.Runtime`; the C
API's `catch` blocks log it exactly as upstream does.

`mtmd.zig`, `helper.zig` and `helper_gen.zig` hold every `mtmd_*` and
`mtmd_helper_*` function as a `pub fn ... callconv(.c)` with no export;
`exports.zig` exports all 82, and `harness_root.zig` re-exports the port
without them -- libllama's pattern. libmtmd reaches libllama through the
port's `api`, never the C symbols, so the same code runs beside upstream's
libmtmd and `libllama-ref.a` in one process. `mtmd.h`'s structs are declared
by hand (a second import of `llama.h` would give a second `*llama_model`),
and `mtmd-diff` checks each against the C at compile time, field by field.

`llama_model_get_tok_embd` was missing from libllama's port: it is in
`llama-ext.h`, C++ linkage, so the 240-symbol `nm` comparison could not see
it, and nothing called it before `mtmd-helper-gen.cpp`. It is now in
`api/model.zig`, unexported like the other eleven `llama-ext.h` entries.

### The vendored decoders

stb_image and miniaudio are single-header C libraries upstream compiles
into `mtmd-helper.cpp`. Linking them as C would put C back into the
binary; they are ported to Zig in `src/mtmd/vendor/`, as far as
`mtmd-helper.cpp` reaches them -- `stbi_load_from_memory` with every default
format, and `ma_decoder` over WAV, MP3 and FLAC with its mono mixdown,
linear resampler and filter -- and gated by `make mtmd-decode-diff`, which
compiles the headers unmodified, as C++17 with the helper's defines, and
diffs on bits over synthesized, encoder-made (`sips`, ImageIO, `afconvert`),
truncated and corrupted media. The SHA-256 bitmap id is `std.crypto`'s,
checked against `vendor/hash` by `mtmd-diff`'s bitmap ids.

Four things shaped it:

- **miniaudio tries each backend twice.** `ma_decoder_init_memory` runs
  every decoder over its own memory stream, whose out-of-range seek fails,
  and when all fail retries them through the decoder's callbacks, whose
  seek clamps -- a file one rejects the other can accept. `audio_io.zig` is
  one stream type with both modes, and the read loops are ported as loops:
  MS-ADPCM yields more frames on a second call.
- **dr_mp3 on aarch64 is NEON**, and `vmlaq_f32` fuses where
  `vaddq(vmulq)` does not; the decoder struct mirrors the C's layout,
  because the Huffman reader over-reads into it.
- **The whole-file diff could only say which sample differed.** A
  frame-by-frame diff of the decoder state found the one real bug, a
  `u1 + u1` overflow in the MP3 scalefactor partition lookup that changed
  decoded samples.
- **Two injections escaped until sweeps were added** -- an unfused term in
  the MP3 synthesis and `<` for `<=` in the resampled length -- and a
  synthesis-rounding sweep and a resampled-length sweep now catch both.
  9 of 10 caught; the tenth, 24-bit WAV scaling done in `f32`, cannot change
  a value.

The vendored code's own undefined behaviour is recorded, not reproduced:
a truncated PIC makes stb pass a null buffer to `stbi__convert_format`,
and at the helper's `req_comp` of 3 **upstream crashes** -- the harness runs
those in a forked child and the port returns a failure; stb leaves
uninitialized memory a corrupt file can expose (JPEG component planes, TGA
rows, BMP palette entries past the parsed size, the HDR flat loop's pixel on
a short read), where the reference is built over `calloc` and the port
zeroes, and those cases compare outcome and size but not pixels; dr_mp3 can
read past its main-data buffer and its Xing/LAME parse past the input; and a
WAV claiming an enormous frame count makes upstream's `vector::resize`
throw, where the port returns false. With FLAC, 999 files and their
damaged copies: 131,726 image checks and 542,841 audio checks, 0 differing,
ReleaseFast and ReleaseSafe; 772 inputs crash the reference (run in a forked
child) and 482 are unspecified (compared on outcome and size).

**FLAC came last, native and Ogg.** Ogg-FLAC is reachable from the helper
although its sniff looks for `fLaC`: the sniff also accepts an ID3 prefix,
dr_wav refuses the file, and dr_flac skips the tag and accepts `OggS`. The
gate's FLAC corpus is `afconvert`'s (16- and 24-bit, mono to eight
channels, at rates that resample) and a FLAC/Ogg encoder in the harness that
writes every subframe type, LPC to order 32, both rice codings, escapes,
wasted bits, all four stereo assignments, 4- to 32-bit samples, odd and
variable block sizes, every sample-rate code, seek tables, and on the Ogg
side random page splits, a foreign stream and a bad-CRC page: 21,564 checks,
15,557 decoded on both sides. dr_flac's NEON residual decoder is
integer-equal to the scalar one, but its stereo converters are not: at 24
bits or fewer they convert at a 24-bit scale times 2^-23, and the port
follows `bitsPerSample` as the C does. 14 injections, 12 caught, two
provably inert -- a logical for an arithmetic `>> 1` before a left shift, and
the mid-side tail for its body, where mid's low bit is side's. The 32/64-bit
prediction-threshold fault escaped twice: it needs full-scale coefficients
and a nonzero shift, or the 64-bit sum truncates to exactly the wrapped
32-bit one. A failed subframe decode is ignored upstream, so a frame whose
CRC matches by chance plays stale buffer contents; the reference is built
over `calloc` and the port zeroes. `u32 << 32` in the converters above 24
bits is unspecified and not compared.

### `llama-tts` and `llama-mtmd-cli`

Zig binaries under `tools/` over libmtmd's port, gated by `scripts/tools-diff`
against upstream's own, built unmodified over `libmtmd-ref` and
`libllama-ref`: stdout, stderr, the exit code and every file written. `tts`,
28 cases: Qwen3-TTS and pocket-tts, speaker WAVs at 16 and 44.1 kHz, CPU
and GPU, the error paths, a strong repeat penalty and the uncapped
512-frame run, the WAV hashed. `mtmd-cli`, 39: eight vision projectors,
M-RoPE among them, two audio ones and one with both, five legacy chat
templates, and the conversation loop with line editing. `tools/common/`
gained the full `common_sampler` (`sampling.zig`), the console's line
reader (`console.zig`, POSIX; Windows falls back to plain lines) and the
non-Jinja chat templates (`chat.zig`); `--jinja` and `MTMD_DEBUG_GRAPH` are
refused.

**The fixture models answer much the same whatever they are asked**, so
output alone could not see a misformatted prompt: four injected faults
escaped until cases logged the formatted prompt (`-lv 5`) and token counts
(`--perf`). **The REPL is gated under `!tty`**: upstream echoes typed input
to its terminal while its log reaches stdout from a worker thread, so where
the echo lands in stdout is upstream's own race; the echo is not compared,
and editing is checked through the prompt it produces. 15 injections, 12
caught; the escape is vicuna's antiprompt, which a random model never
produces, and two cannot change any output. Its one fix outside the tools
was in libllama: a `GGML_ASSERT`'s line, `llama-context.cpp:1706` for
1711, which made six abort cases differ.

**That line was not alone.** A sweep matching every hand-written
`file, line` of an assertion or abort under `src/` and `tools/` against the
upstream line holding the same expression found 79 wrong -- most a few
lines off, a run of `context.zig`'s 21 lines off and `memory_recurrent.zig`'s
14, one in `clip-impl.h` pointing at a `#define` -- and corrected them. An
abort's message names that line, and a gate compares it only when the
abort fires, so each was waiting for the case that reached it.

### The fixtures

No mmproj file exists here and upstream ships no generator.
`harness/mmproj_fixtures.cpp` writes one small random mmproj per projector
type from the keys and shapes `clip.cpp` reads, and pairs each with an arch
fixture whose `no_vocab` tokenizer is rewritten as SPM with the special
tokens `mtmd.cpp` and `mtmd-helper-gen.cpp` look up. `make mmproj-fixtures`
checks every one through upstream's libmtmd on the CPU and Metal: 59
fixtures cover all 60 types, 118/118 runs, and `gemma3na` -- declared, never
loadable -- is checked to fail. Seven types needed a workaround for an
inconsistency in upstream itself (voxtral's stack factor, step3vl's
hard-coded 504-pixel crops, hunyuanvl's M-RoPE assertion, gemma3nv's square
projection, the SAM block count deepseekocr ignores, cogvlm's null fused
qkv, lfm2a's position table); each is in the harness's comments. The check
is a precondition, not a gate: it proves the reference runs a fixture, not
that the port matches it.

### `make mtmd-diff`

One process per side, `ref` or `port`, each linking both libraries and
calling one through a table of C function pointers of identical type, so
both run the same script. The transcript is everything that crosses the C
boundary: every log line of libmtmd, its helpers, libllama and ggml; with
`nodes`, every encoder graph node (`cb_eval`: name, op, type, shape, a hash
of its data); the capability getters; every chunk `mtmd_tokenize` makes --
type, tokens, counts, ids, every decoder position, the serialized bytes,
the load of a truncated buffer; every embedding on bits through
`mtmd_encode_chunk`, a copy, a placeholder and the batch API; the logits
after `mtmd_helper_eval_chunks`; a real JPEG (`tools/mtmd/test-1.jpeg`)
and a stereo 22,050 Hz WAV through `mtmd_helper_bitmap_init_from_file`; and
for the two TTS fixtures, the generation helper's hidden state at every
step and the output WAV. Every fixture runs on the CPU and the GPU, with
nodes hashed and without -- `cb_eval` splits the graph at every node, so
only the second pass sees the scheduler fuse. An abort counts as a match
when both sides abort with the same message; the backtrace, whose threads
interleave differently run to run, is dropped.

**236 of 236 identical** (59 fixtures, CPU and Metal, per node and fused),
and **118 of 118 with the harness built ReleaseSafe** on the CPU, so the
port's code runs with Zig's safety checks on. Two of the 236 are an abort on
both sides: upstream's gemma4v CPU path fails
`GGML_ASSERT(i01 >= 0 && i01 < ne01)` in a `GET_ROWS` on the JPEG's
304x224 slice, reading past its position table. Metal does not check the
index. Before the two counted as a match, the first full run read 234 of
236: the abort's backtrace interleaves its threads differently run to run.

**Its first version read the wrong library.** Two lines of the harness
called `c.llama_model_n_embd_inp` -- the C symbol, `libllama-ref`'s -- on the
port's model, so the port side hashed embeddings over a garbage length and
every encode differed in its last float while every graph node matched.
The function table covers libllama now, and a node-level match with an
output mismatch is the signature to look for.

Injected faults, 10 of 12 caught; the two escapes changed no hash:

| Fault | Fixture | Caught |
|---|---|---|
| mrope decoder `x` taken from the row | qwen2vl | yes |
| `image_idx` serialized after `n_temporal_merge` | mlp | yes |
| an audio chunk one token long | ultravox | yes |
| a row-end token after every row | idefics3 | yes |
| `img_beg` tokenized without special tokens | gemma3 | yes, after the fixture change below |
| helper M-RoPE 2-d positions with `x` and `y` swapped | qwen2vl | yes |
| helper positions `pos_0 + 2i` | pockettts | yes |
| qwen3tts without the `tts_pad` overlay | qwen3tts | yes |
| attention scale from `d_head + 1` | gemma3 (Metal) | yes |
| `gelu` as `gelu_quick` | gemma3 (Metal) | yes |
| helper positions `pos_0 + i + 1` | pockettts | no: a uniform shift changes no hash |
| `clip_encode` running one thread | mlp | no: no hash on that fixture depends on the CPU's thread count |

**The marker injection escaped at first, and that was the fixtures.** Their
128-token vocabularies -- the size is the arch fixtures', tied to the
embedding tensors -- had no room for the begin and end markers `mtmd.cpp`
wraps media in, so `<start_of_image>` tokenized as characters with special
parsing on or off. The two free slots now hold gemma3's markers: one
fixture where the marker is a token, as it is in a real model, and the
injection fails it. Every other marker still tokenizes as characters.

**The injection runner once left a fault in the tree.** It restored by
finding the replacement text, which had to be unique; `ggml_gelu_quick(ctx0,
cur)` also occurs in the genuine `gelu_quick` arm, the restore refused, and
`gelu` stayed `gelu_quick` through four later injections and two full runs
-- which therefore failed, and three of those injections read as caught
because of it. It restores by byte offset now and checks every site
afterwards, and the results above are from a clean baseline of 236/236.

**`make mtmd-exports`**: the port's library root compiled with its `llama`
import pointed at libllama's harness root, so only libmtmd's own exports
are listed, against `libmtmd-ref.a`'s `mtmd_*`: 82/82. Hiding one fails it.

### The preprocessor gates

`make mtmd-image-diff` and `make mtmd-audio-diff` compile `mtmd-image.cpp`
and `mtmd-audio.cpp` unmodified beside a stand-in for `clip_ctx`, so the
preprocessors run without a model, and compare every output float on bits
and every log line. Images: 2,536 cases, 8 of 9 injections caught -- the
escape, replacing libc++'s `std::sort` of the target ratios with a stable
sort, needs an aspect exactly midway between two equal-area tilings in a
set of 30 or more, which none is. lfm2's `nearbyint` is round-half-to-even
and only sizes sitting on its tiling threshold see it. Audio: 3,309 cases,
260M checks, 11 of 13 caught; the zero-frame tail branch is dead code, and
the variance loop's fusion reaches a float only through a square root and a
division, so it is held to the reference's disassembly instead: products
rounded in groups of four, the last `t % 4` fused. On the Mac the release
reference folds the sine table into `__sincosf_stret` and `pow(10, x)` into
`__exp10`; the port keys both on Darwin, unmeasured on Linux.

### Upstream's own defects, found on the way

Recorded, reproduced where defined, never fixed:

- granite4_vision's hparams error passes two `size_t` to `"%s ... %d %d"`
  with no `__func__`; mimo_audio's missing-tensor error has one `%s` more
  than its arguments.
- gemma4a, lfm2a, parakeet and granite_speech index `model.layers[il]`
  relying on a shared loop to have sized it.
- Both MiniCPM-V input arms write `int[1024]` bucket arrays with no bounds
  check.
- `select_best_resolution` returns an uninitialized size when no candidate
  is taken; the port returns 0x0.
- The conformer asserts below two hops of audio; qwen3tts's speaker
  preprocessor reads before its buffer below its pad; gemma4a trims `n_len`
  and keeps the old row stride (replicated); whisper asserts unless a hop is
  a hundredth of a second.
- `mtmd_context`'s constructor leaves `pos_type` indeterminate without a
  text model, and a throw from it leaks the clip contexts: its destructor
  never runs and the pointers are raw. The port leaks them too, so freeing
  them logs nothing upstream does not.
- `mtmd_get_cap_from_file` catches the loader's exception and logs it; an
  earlier draft of `clip.getCap` aborted instead.

## llama.xcframework

Decision 63: `make xcframework` builds a drop-in for the `llama.xcframework`
upstream's `build-xcframework.sh` writes, in place of the static
`llamazig.xcframework`. `xcfwk/` is `xcframework/` now, after the Makefile
target. The reference is upstream's own: `make -C xcframework upstream`
runs its script, with CMake from `.tools/`, over a `git archive` of the pin,
so nothing is written under `llama.cpp/`. Built here it took four minutes.

**What upstream's is, measured from that build.** Seven libraries --
`ios-arm64`, `ios-arm64_x86_64-simulator`, `macos-arm64_x86_64`,
`tvos-arm64`, `tvos-arm64_x86_64-simulator`, `xros-arm64`,
`xros-arm64_x86_64-simulator` -- each a dynamic framework, install name
`@rpath/llama.framework/llama` (`.../Versions/Current/llama` on macOS),
with a dSYM; minimums iOS 16.4, macOS 13.3, tvOS 16.4, visionOS 1.0, the
device binaries re-marked by `vtool` so their SDK field equals their
minimum; eleven headers, `mtmd.h`, `mtmd-helper.h` and `ggml-blas.h` among
them; a module map that links `c++`, Accelerate, Metal and Foundation and
not the framework itself; 4,710 exported symbols, 993 of them functions the
headers declare. The binary is libllama, libggml with the BLAS backend
(`GGML_BLAS_DEFAULT` is on for Apple) and libmtmd, linked against
Foundation, Metal, Accelerate and libc++.

**Ours.** `bundle.sh` transcribes upstream's `setup_framework_structure`,
so headers, module map and `Info.plist` are upstream's bytes. Each slice
architecture is `zig build xcframework -Dblas`, a new step: `libggml.a`
and `libmtmd.a`, the mtmd module's own library, which carries libllama
because the module imports it -- `libllama.a` beside it would define every
`llama_*` twice. Those and Zig's compiler_rt are prelinked with `ld -r` to
`exports.txt`'s prefixes, linked `-dynamiclib` with upstream's install
name, `lipo`-joined, `vtool`-marked on devices, `dsymutil`-ed and stripped,
exactly the order upstream's script uses. The toolchain is `../zig17`,
which finds every Apple SDK and links for the simulators; with it the
shim SDK the static build needed for the simulator is gone.

**Which CPU each slice is for** was the decision that mattered. Upstream's
CMake names none (`GGML_NATIVE` is off), so Apple clang picks its default
per target, measured with `clang -###`: `apple-m1` for macOS and every
arm64 simulator, `apple-a7` for iOS and tvOS devices, `apple-a12` for
visionOS, `penryn`/`core2` for x86_64. A slice built for a newer CPU
than upstream's would not run everywhere upstream's does -- the static
framework's `apple_a14` and `apple_m3` excluded the iPhone 8 through 11
and every M1 -- so ours takes the same arm64 CPUs. x86_64 is
`x86_64_v3`, the port's x86 floor; every Intel Mac macOS 13.3 supports has
AVX2.

**Upstream's universal slices compile ggml-cpu generic.** Its macOS and
simulator builds set `CMAKE_OSX_ARCHITECTURES="arm64;x86_64"`, and
`ggml_get_system_arch` (`ggml/cmake/common.cmake:28`) recognises only
`arm64` or `x86_64` exactly: both builds log `GGML_SYSTEM_ARCH: UNKNOWN`
and "Unknown CPU architecture", and define `GGML_CPU_GENERIC`, so
`arch-fallback.h` renames every `_generic` kernel to the real name. The
device builds log `ARM detected`. Ours does not reproduce that: our macOS
and simulator slices run the port's NEON kernels at `apple_m1`, the
baseline every gate runs. So on the macOS CPU path upstream's framework
and ours differ -- 2 of 6 prompts, which `dropin-macos` reports and does
not gate -- and on Metal they are identical.

**Gating `apple_a7` found the port wrong there.** The A7 has neither
`fullfp16` nor `dotprod`, so its kernel arms are `generic`'s, which Stage 5
step 2 made exact -- but `ops-diff` at `GATE_CPU=apple_a7` failed on
`conv_3d_direct`, `ssm_conv` and a Mamba-1 `ssm_scan`, and `node-diff` and
`parity-port --cpu` failed after them, while `repack-diff` passed; the same
at `apple_a12`. Those are the `simd.strict_loop` sites, which took every
`apple_*` model to have the A14's vectorizer shape. The reference's `ops.o`
at `apple_a7` and at `apple_a12` disagrees: two `fmul.4s` per body, an
8-iteration guard and a 32-byte overlap guard -- `generic`'s shape. LLVM
gives every Apple core the same cost model and sets
`FeatureMaxInterleaveFactor4` only on the A14, A15, A16, A17, M4 and M5
tunes (`AArch64Processors.td`); the M1, M2 and M3 alias the A14, A15 and
A16. `strict_loop` now keys Apple cores on that feature, and
`features.golden_generic` keys on `fullfp16`, `dotprod` and it rather than
on the model's name, so the unit tests at `apple_a7` read the `generic`
goldens and pass. After: `ops-diff`, `node-diff` and `parity-port --cpu`
pass at both CPUs.

**One link failure Zig owns.** The x86_64 simulator slices failed on
`_readdir$INODE64`: Zig 0.17's `std.c.readdir` binds the `$INODE64` name
for every x86_64 Darwin target, and only macOS's libSystem has it -- the
simulator SDKs are 64-bit-inode only (`__DARWIN_ONLY_64_BIT_INO_T` in
their `sys/cdefs.h`) and export plain `_readdir`. `backend_reg.zig` binds
`readdir` itself there.

**Exports.** Every header-declared function upstream's binary exports is
exported by ours, per slice and architecture. Upstream's also exports 203
C symbols no shipped header declares; the 98 of them from
`ggml-quants.c` -- `quantize_*`, `dequantize_*`, `iq2xs_*`, `iq3xs_*` --
are the port's too and are exported, and the rest (`stbi_*`, `XXH*`,
`sha256_*`, `unicode_*`) are C and C++ internals the port implements in
Zig under other names. Upstream's module map does not link the framework,
so a `swiftc` command line names `-framework llama`, as Xcode does when
the framework is in the target.

**The gates, run.** `make -C xcframework verify`: 7 slices, 11
architectures, each prelinked object 1,625 exports, all in `exports.txt`,
compiler_rt's math inside, no libc++ anywhere; `stub.m` links on every
target with `-framework llama` alone; dSYM UUIDs match. `compat` against
`make upstream`'s build: every slice's bundle tree identical, 91 files
byte-identical, every binary's architectures, type, install name,
platform, minimum OS and SDK identical, no dependency upstream's lacks,
and 993 header-declared exports present in each of the 11 slice
architectures. `negative`: 13 of 13 injected faults caught -- the nine
structural ones and four for `compat`: a header edited, the module
renamed, `mtmd_*` unexported on one architecture, a device binary not
re-marked by `vtool`. `smoke-macos` 6 / 6 on Metal and 6 / 6 on the CPU
against the port's `apple_m1` libraries, with app C++ beside it;
`smoke-sim` 6 / 6 on the iOS, tvOS and visionOS simulators each.
**`dropin-macos`**: one Swift binary linked against upstream's framework
runs on ours with no relink -- Metal 6 / 6 identical to upstream's own
framework and to the port, the CPU 6 / 6 identical to the port and 4 / 6
to upstream's generic kernels. **On the iPhone 15** (`smoke-device`): 12 /
12 byte-identical, CPU and Metal, against the port's `apple_a7` libraries.
**`swiftui`**: upstream's `examples/llama.swiftui`, untouched, builds
against ours for the simulator and the device, embeds and signs our
binary (the same UUID), and is running on the phone 8 s after launch --
checked to see a process that is there and miss one that is not.
`pkg-test` passes on macOS and the simulator.

**Two things the dynamic form needed that the static one did not.** The
test app's project had no `@executable_path/Frameworks` runpath, which
every Xcode template sets, so its first run died in dyld: `Library not
loaded: @rpath/llama.framework/llama`. And embedding signs the copy, so
"is our binary embedded" compares UUIDs, not bytes.

**The A7 CPU path holds no repacked weights.** On the phone the CPU
footprint is 118-127 MiB, where the `apple_a14` slice held 1,338 MiB:
every NEON arm of `ggml_repack_get_optimal_repack_type` asks
`ggml_cpu_has_dotprod()` (repack.cpp:4584 on), which the A7 lacks, so the
weights stay in the mapped file. Upstream's `ios-arm64` slice does the
same.

**Not run**: the x86_64 slices -- this Mac has no Rosetta and there is no
Intel Mac -- and the tvOS and visionOS devices. `PLAN.md`, "What is left".

## Upstream sync v0.3.0 → v0.4.0

The first sync (`SYNC.md`, `UPSTREAM.md`). The pin moved from `v0.3.0`
(`c1d0e7a00`, `b10621`) to `v0.4.0` (`5266f24da`, `b10809`), ggml from
0.22.0 to 0.23.0, and the port's version with it: `0.4.0`.

### What moved

`scripts/upstream-map` mapped the diff onto the citations: **456 touched**
(each re-read at v0.4.0 and re-cited), **217 hunks no citation covered**
(new code, each read and ported or ruled out of scope), **29 changed files
with no citation at all**, and **7,601 untouched citations repinned**
mechanically. `scripts/upstream-relines` moved **529** `__LINE__` literals,
the abort-message line numbers no citation carries. Every citation in
`src/`, `cli/` and `tools/` is now at `@5266f24da`; `port-links` reads
them all at that one commit.

The upstream changes ported, by area:

- **ggml**: `GGML_GLU_OP_SWIGLU_CLAMP`; sparse flash attention
  (`ggml_flash_attn_ext_set_n_kv_max`, op_params slot 4);
  `ggml_backend_graph_optimize_params` and the scheduler's `add_alloc_dep`;
  `ggml_op_alloc_size_may_expand`; the meta backend's `set_usage`; a
  batched `conv_transpose_2d`; `SET_ROWS` admitted by the backward build's
  in-place assert.
- **CPU**: `iqp.cpp`'s IQ panel GEMM (x86 AVX2 at run time; compiled and
  `port-coverage`d everywhere).
- **Metal**: the new pipelines (SSD scan, radix top-k, sparse FA vec),
  2,630 new tuning rows and the `A18_PRO` device id.
- **CUDA and Vulkan**: their host-layer changes, gated on the GB10
  (pickup).
- **libllama**: lazy tensor loading (`llama_lazy_mode`, a context map keyed
  on buffer type and laziness); per-layer `n_expert_used` and `n_ff_exp`;
  the PLE fields; `non_causal_type`; `llama_memory_hybrid_idx`; the
  `qwen4exp` architecture (147 entries now); session format versions 10
  and 3; `max_buf_size` for the quantizer.
- **libmtmd**: the `deepseek4v` projector (61 types); two new exports (84).
- **Tools**: ten new upstream flags in `cli/upstream_flags.zig`; `-lzm` and
  `-ncffn` deferred (`UPSTREAM.md`).

The version headers are now generated as upstream generates them:
`ggml-version.h`, `llama-version.h` and `build-info.cpp` come from the
`.in` templates through `addConfigHeader`, in both `build.zig`s, instead
of `-DGGML_VERSION` defines that had to be edited by hand.

### The tooling, and what it got wrong first

A sync needs tools no stage did, and each one was wrong on first use.

- **`upstream-map --repin`** replaced numbers in sequence, so a line
  citing 545 and 546 became 546 and 547; it rewrote the `@sha` of a second
  citation sharing a line; it cut a definition's extent at the `;` inside
  a `for (` header; and it skipped the second citation on a line. All four
  fixed by rewriting by position and tracking parentheses. A repinned
  citation must also have **identical text** at both commits, and an
  extent that looks wrong is reported as `SUSPECT EXTENT`; the text check
  was negative-tested.
- **`upstream-relines`** first matched only `"<file>", <N>`, and left
  every line number handed to a helper (`outOfBounds(248, ...)` and kin)
  at v0.3.0. Its `HELPERS` table now names each such helper. It is not
  idempotent: run it once per sync.
- **The generators hard-coded the pin.** Each carried `c1d0e7a00`, so
  regenerating after the move would have cited new lines at the old
  commit. They read `build/upstream.zig` now, through
  `scripts/upstream_pin.py` and, for the bash golden scripts,
  `scripts/upstream_pin.sh`.
- **`gen-hparams` dropped every member after a one-line member function**
  in `llama_hparams`, silently; v0.4.0 added one mid-struct. It skips
  them now; 174 members. `gen-kargs` emitted a pre-0.17 std call.
- **`tuning-diff` stopped sweeping at `M5_ULTRA`**, so `A18_PRO`, new at
  v0.4.0, was never compared. It sweeps 64 ids now, behind a
  `static_assert`; negative-tested. `tuning_table.zig` is generated
  (`gen-tuning-table`), reproducing the v0.3.0 table exactly before it
  took the new rows.
- **Metal's `graph_optimize` signature was hidden behind an `@ptrCast`**,
  so upstream's new third parameter compiled without complaint. It is
  typed now.
- **`make validate` runs `clean`** (the styleguide's `validate: clean
  format lint build test`), which deletes `zig-out/` and with it the arch
  and mmproj fixtures. The first full gate run generated them first and
  lost them; `SYNC.md`, Step 7, runs `validate` first.

### The gates

The Mac set, run in full: 54 of 58 passed first time. The four:

- **`api-diff`** (and `validate`, which runs it) compared
  `llama_model_default_params` as raw bytes. v0.4.0's `lazy_mode` field
  opened four bytes of padding after `main_gpu`, which the C++'s
  aggregate initialiser happens to zero and a Zig struct literal does
  not. The dump now copies the fields into a zeroed struct
  (`fieldsOnly`); an injected `main_gpu = 1` still fails it.
- **`quant-diff`** died in the C++: since v0.4.0 **upstream aborts saving
  any vocab-only model** -- `add_kv_from_model` asks `hparams.n_ff_exp()`,
  now an accessor bounded by `n_layer_all`, which `load_hparams` never
  reads for vocab-only (llama-hparams.cpp:79). The port aborts on the
  same line. Vocab-only saves now go through `sameDeath` in forked
  children, as charsmap saves did; the tokenizer keys the saver writes
  are covered by the full models alone.
- **`adapter-diff`**, gpu configuration only: no combination of the
  port's six candidate copies per adapter reproduced the C++'s
  `unordered_map` order, which is keyed on heap addresses. Rerun
  unchanged, it passed, 278,152 checks. Allocator luck, not the sync;
  left as is.
- **`validate`** stopped at `api-diff`; rerun after it.

**One upstream behaviour change shows only in an abort message.** At
v0.3.0 several of `opt-llama-diff`'s weights stopped at the backward
build's "inplace operations are currently not supported", reached through
the KV cache's `ggml_set_rows`. v0.4.0 admits `SET_ROWS` there, and all
three now abort on the attention gate's `sigmoid` instead -- on both
sides.

**Every golden reproduced at v0.4.0** (`vecdot-golden`, `vec-golden`,
`quants-golden`, at the baseline and `generic`, and `iq-grids`): only the
provenance line changed, apart from the `IQ4_NL` `zeros` entry, which
reads uninitialized memory in the C and is skipped by its test. The
CMake reference those scripts link (`make buildmacos`) was rebuilt at
v0.4.0 first: it is not moved by `zig build reference`, and its libraries
dated from before the sync. `SYNC.md`'s list of what a sync rebuilds now
names it.

`arch-model-diff` 160 pass, the two `REF-ABORT`s known at v0.3.0 (PLM,
`rwkv6qwen2`); `context-diff` 2,256,287 checks; `mtmd` gates 260M checks;
`tools-diff`, `cnv-diff`, `fit-diff`, `jinja-diff`, `ops-diff`,
`repack-diff`, `node-diff` and `parity-port` at the baseline, `apple_m3`,
`generic`, `apple_a7` and `apple_a12`; `struct-layout`, `tuning-diff`
(11.3M lookups), `metal-diff`, `meta-diff`, `blas-diff`. The GB10 and
both Windows machines are pickups (`PLAN.md`, "Pickup for the v0.4.0
sync"), and must clear before v0.4.1 starts.

## The v0.4.0 GB10 pickup: CUDA and Vulkan

Items 1-3 of `PLAN.md`'s "Pickup for the v0.4.0 sync", run on the GB10.
**The port needed no change.** Every CUDA and Vulkan gate passed on the
committed Zig; the two fixes are in harnesses.

**The checkout had not moved.** `llama.cpp/` and
`llama.cpp.zmake/llama.cpp` were both still at `c1d0e7a00`, and
`zig-out/lib` at v0.3.0: `make cli` failed on the seven v0.4.0 names
the old headers lack, and `gen-arch-fixtures` failed to link against
the old `libllama-ref.a`. Pulling the sync commit moves the pin, not the
checkouts; `git submodule update llama.cpp`, the zmake checkout to the
same commit, and `zig build reference --release=fast` before anything
else.

| gate | result |
|---|---|
| `cuda-backend-diff` | 2,805 + 8,387 answers identical |
| `cuda-parity-cli` | 6 / 6 |
| `cuda-node-diff` | 2,750 nodes and 33 fused decodes identical, 1 and 4 threads |
| `cuda-ops-diff` | 15,757 / 15,757 bit-identical, 14 `NONDET`, after the fix below |
| `gen-mmq-config`, `gen-fattn-config` | output unchanged |
| `vk-push-layout` | pass |
| `vk-repro` | reproducible on all four arms, per node and fused |
| `vk-backend-diff` | every scenario identical |
| `vk-parity-cli` | 6 / 6 |
| `vk-node-diff` | identical on all four arms, per node and fused |
| `vk-ops-diff` | with the harness fix below: coopmat2 and nodot 16,205; KHR and none 16,204 + 2 `REFHANG`; no `NONDET` on any arm |
| `vk-node-diff`, `vk-parity-cli` | rerun after the `platform-libs` change below: identical on every arm; 6 / 6 |

The 14 `NONDET` (10 before the sync) are the two sources measured in
Stage 6, CUB's `CUMSUM` and the fused MMVQ `MUL_MAT_ID`; v0.4.0 added
cases of both.

**`cuda-ops-diff` segfaulted in the reference.** v0.4.0's
`test_topk_qsa` is the first whole-graph case with a `ggml_cast`, a
`CPY` whose `src[1]` is the node itself. `graph_copy_dup_tensor`
(ggml-backend.cpp:2153 @5266f24da) stores a node's copy only after
copying its sources, so
the self-reference finds the hash entry already present and returns the
still-NULL slot. Upstream's CPU, the copied side in upstream's
`test-backend-ops`, never reads `src[1]`; the reference CUDA backend,
the copied side here, calls `ggml_cuda_cpy(ctx, src[0], src[1])` and
faults. The port's `ggml_backend_graph_copy` reproduces the C, as it
must. `harness/cuda_ops_diff.c` restores the self-references after the
copy. The unfixed harness segfaults on the four `TOPK_QSA` cases alone;
the fixed one passes them bit-identical.

The crash was misplaced twice on the way, and both are worth knowing.
Under `make`, stdout goes to a file and is fully buffered, so the last
case printed was `TOP_K`, not the case that crashed: rerun with `stdbuf
-o0` before believing the last line. And the reference objects carry no
debug information, so the symbol in a backtrace is the nearest export;
here it was right, which the disassembly confirmed.

**The MLA cases now hang on the no-coopmat arm too.** `hsk=576,
hsv=512, nh=1, nr23=[20,1]`, q8_0, V a view of K, at `kv=113` and
`kv=1024` -- `REFHANG` on KHR, as before, and now on `none` as well. On
`nodot` both compute, bit-identical. And the KHR arm's
`hsk=256, kv=16384` stall is gone: that part finished 135 / 135 in one
process. Both are upstream's host layer or the driver, against itself.

**The `qwen4exp` fixture cannot reach the TOPK_QSA fusion.** Vulkan
fuses it only in the radix regime, `log2(width) + 1 >= 11`, so a width
of 1,024 or more; `test-llama-archs.cpp:289` writes `indexer_top_k = 8`
and a compress ratio of 4, so the width is at most 11 at any context.
The pickup item assumed otherwise. The fusion is reached only by
`vk-ops-diff`'s `TOPK_QSA` cases at widths 1,500 and 2,000.

`harness/node_dump.c` now takes a prompt spelled `tokens:<id>,...` as
token ids, never tokenized: the fixtures carry a `no_vocab` vocabulary,
whose tokenizer aborts. `vk-node-diff` on `qwen4exp-moe` with 34 ids:
1,140 nodes and 33 fused decodes identical on all four arms. Negative
test: the port side alone with coopmat off fails both passes.
`cuda-node-diff` on Qwen3.5 is unchanged with the new harness.

**The fused TOPK_QSA had never been compared.** `vk-ops-diff` read
4 / 4 bit-identical on `TOPK_QSA`, and `GGML_VK_PERF_LOGGER` showed what
that meant: the port ran each wide case fused (one `TOPK_QSA GET_ROWS`
dispatch), the reference unfused (`ADD`, `CONT`, `CPY`, `GET_ROWS`,
`TOP_K`), the two differed, and the reference's unfused radix `TOP_K`,
re-run against itself, differed too -- so every one was excused as
`NONDET`. The reference could not fuse for the same reason
`cuda-ops-diff` crashed: `ggml_vk_can_fuse_topk_qsa` wants a use count
of 2 on the `CPY`, its own `src[1]` included, and the copy had dropped
it. `harness/vk_ops_diff.c` restores the self-reference in both of its
copies. Then both sides fused, and still disagreed, run to run: the
fused radix top-k orders its output differently every time, the
reference against itself included. Upstream compares a top-k as a set of
indices (`test_topk_qsa::err`), and its inputs are built to have no
ties, so the harness now does the same where the bits differ: a `TOP_K`
whose rows hold the same indices in another order prints `TOPKSET` and
passes, and one with any other index fails. Negative-tested: the port's
fused dispatch given `.k = width - 1` fails all three wide cases.

### Item 4: the Linux arm64 CPU set

| gate | result |
|---|---|
| `ops-diff` | 762 identical |
| `repack-diff` | 116 identical |
| `node-diff` | 2,750 nodes and 33 fused decodes identical, 1 and 4 threads, after the fix below |
| `parity-port --cpu` | 6 / 6 |
| `host-arms-diff` | 18 / 18 |
| `quant-diff` | 142,162,171 checks, 0 failures -- its first run on Linux |

**`node-diff` stopped in `libm_check`**, before comparing anything: the
reference binary defined `expf`, `logf`, `cosf` and the rest itself,
from Zig's `compiler_rt`. v0.4.0's `ggml_backend_load_best` calls
`fs::current_path(ec)` and iterates a `directory_iterator` with an
`error_code`, which pulls libc++'s `operations.o`, which needs
`__divti3`. Zig drops `-lgcc_s` and searches `compiler_rt` last; its
single object came in for `__divti3`, and its math functions with it,
on the reference side only. `ld.lld --why-extract` named the chain.
`scripts/platform-libs` now puts gcc's `libgcc.a` ahead of it on Linux,
which supplies `__divti3` alone; the port's binary is symbol for symbol
unchanged. `quant-diff` called `xcrun` and was Mac-only; it links through
`platform-libs` now.

### Upstream's tools on Linux

None of the twelve had been built on Linux. The port's needed two fixes,
both on paths no Mac build compiles: `tools/common/cpu.zig` called
`std.c.fgets`, which `std.c` does not declare, and then sliced a `[:0]u8`
into itself; `tools/quantize.zig`'s `equivalent` used `std.c.Stat`, which
is `void` on Linux, and takes `statx`'s device and inode there now.
Upstream's tools, the libmtmd gate programs and the setup helpers linked
`libggml.a` without the CUDA libraries' path: `build/tools.zig` and
`build/mtmd*.zig` receive the toolkit now, and call
`llamacpp.addCudaRuntime` -- `addCudaLibraries` without libstdc++. The
first attempt used `addCudaLibraries` itself, and every argument error
in upstream's tools then aborted: Zig's libc++abi threw, libstdc++'s
`__gxx_personality_v0` unwound, and no exception could be caught
(`scripts/platform-libs` describes the same trap). The two setups that
build helpers link through `platform-libs` now.

What differed between the sides was the machine, and each is masked or
switched off for both:

- **Free memory** moves by a few MiB between the two runs on CUDA, which
  reads unified memory from `/proc/meminfo`. `scripts/tools-diff` masks
  it for every tool, as `api_diff.zig`'s `normalize` does: the device
  list, the loader's line, `available_memory_kb`, and `--fit`'s
  breakdown row and the two lines that quote it.
- **An abort's backtrace**: ggml attaches gdb on Linux, which prints
  thread ids and addresses. `GGML_NO_BACKTRACE=1`, which both sides
  honour.
- **`common_log`'s worker thread** prints a parser error after the usage
  text on the GB10, every time, and before it on the Mac; three
  `tokenize` cases compare stderr unordered (`~`), as the script provides.
- **A timestamp after the loader's progress dots**, on CUDA only: one more
  `imatrix` mask.

**Every tool passes, all twelve.** 104 cases across five tools load
`$PLAIN` (`plain-d6_base_005000.gguf`, zigtrain's f32 base model), which
the GB10 did not have at first; without it they ran with both sides
failing to open the file -- passing having compared nothing, apart from
`cvector-generator`, which upstream segfaults on where the port reports
it. With the model copied to the repository root and passed as
`PLAIN=$PWD/plain-d6_base_005000.gguf`, all five pass: `gguf-split` 29,
`cvector-generator` 13, `imatrix` 35, `perplexity` 31, `bench` 48, and
75 of the 104 `$PLAIN` cases exit 0 on both sides -- the rest are the
cases' deliberate errors. `make qwen3` fetched the other model six cases
use.

### DESKTOP-BR7J5S9: the x86_64_v3 items, from the GB10

Items 5 and 6 of the pickup, and half of 7, run from the GB10 for the
first time: the gates cross-build the same way from either host, and
`zigupdater` (the binary is no longer called `zupdater`; `ZUPDATER` names
it) links the GB10 to the laptop as it linked the Mac.

| gate | result |
|---|---|
| `x86-quants-diff` | 28 kernels identical |
| `x86-repack-diff` | 116 checks identical |
| `x86-ops-diff` | 826 lines identical, 1 and 3 threads, the `iqp` cases among them |
| `x86-node-diff` | 5,505 lines identical: 2,750 nodes at 1 and 4 threads, and the fused decode |
| `x86-node-diff`, `IQ4_XS` | 5,505 lines identical, the model loaded as `IQ4_XS - 4.25 bpw` in every run |
| `win-host-diffs` | `host-arms-diff` 168 lines, `libcxx-diff` 452,904, `regex-diff` 43,537,552, `unicode-diff` 7,070,454 |

**The x86 per-kernel gates needed `nm` to read COFF, and on the GB10 it
cannot.** `x86-quants-diff` and `x86-repack-diff` rename every symbol the
reference object defines, taking the list from `nm` on the Windows
object; GNU binutils on Linux arm64 know no PE/COFF target, printed
nothing, and the unrenamed reference collided with the port at link time.
`scripts/coff-defined` reads the COFF symbol table, both the plain and the
`/bigobj` header, and the two scripts use it where `nm` cannot read the
object -- the Mac's LLVM `nm` still lists them there. Checked against
`nm` on the same sources built as ELF: 27 and 45 symbols, identical.

**The IQ panel runs on both sides** (item 6). `harness/win_ops_dump.c`
takes an optional `NAME=VALUE` it sets with `_putenv` -- `push` carries no
environment -- and `X86_PORT_ENV` passes one to the port side only. With
`GGML_NO_IQ_PANEL=1` there, 56 lines differ at each thread count, every
one an `iqp mul_mat` case: the panel, not `vec_dot`, is what the 826
identical lines compared.

**`quant-diff` has no Windows mode.** Its harness is host-only, and its
inputs -- upstream's vocab files and the local models -- would have to be
on the laptop, which `push` cannot put there. The pickup asked for it
because quantize now maps its input on Windows (quant.zig's
`llama_model_quantize_impl`); that needs a Windows harness first.

The first `x86-ops-diff` and `x86-node-diff` runs failed to start: a
second `zigupdater` process, a `tty` probe, held the GB10's host lock.
The link allows one host per machine.

### `make validate`

`validate` stopped at `lint`: there is no `zlint` on the path, and the
local `../zlint` build takes no `-c`, so the styleguide's rules cannot be
applied here. `make build test` ran in its place, as the Makefile's
no-styleguide `validate` does, and failed in `api-diff`, on two harness
faults the Mac's heap had hidden: the chat-template section read each
model's template after the model was freed, and the `file_ptr` load
compared its log without `normalize`, so CUDA's free memory differed.
After both, two runs clean, 70,224,784 checks. The iOS library build
cannot run on Linux (no SDK); the Mac covers it.

Not run here: `lint`.

## The v0.4.0 Mac re-checks

The Mac's half of the v0.4.0 pickup (`PLAN.md`, "Mac, after the GB10
pickup"), after `7c9cbc1` changed the tools' link lines and every
`tools-diff` case's environment.

- `make validate`, the styleguide's: clean. 8,295 citations at
  `5266f24da`, 131 graph nodes, 16 scheduler assignments, every
  in-process diff identical.
- `tools-diff`, all twelve tools: identical. `gguf-split` 29,
  `quantize` 46, `tokenize` 47, `fit-params` 18, `export-lora` 18,
  `batched-bench` 11, `cvector-generator` 13, `imatrix` 35, `perplexity`
  31, `bench` 48, `tts` 30, `mtmd-cli` 44.

The first `tools-diff` pass ran straight after `validate` and four tools
failed -- `export-lora` and `imatrix` in their setups, `batched-bench` and
`cvector-generator` on 9 cases -- every one on a missing
`zig-out/arch-fixtures/*.gguf`: the styleguide's `validate` runs `clean`.
After `scripts/gen-arch-fixtures` all four passed. `SYNC.md`'s Step 7
already says so for the gates; it holds for `tools-diff` too.

**JWHLPTOP stays open.** Its agent, and the GB10's and DESKTOP's, sat at
"waiting for …" on 2026-10-28. Marc directed the v0.4.1 sync to start
over that one open pickup; the AVX-512 items carry into v0.4.1's list.

## Upstream sync v0.4.0 → v0.4.1

The second sync. The pin moved from `v0.4.0` (`5266f24da`, `b10809`) to
`v0.4.1` (`b29c606e2`, `b10964`), ggml from 0.23.0 to 0.24.0, and the
port's version with it: `0.4.1`. It started over one open pickup, v0.4.0's
JWHLPTOP items, at Marc's direction; they carry into this sync's list.

### What moved

`scripts/upstream-map` gave **197 touched citations**, **168 hunks no
citation covered**, and **43 changed files with no citation**. The repin
moved **7,231 citation sites** in 533 files. `upstream-relines` moved
**316** abort-message line numbers and left none for hand work. Upstream
added three files to the build: `ggml-metal-fusion.cpp`,
`common/json-schema.cpp` and `common/parsers/*.cpp`. Both `build.zig`s now
list them.

The work was split across six agents, each with its own set of files: ggml
core with CPU and Metal, CUDA, Vulkan, the libllama core with the existing
models, the three new architectures, and tools with the CLI and libmtmd.
Before they started, the lead moved the reference, ran the generators and
both mechanical moves, and ported `ggml_prec_set_acc` and
`ggml_prec_set_src`, which libllama and libmtmd now call instead of the
deprecated `ggml_mul_mat_set_prec` and `ggml_flash_attn_ext_set_prec`.

**What was ported, by area:**

- **ggml**: `ggml_prec_set_acc` and `ggml_prec_set_src`, writing op_params
  slot 0 (or 3 for flash attention) and slot 3 for src 1. Since
  `MUL_MAT_ID` now takes an accumulator precision too, `graph-diff` gained
  144 dumps covering every op arm and refusal. In the scheduler, the two
  grow functions log at DEBUG, the "too many inputs" split check is gone,
  and `compute_splits` skips an empty ids tensor.
- **CPU**: repack dispatch gains the VXE arm, which is always 0 off s390x.
  `CACHE_LINE_SIZE` needs nothing: the port always used 64. The
  `arch-fallback.h` removals are all in the s390x block.
- **Metal**:
  - `ggml-metal-fusion.cpp` → `src/ggml/metal/fusion.zig`, 17 C-ABI
    exports. The table both `graph_optimize` and the encoders consult is the
    single source of truth. The `ggml_metal_fusion_info` debug and stats
    object is called by the `.m` files.
  - The encoders consume the table: `op_bin` (snake and the ADD chain),
    `op_norm`, and `op_gated_delta_net`, which gains the new GDN_CACHE
    fusion (`nb_out` and a `state_out` binding).
  - The IQ mat-vec kernels have new `_SPLIT` shapes (seven `N_R0_*_SPLIT`
    constants in `impl_c.zig`).
  - There are 319 new tuning rows and five new proc-address entries.
- **CUDA**:
  - `GGML_CUDA_FA_QUANTS` replaces `GGML_CUDA_FA_ALL_QUANTS`, and the build
    takes `-Dcuda-fa-quants` (see below).
  - Flash attention: the f16 fallback for a K/V pair with no compiled
    instance, stream-k at Ada and newer, and K≠V accepted.
  - The cuBLAS BF16 arm. mmq sizes its tiles from the new `ncols_opt`.
  - The mmvq Q4_K/Q5_K retune for Ada and Blackwell.
  - The device-only changes need no host change.
- **Vulkan**:
  - Upstream's matmul refactor: one quant shader with an A-type spec
    constant, and pipelines keyed by a map.
  - TQ1_0 throughout; the three DeepSeek-V4 hyper-connection ops.
  - Fusions: RMS_NORM `MUL_ADD`, `MUL_ADD_MUL` and `VIEW_SET_ROWS`, and
    UNARY+MUL.
  - `graph_optimize` now calls the scheduler's `add_alloc_dep`.
  - Misaligned descriptors, a 2-D FILL, and the swapped-input mat-vec.
  - NVIDIA's submit mutex, which is live on the GB10.
  - Out of scope: the `GGML_VULKAN_RUN_TESTS` and `CHECK_RESULTS` hunks, and
    a 32-bit Vulkan-Hpp logging operator.
- **libllama**:
  - Three architectures (150 entries): `hy_v4` (independent
    hyper-connections scaled by the new `hc_magnitude`, and a DSA lightning
    indexer whose shared layers reuse the last full layer's top-k),
    `maple` and `spark2_5`.
  - The shared helpers `build_gdn_l2_norm` (models.h) and a `build_qkv`
    overload with per-q/k/v head shapes and `reshape`.
  - Pre-tokenizers `hy_v4` and `spark2_5`.
  - The grammar's repetition threshold changed from `>=` to `>`.
  - The lazy-mode fallback when a device lacks mmap.
  - `create_memory`'s HY_V4 arm, with a generalised nextn filter.
  - `llama_sampler_chain_n` now returns `int32_t`.
  - Twenty-odd existing models moved to the new helpers.
- **Tools and the CLI**:
  - `--log-jsonl`/`--no-log-jsonl`, with `common_log`'s JSONL mode and
    `fit_memory_breakdown` record. These need `cli/json.zig`, which ports
    nlohmann's escaping and was compared on 200,000 random strings.
  - `--mlock`, `--mmap` and `--direct-io` (and llama-bench's `-mmp`/`-dio`)
    are removed upstream, and new cases check that they are now rejected.
  - The mmproj device now follows `--device`.
  - `--log-jsonl` is deferred for our `llama-cli` (`UPSTREAM.md`).
- **libmtmd**: `ggml_prec_set_acc` replaces the old setter. The
  `mtmd-helper.cpp` hunks are all under `MTMD_VIDEO`, which the reference
  does not define.
- **Not ported, by the scope map**:
  - `common/chat.cpp`'s 2,534-line move of its format-specific parsers
    into `common/parsers/`, together with tool-call parsing, json-schema and
    the PEG parser;
  - the s390x repack;
  - `mmq-config-gcn.cuh`, which is AMD.

### Decisions taken in this sync

- **`-Dcuda-fa-quants` defaults to `all`, not upstream's four pairs.** This
  keeps the compiled kernel set, and so the gates' coverage, as it was at
  v0.4.0. Upstream's default only matters for gating the new f16 fallback,
  which is a separate GB10 pickup item at that list.

### What the tooling and the generators got right and wrong

- **The repin cannot see a wrapped citation in two CUDA files**:
  `convert.zig:104` and `cpy.zig:163` stayed at `@5266f24da`. The CUDA
  agent re-cited them by hand. They are the same class as the wrapped
  citations `port-links` once skipped.
- **`upstream-map` does not cover `harness/`**: its citations stayed at A -- 31 in all, the agents
  repinning those in their own harnesses. The rest, on lines such as
  `repack_diff.zig`'s `Mirrors …` and `libcxx_ref.cpp`'s comparator notes,
  stayed at A. The lead mapped each one through the A..B diff and
  repinned only lines in unchanged regions whose text is identical at both
  commits. Five line numbers moved, and none needed hand work.
- **`impl_c.zig` is hand-kept**: the seven `_SPLIT` constants had to be added
  to it and to `struct_layout.cpp` by hand, as `SYNC.md` says.
- **`bpe.zig` calls its pattern table generated, but no generator exists.**
  `spark2_5`'s patterns were converted byte for byte, and `vocab-diff`
  confirms them (an injected missing pattern failed 492 checks).
- **A stopgap that compiled was not a stopgap that worked.** To keep the
  tree linking before the Metal port, the lead compiled upstream's
  `ggml-metal-fusion.cpp` as C++. Everything linked. But the v0.4.1 `.m`
  calls `ggml_metal_op_init` with a new argument list (`finfo` in place of
  `use_fusion` … `debug_fusion`), and the v0.4.0 Zig still took the old
  one. Every argument after the graph was shifted:
  - `gen-mmproj-fixtures` failed 38 of 122 checks on the GPU, with a
    `ggml-alloc` assert and all-zero chunks;
  - the new-architecture agent saw the same assert in `arch-model-diff` on
    the reference side;
  - nothing at link time noticed, because the `.m` declares the function
    from a header the Zig does not import.

  The Metal port fixed it.
- **`validate`'s lint caught a `catch unreachable`** in `cli/json.zig`; the
  escape is now written digit by digit, and its test covers it.

### The gates

`make validate` passed once the lint finding above was fixed. Then the Mac
set ran in full, from fixtures regenerated after validate's clean. Every
gate passed but one, and that one was the apparatus. A second hole showed
in the fixture count, not in any verdict:

- **`xcframework-check`'s macOS CPU smoke differed on 4 of 6 prompts**
  against a reference driver built on Oct 26 at v0.4.0. The rule for
  `driver-<cpu>` depended only on `test/driver.c`, so the pin move never
  rebuilt it, and the smoke was comparing two releases. The driver now
  depends on `build/upstream.zig` as well. With it rebuilt, the
  run reached `compat`, which failed the same way: `build/upstream`,
  upstream's own `build-xcframework.sh` output, also dated from v0.4.0,
  and its headers differ by the sync. `make upstream` now records the
  commit it built at, and `compat` refuses a build at any other one.
  **Both had accepted whatever an earlier run left behind**, the same
  hole `check-reference-pin` closes for the bit-exact gates.
- **`arch-fixtures` wrote 164 fixtures, where v0.4.0 wrote 162 and three
  architectures were added.** Gemma 4 had lost its fixture: since v0.4.1
  `load_arch_hparams` reads the SWA pattern with `get_arr`
  (gemma4.cpp:5), which rejects the scalar `get_gguf_ctx` writes.
  Upstream's own loop skips Gemma 4, so nothing upstream noticed. A
  "no fixture" line is not a failure, so nothing here noticed either. The
  libllama agent reported it; the lead nearly passed it over. The
  `get_arr` port and the fused-QKV path had only `gemma4-assistant-diff`,
  whose target is a Gemma 4 it builds itself. `arch_fixtures_extra.cpp`
  now writes the pattern as a bool array, every fifth layer full attention
  as the recipe's comment says, and `gemma4-dense` passes gpu:22 cpu:22.
  With layer 4 marked SWA in the port it fails, the port's load
  stopping on a tensor count. `arch-fixtures` now writes 165.

| Gate | Result |
|---|---|
| `make probe` | both runs |
| `ops-diff`, `repack-diff`, `node-diff`, `parity-port`, `parity-port-cpu` | identical at `baseline`, `apple_m3`, `generic`, `apple_a7` and `apple_a12`: 762 op results, 116 repack checks, 6/6 and 6/6 |
| `node-diff --gpu` | every node and every fused decode identical |
| `backend-ops` | 22,687 op configurations, both backends; `--diff` 26,869 lines identical |
| `metal-diff` | 2,200 answers |
| `tuning-diff` | 11,290,089 lookups |
| `struct-layout` | 318 layout facts, and the 3,177 of `kargs` |
| `abi-check` | 34 entry points |
| `sampler-diff` | 40,308 records, 19 of them the new `edge2000` grammar |
| `context-diff` | 2,313,167 checks |
| `arch-model-diff` | 162 pass, 0 fail, the two `REF-ABORT`s known since v0.3.0 (PLM, `rwkv6qwen2`); Gemma 4, run alone once its fixture was restored, passes |
| `gemma4-assistant-diff` | dense and MoE, Metal and CPU |
| `jinja-diff` | 69 templates × 7 conversations, both time zones |
| `cnv-diff` | 16 scenarios |
| `fit-diff` | 68 cases, ReleaseFast and ReleaseSafe |
| `tools-diff` | every tool's cases, `llama-tts` 30 and `llama-mtmd-cli` 46 included |
| `blas-diff`, `blas-node-diff` | identical; 366 of 384 nodes computed differently without BLAS |
| `mtmd-gates` | `mtmd-diff` 244 identical (4 aborting alike), the image, audio and decode diffs, 84 exports |
| `adapter-diff` | 283,372 checks |
| `quant-diff` | 143,981,322 checks |
| `opt-diff`, `opt-llama-diff` | 89,924 checks; 6 abort cases alike |
| `meta-diff` | 78,015 end-to-end lines |
| `host-arms-diff` | 18 scenarios |
| `parity-cli` | 6/6 |
| `xcframework-check` | every slice's smoke 6/6, the negative tests, the port gates, `compat` (91 files byte-identical, 10,989 exports) and `dropin-macos`; upstream's generic-CPU macOS slice against ours, reported and not a gate, differs on 3 of 6 prompts where v0.4.0 differed on 2 |

### Injections

Each new gate case and changed function was made to fail once:

- **Metal** (`metal-diff`), 4/4: an ADD pattern dropped from the fusion
  table; GDN's `n_written` with `@max` for `@min`; the snake closure check
  neutralised; the ADD-chain buffer check moved from full mode to structural.
  **`node-diff --gpu` cannot see a dropped fusion pattern**: fused and
  unfused Qwen3.5 decodes give bit-identical logits, which is why the table
  check lives in `metal-diff`.
- **`graph-diff`**: removing `prec_set_src`'s `idx != 1` refusal makes 116
  lines differ.
- **libllama**: `build_gdn_l2_norm` with `eps` in place of `eps/n`
  (`arch-model-diff` fails on qwen35 and kimi-linear); spark2_5's fourth
  BPE pattern dropped (`vocab-diff`, 492 checks); the grammar threshold back
  to `>=` (`sampler-diff`'s new `edge2000` case).
- **New architectures** (`arch-model-diff`, CPU): hy_v4's `hc_magnitude`
  scale set to 1.0; maple's Q normed with the K norm; spark2_5's gate
  `sigmoid` as `silu`.
- **Tools**: the fit breakdown's device row writing `context` as `compute`
  (`tools-diff perplexity`); the mmproj device forced to the GPU
  (`tools-diff mtmd-cli`, the `-dev none` case); a citation drifted by one
  line (`port-links`).

### Pickups

`PLAN.md`, "Pickup for the v0.4.1 sync". The CUDA host layer has not
compiled on any machine: this Mac has no toolkit, and that is the GB10's
first item. The Vulkan host layer compiled here against a stub
`libvulkan.dylib` with every glslc feature on, so every comptime arm
type-checked, but none of it has run.
