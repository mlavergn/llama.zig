# SYNC.md -- the upstream-sync prompt

Not a port and not a record: this is the prompt that moves the port forward
through upstream's releases. Paste it, or say "run SYNC.md". It can be re-run
at any point. It reads where it is from `UPSTREAM.md` and carries on from
there. The plan behind it is `MAINTENANCE.md`; its Decisions are cited as M1–M20.

## Settings (edit these before a run)

| Setting | Value | Meaning |
|---|---|---|
| Version mode | `upstream` | The version tracks upstream (M6, M12). It is `0.3.0` from the commit that closed the pre-sync work. A completed sync sets it to the bare upstream tag (`0.4.0`); commits between syncs add build metadata (`0.3.0+1`, `+2` …) |
| Stop after | `latest` | A release tag (`v0.4.0`), or `latest` |
| Push | `yes` | `git push origin master` after each sync commit, so the GB10 can pull it for its pickup. `no` stops the loop at each sync's pickup |

---

## The prompt

You are moving the Zig port of llama.cpp forward through upstream's releases,
**one release tag at a time**. Each tag goes through the same steps: pull it,
diff it against the current basis, port what changed, gate it, record it,
commit, then start on the next tag. Read `CLAUDE.md`, `MAINTENANCE.md` and
`UPSTREAM.md` (if it exists) first. Then follow the steps below until you hit
**Stop after**, a real blocker, or the latest release.

### What this prompt grants, and what it does not

- **Granted:** moving `llama.cpp/` (the submodule) and
  `llama.cpp.zmake/llama.cpp` to the next release tag; editing
  `llama.cpp.zmake/build.zig` when upstream adds, moves or drops sources (M4),
  and committing that in `llama.cpp.zmake`; editing `.gitmodules`; **one
  commit per completed sync** in this repository, plus one for the tooling
  (iteration 0) and one per pickup fix; pushing `master` if **Push** is `yes`;
  fixing `../zigjinja` on its current branch, committing there and pinning
  the commit, when `jinja-diff` needs it (M13, M16).
- **Still forbidden:** editing any file under `llama.cpp/` or
  `llama.cpp.zmake/llama.cpp/`; jumping more than one release (M3); adding a
  dependency; force-pushing; changing the Settings above on your own.

`CLAUDE.md` says `llama.cpp/` "is pinned deliberately". That rule now means
"moved only by this procedure". Rewrite that paragraph and `PLAN.md` Decision 3
("Freeze at v0.3.0") in iteration 0 to say so.

### Step 0: preflight (every run)

1. `git status` must be clean in this repository, in `llama.cpp.zmake` and in
   `../zigjinja`. Check `ps` for another agent or a `zig build` running in
   this tree. If the tree is dirty or busy, stop and say what you found
   (memory: parallel-agents-shared-tree).
2. Read the basis from `UPSTREAM.md`. If it does not exist yet, the basis is
   `v0.3.0` = `c1d0e7a00` and you start with iteration 0.
3. Upstream is **`https://github.com/ggml-org/llama.cpp`**, and that is the
   only source. Fetch from that URL explicitly, never from whatever remote the
   checkout happens to name:
   `git -C llama.cpp fetch --tags https://github.com/ggml-org/llama.cpp.git`.
   Then ask GitHub what it has released:
   `gh api 'repos/ggml-org/llama.cpp/releases?per_page=100'`. The releases
   are the non-prerelease `v*` entries, and `releases/latest` names the
   newest one. The `bNNNNN` entries are per-commit prereleases, so skip them.
   List the releases after the basis in `sort -V` order, patch releases
   included; the next tag is the first one on that list. Check that the
   fetched tag resolves to the commit
   `gh api repos/ggml-org/llama.cpp/commits/<tag> --jq .sha` gives. A
   release's `target_commitish` is the branch head when the release was cut,
   later than the tag (`v0.4.0`: tag `5266f24da`, target `427291b5b`), so
   don't use it. If the list is
   empty, report that the port is current and stop.
4. **Pickups from the previous sync must be clear before this one starts**
   (M7). If `UPSTREAM.md`'s last history row shows a pickup open, run it now
   (Step 9) before going further.

### Step 1: iteration 0 (once, while `UPSTREAM.md` or the sync tooling is missing)

Do `MAINTENANCE.md` Phases 1–3 as written there. In brief:

- **`UPSTREAM.md`** records:
  - the basis: tag, full SHA, the short SHA citations carry, build tag
    (`git describe --tags --match 'b*'`), date and remote;
  - a machine-readable line other scripts can read:
    `basis: v0.3.0 c1d0e7a004015f23bc0233470b747b596f29b264 b10621`;
  - the history table: from → to, date, port version, port commit, hunks
    ported and deferred, and pickup status for each of GB10, DESKTOP and
    JWHLPTOP;
  - the deferred hunks, each with its reason;
  - the scope map (ported / compiled unmodified / generated / not ported).
- `.gitmodules` and `llama.cpp/`'s `origin` move to
  `https://github.com/ggml-org/llama.cpp.git` (M11). `.gitmodules`'s
  `branch =` follows the basis from now on.
- **`scripts/port-audit`** is the reverse check of `port-links`, built from
  the AST of what the target actually compiles (M5, M9). Teach `port-links`
  the `inlining` clause and the forward check. Resolve `qwen3tts.cpp`,
  delete `src/ggml/.mc.zig`, and give `epilogue.zig` citations. Fill the
  gaps the audit reports, and add `port-audit` to `make validate`.
- **`scripts/upstream-map A B`**, **`scripts/upstream-repin A B`**, and
  **`port-links --basis <sha>`**, which reads the basis from `UPSTREAM.md`.
- **Make every new checker fail before you trust it**, as `CLAUDE.md`'s
  negative-test rule requires. For `port-audit`: drop a citation, drop an
  exclusion, cite a dead arm. For `upstream-map`: a hunk inside a cited
  function must map to that function, and a new definition must come out
  unmapped. For `upstream-repin`: a changed definition must not be repinned.
- The zigjinja pin (`MAINTENANCE.md` Phase 0 step 6, `PLAN.md` Decision 58):
  if the commit is on its remote, pin it by URL and hash. If it is not, keep
  the path dependency and record it in `UPSTREAM.md`.
- Update `CLAUDE.md` and `PLAN.md` as above, and mark Phases 1–3 done in
  `MAINTENANCE.md`.

Commit (Step 10) as the tooling commit, then go on to Step 2 with the first
release. **`v0.3.0 → v0.4.0` is the tooling's first real run**: when the
tooling gets something wrong there, fix the tooling, not just the output, and
negative-test the fix.

### Step 2: measure the release (A = basis, B = next tag)

`git -C llama.cpp diff --stat A B` and `git log --oneline A B` over the ported
paths, plus the release notes. Measure before estimating: `CLAUDE.md` records
two survey counts that were low. Write a short sizing note in your scratchpad
before touching anything: files changed, lines added and removed, new files,
deleted files, and new architectures, ops, flags and `llama.h` symbols.

### Step 3: move the reference

1. Check out `B` in `llama.cpp/` and in `llama.cpp.zmake/llama.cpp`, and
   update `.gitmodules`. `scripts/check-reference-pin` must pass.
2. If upstream added, moved or deleted sources, update
   `llama.cpp.zmake/build.zig` and `build/llamacpp.zig`. Rebuild the zmake
   reference and `zig build reference` (`libllama-ref.a`), both
   `--release=fast`. **Confirm both builds succeeded and that the artifacts
   changed** (hash them). A failed build leaves the old library in place, and
   every gate then passes against `A` (memory:
   verify-the-apparatus-not-the-verdict).

### Step 4: the work list

Run `scripts/upstream-map A B` and save its output to the scratchpad. It
yields three kinds of work: touched Zig functions; **unmapped hunks** (new
definitions, uncited files, header, struct and macro changes); and
deletions. Each item ends in exactly one state: ported, deferred with a reason
in `UPSTREAM.md`, or out of scope according to the scope map.

### Step 5: regenerate the generated class

Never hand-edit generated files.

- **Scripts:** `gen-arch`, `gen-kargs` (then `zig fmt` and `make
  struct-layout`), `gen-hparams`, `gen-model-tables`, `gen-layer`,
  `gen-unicode-data`, `gen-mtmd-names`, `gen-vk-push`, `gen-tuning-table`
  (then `make tuning-diff`), `gen-upstream-flags` and `iq-grids`. Each stamps
  the pin from `build/upstream.zig` (`scripts/upstream_pin.py`); move
  `build/upstream.zig` in Step 3 *before* running any of them.
- **When a generator stops on new C++**, extend the generator, never the
  output. At v0.4.0, `gen-hparams` silently dropped every member after the
  first member function declared mid-struct, and `gen-model-tables` stopped
  on an `if` in a conditional RoPE arm.
- **Hand-kept lists the generators feed:** `impl_c.zig`'s `count` and
  `harness/struct_layout.cpp`'s constant list.
- **Goldens:** `vecdot-golden`, `vec-golden` and `quants-golden`, each with
  its `GOLDEN_CPU=generic` sibling (`scripts/upstream_pin.sh` stamps
  them). They link upstream's CMake build, which nothing else moves: run
  `make buildmacos` first, or the kernels are linked against the previous
  release's `libggml-base`. The `GOLDEN_CPU=generic` run of `vecdot-golden`
  also writes `src/ggml/cpu/quants/golden_generic.zig`, which nothing
  imports and the tree does not track; delete it. `quants-golden`'s
  `IQ4_NL` `zeros` entry reads uninitialized memory and differs every
  capture: restore the committed line.
- **Fixtures:** `arch-fixtures` and `mmproj-fixtures`.
- **GB10 only:** `gen-fattn-config` and `gen-mmq-config`, the CUDA fatbins
  and their kernel table, and the Vulkan SPIR-V that is embedded from the
  reference build. Put these on the GB10 pickup.

**Diff each regenerated file against its previous version.** A change in a
generated table is a work item in its own right: a new enum ordinal, struct
field or kernel argument has a consumer to port.

### Step 6: port

Work through the list in dependency order: ggml core, then the CPU kernels,
then the backends, then libllama, then libmtmd, then `cli/` and `tools/`.
Every touched definition is re-read at `B` and cited at `B`
(``(file.c:N @<B-short-sha>)``). `CLAUDE.md`'s porting rules apply
unchanged: name a fused multiply-add only where an oracle confirms it,
`impl.one` before indexing a C pointer, no `@panic` in library code, port a C
`assert` as nothing, and so on.

**New features inside the ported scope are ported in the same sync** (M8).
Each kind has its own path:

| New in upstream | What to do |
|---|---|
| Architecture | A file under `src/llama/models/`, a registry entry, a hparams arm, and an `arch-model-diff` fixture (`test-llama-archs.cpp` usually brings one) |
| Op | Constructor, CPU kernel, the Metal encoder in `metal/ops.zig`, CUDA host launches, Vulkan host, `supports_op` on every backend, plus cases in `graph-diff` and `ops-diff` |
| Sampler | A port plus `sampler-diff` coverage |
| Flag | `cli/args.zig` (or `tools/common/`), `upstream_flags.zig`, and a `parity-cli` / `tools-diff` case |
| `llama.h` or `mtmd.h` symbol | `api/` plus the export, and the `nm` count equal to the reference |
| Projector type | An `arms_*.zig` arm and a `mmproj-fixtures` type |

**The port is backend-generic.** Port CUDA and Vulkan host changes here, on
the Mac, even though they are gated on the GB10 (memory:
port-is-backend-generic). A brand-new upstream tool or backend is outside the
scope: record it as not ported in the scope map rather than porting it.

If you split the work across sub-agents, give each one a disjoint set of
files and its own scratch directory. They must not register a file in a
`module.zig` until `zig build test-llama` passes with it (memory:
parallel-agents-shared-tree).

Then do the two mechanical moves, **before** hand-porting anything:

- `scripts/upstream-map --repin A B` moves every citation whose definition
  is unchanged. It refuses one whose text at B differs from A.
- `scripts/upstream-relines --write A B` moves the `"<file>.cpp", <LINE>`
  literals the port passes to `impl.assert` and `impl.abort`. Those are the
  C's `__LINE__`, and gates compare abort messages. Its "changed at B" list
  is hand work.
  - **Run it exactly once.** It is not idempotent: a second `--write`
    shifts the moved literals again.
  - Pass `--keep <zig>:<file>:<N>` for any literal already set to B by hand,
    and `--skip <prefix>` for files another agent is editing.

After the port, `port-links` (and, once they exist, `port-links --basis B`
and `port-audit`) must be clean, apart from the deferrals listed in
`UPSTREAM.md` (M2). `git grep '@<A>' src cli tools` must be empty too.

Also update the version files: `build/upstream.zig` (tag, `LLAMA_VERSION`,
ggml's own `GGML_VERSION` from `ggml/CMakeLists.txt`, the commit, the build
number from `git describe --tags --match 'b*'`), `llama.cpp.zmake/build.zig`'s
copies, and the two literals in `src/ggml/runtime.zig` and
`src/llama/api/llama.zig`. `scripts/check-reference-pin` fails until they
agree. Then move every file header's "at vX (`sha`)" provenance line.

### Step 7: Mac gates (these block; M7)

1. `make validate` **first**, then `make probe` (both runs). On a machine
   with `../styleguide/Makefile.in`, `validate` is the styleguide's: it runs
   `clean`, which deletes `zig-out/` and `.zig-cache/`, and `zig fmt`, which
   may rewrite sources. Fixtures generated before it (`make arch-fixtures`,
   `make mmproj-fixtures`) are lost, and the gates that read them then run
   without them. Generate them after it.
2. The bit-exact CPU set: `ops-diff`, `repack-diff`, `node-diff`,
   `parity-port` and `parity-port-cpu`. Run it at the default `GATE_CPU`,
   then at `apple_m3` and at `generic`, plus `apple_a7` and `apple_a12`
   whenever the sync touched `src/ggml/cpu/`.
3. `node-diff ARGS=--gpu`.
4. Every other gate in `CLAUDE.md`'s list that runs on the Mac:
   - `graph-diff`, `backend-ops` (with `--diff`), `sched-diff`, `metal-diff`,
     `tuning-diff`, `struct-layout` and `abi-check`;
   - `sampler-diff`, `libcxx-diff`, `regex-diff` and `unicode-diff`;
   - `vocab-diff`, `arch-diff`, `loader-diff` and `context-diff`;
   - `arch-model-diff` and `gemma4-assistant-diff`;
   - `jinja-diff`, `cnv-diff`, `fit-diff` and `tools-diff`;
   - `blas-diff`, `blas-node-diff` and `mtmd-gates`;
   - `api-diff`, `adapter-diff`, `quant-diff`, `opt-diff` and
     `opt-llama-diff`;
   - `meta-diff`, `batch-diff`, `chat-diff` and `host-arms-diff`;
   - `parity-cli` and `xcframework-check`, after `make -C xcframework
     upstream` rebuilds upstream's own framework at the new pin (`compat`
     refuses a build made at any other commit).

   `make arch-fixtures` reports an architecture the C++ cannot build as a
   "no fixture" line, not a failure. Compare its count with the last
   sync's: one that went missing takes its `arch-model-diff` coverage with
   it, as Gemma 4's did at v0.4.1.

**A gate that could not build is not a pass.** For every gate case you added
in this sync, inject a fault, watch the case fail, and undo the injection by
reverse replacement. When a gate fails, fix the port, not the gate, unless the
gate itself is what is wrong. In that case, say so in `NOTES.md`.

### Step 8: record

- **`UPSTREAM.md`:** the new basis line, a history row (pickups `open`), and
  the deferrals.
- **`NOTES.md`:** a section "Upstream sync A → B". It covers what moved, what
  the work list held, what each gate caught, the injections and their
  results, and anything the tooling got wrong.
- **`PLAN.md`:** the GB10 and Windows pickup lists for this sync.
- **`CLAUDE.md`:** every count that changed: exported symbols,
  architectures, model files, projector types, gate check counts and the
  pinned tag and commit in "`llama.cpp/` is reference material only". A
  stale number in `CLAUDE.md` is a bug.

### Step 9: pickups

These must clear before the next sync (M7), so drive them now. The commit
from Step 10 has to reach GitHub first.

- **GB10** (`spark-dbff`, checkout `~/src/mlavergn/llama.zig`, through
  `zupdater tty`; memory: driving-agents-tty). Move `llama.cpp/` and the
  zmake checkout to the new commit before anything else:
  - the Step 5 regenerations, with the results committed back from the Mac;
  - `cuda-ops-diff`, `cuda-node-diff` (per node and fused),
    `cuda-backend-diff` and `cuda-parity-cli`;
  - `vk-toolchain` if the pin moved;
  - `vk-repro`, `vk-ops-diff`, `vk-node-diff`, `vk-backend-diff`,
    `vk-push-layout` and `vk-parity-cli`, on all four arms;
  - the Linux arm64 CPU set and `host-arms-diff`.
- **DESKTOP-BR7J5S9:** `make x86-gates` at `x86_64_v3`, plus
  `win-host-diffs`. `[The Agent could not open a terminal]` is usually the
  laptop's Application Control refusing that build of the program; run it
  in a `tty` to see.
- **Every machine:** where the pickup names cases, find each in the gate's
  output. A total does not say which cases ran (`vk-ops-diff` passed
  16,417 at v0.4.1 with no fused case among them).
- **JWHLPTOP:** `make x86-gates` with `X86_CPU=x86_64_v4` and again with
  `tigerlake`. Confirm the 512-bit code by disassembly.

Fix any failure on the Mac, then commit it (Step 10, message `… pickup fix
for B`) and re-run the pickup. Then mark the machine `clear` in
`UPSTREAM.md`.

**If a machine is unreachable, stop the loop and say so.** That is a real
blocker. Do not start the next sync over an open pickup.

### Step 10: version and commit

- **Version:** a sync sets `build.zig.zon` to `B` without the `v`
  (`0.4.0`). Any other commit (the tooling, a pickup fix) adds the next
  `+N` to the current basis (`0.3.0+1`).
- **Message:** `v<version> sync upstream B (<B-short-sha>)`, followed by the
  attribution line from the system reminder. For a pickup fix:
  `v<version> pickup fix for upstream B`.
- **Commit:** one commit in this repository, and one in `llama.cpp.zmake`
  if its `build.zig` changed.
- **Push** if **Push** is `yes`.

### Step 11: next

Report in one or two lines: B, the port version, items ported and deferred,
and pickup state. Then go back to Step 0. Stop only at **Stop after**, the
latest release, or a real blocker:

- a gate failure you cannot resolve;
- a pickup machine you cannot reach;
- an upstream change that needs a decision only Marc can make, such as a new
  dependency, a change that contradicts a `PLAN.md` Decision, or a new
  backend or tool that might belong in scope.

Never stop at a checkpoint whose question can be settled later (memory:
dont-stop-at-checkpoints). If a question can wait, write it into the
`NOTES.md` sync section and keep going.
