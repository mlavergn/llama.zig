# MAINTENANCE.md

Plan to (1) make the port's citations complete against `llama.cpp/`,
(2) record the upstream basis in `UPSTREAM.md`, and (3) define how the port
follows upstream release to release. For review; nothing below is done yet.

## Decided (round 1)

| # | Decision |
|---|---|
| 1 | Citations keep the short SHA (`@c1d0e7a00`); `UPSTREAM.md` maps SHA ↔ tag ↔ build tag |
| 2 | After a sync **every** citation is at the new basis; unchanged ones are repinned mechanically; only listed deferrals may lag |
| 3 | Catch up one minor release at a time: `v0.4.0`, `v0.4.1`, `v0.5.0`, `v0.6.0`, `v0.7.0` |
| 4 | Standing grant to update `llama.cpp.zmake` (pin and `build.zig`) **for upstream syncs only**; recorded in `CLAUDE.md` |
| 5 | A definition folded into its caller is cited **in the caller's comment** (``Ports `f` (…), inlining `g` (x.c:N @sha)``); that counts as its citation |

## Decided (round 2)

| # | Decision |
|---|---|
| 6 | The port's version **mirrors the upstream release it is synced to** (`v0.4.0` …); a sync is one commit, made when asked |
| 7 | Mac gates block a sync; GB10 and Windows runs go on their pickup lists and must clear before the **next** sync starts |
| 8 | New upstream features inside ported scope (architectures, ops, samplers, flags) are ported in the same sync; deferral needs a reason in `UPSTREAM.md` |
| 9 | `port-audit` covers functions, methods, structs/unions/enums, globals, and function-like macros carrying logic; constant macros only where no generated table checks them |
| 10 | **Syncs wait until the chat-template question is settled** (`PLAN.md`, "Questions") |
| 11 | `.gitmodules` and the checkout's `origin` move to `https://github.com/ggml-org/llama.cpp.git` |

## Decided (round 3)

| # | Decision |
|---|---|
| 12 | Version becomes `0.3.0` at the next commit; work between syncs bumps build metadata (`0.3.0+1`, `+2` …); a sync resets to the bare tag. Zig's `SemanticVersion` parses it |
| 13 | **Fix `zigjinja`** until `jinja-diff` passes: provenance, the 20 parse failures, `trim_blocks`/`lstrip_blocks`, local-time `strftime_now`. It stays a separate engine, so `jinja-diff` runs at every sync (upstream's `common/jinja` + templates moved +2.7k / -0.2k lines `v0.3.0..v0.7.0`) |
| 14 | Phases 1-3 wait for the chat-template work |
| 15 | Vulkan (Stage 7) runs **before the syncs**, beside the chat-template work |

## Decided (round 4)

| # | Decision |
|---|---|
| 16 | `zigjinja` work happens directly on its current branch; your uncommitted `.gitmodules`, `LICENSE` and `README.md` changes are left untouched and out of any commit; commits there only when asked |
| 17 | `jinja-diff` passes only on **identical trust marks**, byte for byte, keeping upstream's exemption for bound `bos_token`/`eos_token` |
| 18 | `zigjinja` resolves local time itself, through `std.tz` |
| 19 | Vulkan toolchain on the GB10: check `VK_HEADER_VERSION` first; `sudo apt install glslc libvulkan-dev spirv-headers vulkan-tools` if it includes cooperative-matrix-2, else Khronos's headers pinned at one SDK tag in a user directory. **Approved** |
| 20 | Vulkan is ported at `v0.3.0` now and synced forward with everything else |

## Order of work

1. **Now, side by side:** `zigjinja` against `jinja-diff` (Phase 0);
   Vulkan's GB10 pickup, `vk-repro`, then its gates and host port at
   `v0.3.0` (`PLAN.md`, "Stage 7").
2. Phases 1-3, once Phase 0 is done (Decision 14). `port-audit` covers
   whatever of `ggml-vulkan.cpp` is ported by then.
3. Phase 5, one release at a time; Vulkan's ported code syncs with the
   rest (Decision 20).

## Phase 0 -- `zigjinja` until `jinja-diff` passes (Decision 13)

**Done 2026-10-22** except step 6's pin, which waits for a zigjinja
commit: 68 of 68 templates agree in text, raises and trust, and the clock
under two zones. `PLAN.md` Decision 58; `NOTES.md`, "zigjinja against
`jinja-diff`".

1. Parse failures: the 20 templates, by `jinja-diff`'s reason tally.
2. `trim_blocks` / `lstrip_blocks`, read by the lexer.
3. Trust by provenance: strings carry input/template marks through
   concatenation, filters, `set`, loops and macros, as upstream's
   `is_input` parts do; `cli/chat.zig` reads the marks instead of
   `Environment.finalize` sentinels.
4. `strftime_now` in local time, in `zigjinja` (Decision 18): zone from
   `TZ`, else `/etc/localtime`; `std.tz` parses the TZif, and **its POSIX
   footer rule has to be evaluated too** -- slim TZif files carry no
   transitions past the last rule change, so the current offset lives only
   there. `std.tz` itself is OS-independent, but Windows ships no TZif
   files to feed it: `GetDynamicTimeZoneInformation` there.
   Drop `TZ=UTC` from `scripts/jinja-diff`; run it under two zones, one
   with DST.
5. Pass criterion (Decision 17): text, raises, and trust marks identical
   on all 68 templates.
6. Pin the new `zigjinja` commit; `make jinja-diff` joins the sync gates.
   Work on its current branch (Decision 16).

## Where things stand (measured 2026-10-22)

| | |
|---|---|
| Upstream | `github.com/ggml-org/llama.cpp` (`.gitmodules` still says `ggerganov/llama.cpp`, which redirects) |
| Basis | **`v0.3.0`** = `c1d0e7a00` = build tag `b10621`, 2026-08-25. `llama.cpp/` and `llama.cpp.zmake/llama.cpp` both at it |
| Citation format | ``Ports `ggml_hash_set_new` (ggml.c:6516 @c1d0e7a00)`` -- symbol, file, line, **and revision, per citation** |
| `make port-links` | 5,700 citations in 416 files, all resolve; 1 upstream commit |
| Zig files with no citation | 7 of 458: 4 not ports (`root.zig`, `module.zig`, two goldens), `libcxx.zig` (cites libc++, not upstream), `repack/epilogue.zig`, and a stray tracked `src/ggml/.mc.zig` |
| Upstream units with no citation | 6, all expected except one to confirm: `common/chat.cpp`, `ggml.cpp`, `hbm.cpp`, both `cpu-feats.cpp` (not ported), **`src/models/qwen3tts.cpp`** |
| Upstream definitions with no citation | **~580 of ~4,100** in the 264 cited `.c/.cpp/.cu/.m` files -- regex estimate, includes false positives (CUDA device kernels, which ship as fatbins; inlined helpers; dead `#if` arms) |
| Behind upstream | 4 minor releases: `v0.4.0` 09-04, `v0.4.1` 09-14, `v0.5.0` 09-23, `v0.6.0` 10-05, `v0.7.0` 10-15 (~10-day cadence) |
| Diff per release, ported paths | 4k-15k inserted lines each; `v0.3.0..v0.7.0` totals ~47k inserted, ~16k deleted |

**The format already carries the revision.** What is missing is (a) a
reverse check that every live upstream definition is cited, (b) a record of
the basis, and (c) tooling that turns a tag-to-tag diff into a list of Zig
functions to revisit.

## Phase 1 -- `UPSTREAM.md`

- Basis: tag, commit (the SHA citations carry), build tag, date, remote URL.
- Point `.gitmodules` and `llama.cpp/`'s `origin` at `ggml-org` (Decision 11).
- Sync history table: from → to, date, port version, hunks ported/deferred.
- Scope map, every upstream path in one of four classes:
  - **ported** -- cited Zig; synced by hand through citations.
  - **compiled unmodified** -- CUDA `.cu` (fatbins), Metal `.metal`, the two `.m` files, `cub_shim.cu`'s inputs.
  - **generated** -- regenerated by script, never hand-edited: `gen-arch`, `gen-kargs`, `gen-hparams`, `gen-model-tables`, `gen-unicode-data`, `gen-fattn-config`, `gen-mmq-config`, `iq-grids`, `upstream_flags.zig`, goldens (`vecdot-golden`, `vec-golden`, `quants-golden`), arch fixtures.
  - **not ported** -- other backends, `tools/`, `examples/`, `common/` beyond what `cli/` mirrors.
- Deferred hunks: upstream changes knowingly not taken, with reason.

## Phase 2 -- complete the citations

1. **`scripts/port-audit`** (reverse of `port-links`): for each ported
   translation unit, list every definition the target actually compiles --
   `zig cc -Xclang -ast-dump=json` with the reference's flags, so dead `#if`
   arms drop out -- the kinds Decision 9 lists. Each must be
   cited somewhere in the port -- its own declaration, or an `inlining`
   clause on its caller (Decision 5) -- or appear on an exclusion list with
   a reason (`device-code`, `not-on-target`, `not-ported`).
2. Teach `port-links` the `inlining` clause (each inlined symbol checked
   at its own line). Fill the gaps `port-audit` reports; re-run both.
3. Forward check, in `port-links`: every ported file's header names its
   source paths; every `export`/`callconv(.c)` function cites.
4. Resolve `qwen3tts.cpp`; delete `src/ggml/.mc.zig`; give
   `epilogue.zig` its citations or a "not a port" header.
5. Add `port-audit` to `make validate`; negative-test it (drop one
   citation, drop one exclusion, cite a dead arm).

## Phase 3 -- sync tooling

1. **`scripts/upstream-map A B`**: `git diff A B` over ported paths; for each
   hunk, the citations whose definition extent (from the AST at `A`) it
   touches. Output: touched Zig functions; **unmapped hunks** (new
   definitions, uncited files, header/struct/macro changes); deletions.
2. **`scripts/upstream-repin A B`**: for every citation at `A` whose
   definition is unchanged between `A` and `B`, rewrite to `B`'s line and
   commit; verified by `port-links`. Only changed definitions need hand
   work.
3. `port-links --basis <sha>`, basis read from `UPSTREAM.md`: fail if any
   citation is at another commit and not listed under deferred (Decision 2).
4. `scripts/check-reference-pin` already ties `llama.cpp.zmake/llama.cpp`
   to the submodule; a sync moves both, plus `llama.cpp.zmake/build.zig`
   when upstream adds or moves sources (Decision 4).

## Phase 4 -- the sync procedure (per release tag)

1. Move both checkouts to `B`; rebuild the reference.
2. `upstream-map A B` → work list.
3. Regenerate the generated class; rebuild fatbins/embedded shaders.
4. Port each touched definition and unmapped hunk, new features included
   (Decision 8); cite at `B`.
5. `upstream-repin A B` for everything else.
6. Gates (Decision 7): `make validate` and the bit-exact set on the Mac
   (`baseline`, `apple_m3`, `generic`) block. The CUDA set and Linux
   pickup on the GB10, and `x86-gates` on the laptop, go on the pickup
   lists and must clear before the next sync. New upstream ops and
   architectures get gate cases.
7. Update `UPSTREAM.md` (basis, history, deferred); set the version to
   `B` (Decision 6); one commit, when asked.

## Phase 5 -- catch-up (Decision 3)

Starts once the chat-template question is settled (Decision 10).

| Sync | Ported-path diff |
|---|---|
| `v0.3.0 → v0.4.0` | 140 files, +11.6k / -1.5k -- first run of the tooling; fix it before continuing |
| `v0.4.0 → v0.4.1` | 94 files, +3.9k / -0.9k |
| `v0.4.1 → v0.5.0` | 102 files, +5.3k / -4.9k |
| `v0.5.0 → v0.6.0` | 173 files, +14.9k / -5.9k |
| `v0.6.0 → v0.7.0` | 260 files, +11.7k / -2.5k |

Then follow each new minor release as it lands.

## Questions -- round 1

1. Revision on each citation: keep the short SHA, or the tag?
   - (a) Keep `@c1d0e7a00`; `UPSTREAM.md` maps SHA ↔ tag. Immutable, already enforced, works with `git show`.
   - (b) Switch to `@v0.3.0`. Readable; needs `port-links` changes and a one-time rewrite of 5,700 citations.
   - (c) Both, `@v0.3.0/c1d0e7a00`.

> a

2. After a sync, should every citation move to the new basis?
   - (a) Yes: unchanged ones are repinned mechanically, so "basis = B" is true everywhere except listed deferrals.
   - (b) No: only re-read definitions move; citations keep mixed revisions (what `port-links` supports today).

> a

3. Sync granularity for catching up to `v0.7.0`?
   - (a) One minor release at a time: `v0.4.0`, `v0.4.1`, `v0.5.0`, `v0.6.0`, `v0.7.0`.
   - (b) Straight to the latest release.
   - (c) Rehearse on `v0.4.0`, then jump to the latest.

> a

4. A sync must update `llama.cpp.zmake` (pin, and `build.zig` when upstream adds/moves sources). Standing grant for that?
   - (a) Yes, for upstream syncs only.
   - (b) No, ask each time.

> a

5. Scope of the audit's exclusions: may a definition be excluded as "inlined into its caller" without its own citation?
   - (a) Yes, but the caller's citation must name it (``Ports `f`, inlining `g` (…)``).
   - (b) No: every live definition gets its own citation line.

> a

## Questions -- round 2

6. Versions and commits per sync?
   - (a) One commit per sync, made when you ask; patch bump (`v0.0.33` …); `UPSTREAM.md` history maps port version ↔ upstream tag.
   - (b) Mirror upstream: port version tracks the upstream minor (`v0.4.0` …).
   - (c) No bump per sync; bump only on your schedule.

> b

7. Which gates must pass before a sync counts as done?
   - (a) Mac gates block; GB10 and Windows runs go on their pickup lists and must clear before the *next* sync starts.
   - (b) Every machine blocks each sync.
   - (c) Mac gates only; other machines whenever they are reachable.

> a

8. New upstream features inside ported scope (a new architecture, op, sampler, flag)?
   - (a) Port them in the same sync; defer only with a reason in `UPSTREAM.md`.
   - (b) Sync changes to already-ported code only; new features become separate items.

> a

9. `port-audit` scope in headers?
   - (a) Functions, methods, structs/unions/enums, globals, and function-like macros carrying logic; plain constant macros only where generated tables don't already check them.
   - (b) Functions and methods only.
   - (c) Everything, every macro included.

> a

10. Ordering against the other open work (Windows pickup, chat-template decision, Vulkan)?
    - (a) Phases 1-3 now; syncs next; Vulkan starts after `v0.7.0`, so its reference is built once at the new basis.
    - (b) Phases 1-3 now; Vulkan before the syncs.
    - (c) Syncs wait until the chat-template question is settled.

> c

11. Point `.gitmodules` at `https://github.com/ggml-org/llama.cpp.git` (and the checkout's `origin`)?
    - (a) Yes.
    - (b) Leave the redirecting `ggerganov` URL.

> a

## Questions -- round 3

12. The version between syncs (Decision 6), and now: the basis is `v0.3.0` and `build.zig.zon` says `0.0.32`.
    - (a) Set `0.3.0` at the next commit. Non-sync work bumps semver build metadata, `0.3.0+1`, `+2` …; a sync resets to the bare tag. No collision with upstream's own patch releases.
    - (b) Non-sync work does not bump; the version changes only at syncs.
    - (c) Pre-release suffix, `0.4.0-1` … (sorts *before* `0.4.0` in semver, so reads as older than the sync).

> a

13. The chat-template question, since it now gates the syncs (Decision 10):
    - (a) Port upstream's `common/jinja` (6,349 lines) into `cli/`, cited like the rest. It then follows upstream through the syncs, `jinja-diff` becomes its bit-exact gate, trust is upstream's per-part provenance, and the `zigjinja` dependency goes away.
    - (b) Fix `zigjinja` (provenance tracking, the 20 parse failures, `trim_blocks`/`lstrip_blocks`, local-time `strftime_now`) until `jinja-diff` passes; it stays a separate engine that does not follow upstream.
    - (c) Accept today's state and record the known defects.

> b

14. Phases 1-3 (`UPSTREAM.md`, `port-audit`, sync tooling) don't touch the chat question. Start them now?
    - (a) Yes, now.
    - (b) Wait for the chat-template work too.

> b

15. Where does Vulkan (Stage 7) go?
    - (a) After the `v0.7.0` sync, so its reference is built once at the new basis.
    - (b) Before the syncs, beside the chat-template work.
    - (c) On hold until you decide separately.

> b

## Questions -- round 4

16. Where the `zigjinja` work happens: `../zigjinja` has uncommitted changes of yours (`.gitmodules`, `LICENSE.md → LICENSE`, `README.md`).
    - (a) A new branch in `../zigjinja`, leaving your changes untouched; commits there only when you ask; `build.zig.zon` keeps the path dependency.
    - (b) Directly on its current branch.
    - (c) You commit or stash your changes first.

> b

17. What "`jinja-diff` passes" requires of trust:
    - (a) Identical marks: every byte trusted on one side is trusted on the other (upstream's exemption for bound `bos_token`/`eos_token` kept).
    - (b) Never trust a byte upstream marks as input; over-marking template text allowed. (This is what makes control tokens tokenize as text today.)

> a

18. `strftime_now` in local time, with `zigjinja` pure Zig:
    - (a) `cli/chat.zig` passes a clock callback; the CLI implements it with libc `localtime_r`, which the binary already links. `zigjinja` stays free of time zones.
    - (b) `zigjinja` reads the TZif database itself (`std.tz`).
    - (c) Keep UTC and record it as a known difference.

> b

19. Vulkan toolchain on the GB10 (needs your approval: it changes a shared machine):
    - (a) Check the distro's `VK_HEADER_VERSION` first; `sudo apt install glslc libvulkan-dev spirv-headers vulkan-tools` if it includes cooperative-matrix-2, else Khronos's headers pinned at one SDK tag in a user directory.
    - (b) Always pinned Khronos headers and a `glslc` built from source in a user directory; no `sudo`.
    - (c) `apt` only, whatever its version.

> a

20. Vulkan's basis: `ggml-vulkan` moved +14.0k / -7.9k lines `v0.3.0..v0.7.0`, so a port written at `v0.3.0` is partly rewritten by the syncs.
    - (a) Measure and gate at `v0.3.0` (`vk-repro`, the gates, the reference build), but start the host port only after the `v0.7.0` sync.
    - (b) Port at `v0.3.0` now and sync it forward with everything else.
    - (c) Revisit Decision 15 and move Vulkan after the syncs.

> b

---

**No open questions.** Four rounds answered; the plan is ready for review.
