# Zig 0.16.0 cannot link C++ against the macOS 27 SDK

```sh
make -C testcase
```

Three files and a Makefile. Nothing here depends on llamazig; it reproduces
with the stock `zig` on the PATH. Recipes are not silenced, so **the command
that shows the problem is the line make prints.**

| Target | |
|---|---|
| `make cxx17` | passes — modules off, `INFINITY` defined |
| `make cxx20` | **fails** — modules on, `INFINITY` withheld |
| `make link` | **fails** — the same error, reached by linking libc++ |
| `make sdk` | the guard in the SDK header |
| `make flags` | the `-std` Zig compiles its libc++ with |
| `make modules` | what `__has_feature(modules)` is per `-std` |

`make cxx20` is also the poll: **the day it succeeds, the bug is gone.**

## The problem

The same file, two `-std` values:

```
$ zig c++ -c -std=c++17 minimal.cpp -o minimal.o     # fine
$ zig c++ -c -std=c++20 minimal.cpp -o minimal.o
minimal.cpp:12:20: error: use of undeclared identifier 'INFINITY'
```

Three facts, each harmless alone:

**1. SDK 27 made `INFINITY` conditional.** C23 moved it to `<float.h>`, so the
new `math.h` defers to it. SDK 26.x defined it unconditionally — that is what
changed:

```c
#if !defined(__GNUC__) || !defined(__has_feature) || !__has_feature(modules) \
    || !defined(__has_include) || !__has_include(<float.h>)
#define INFINITY    HUGE_VALF
#endif
```

It is defined *unless every term is false*.

**2. `-std=c++20` and later turn `__has_feature(modules)` on** (`make modules`),
which falsifies the third term. The other four are false anyway under Zig's
flags.

**3. Zig builds its bundled libc++ with `-std=c++23`** (`make flags`).

So the SDK skips the define, and libc++'s own
`__random/clamp_to_integral.h:47` uses `INFINITY` without including
`<float.h>`:

```c++
if (__r >= ::nextafter(static_cast<_RealT>(__max_val), INFINITY)) {
```

That is an upstream libc++ bug the new SDK exposes. Linking is what pulls
libc++ in, which is why compiling works and linking never does — `zig build`
reports it as `error: sub-compilation of libcxx failed`.

## Why an older SDK does not help

SDK 26.0 and 26.5 are present under `/Library/Developer/CommandLineTools/SDKs/`
and both define `INFINITY` unconditionally, but Zig picks the SDK itself via
`std.zig.system.darwin.getSdk` and ignores `-isysroot`. `SDKROOT` and
`DEVELOPER_DIR` do not redirect it either; the `CommandLineTools`
`MacOSX.sdk` symlink also resolves to 27.0 here. All three were tried.

## What it blocks in llamazig

| Works | Blocked |
|---|---|
| `zig build lib` | `zig build` |
| `zig build reference` | `zig build test`, `zig build test-port` |
| `zig c++ -c` | `make validate`, `make port`, `make ref` |
| `scripts/port-coverage` | `make graph-diff`, `make backend-ops` |
| `scripts/port-links` | `make probe`, `make parity-cli` |

Static archives are fine — producing one involves no link step. Everything that
emits an executable is blocked.

## Candidate fixes

1. **A Zig whose libc++ includes `<float.h>`** in `clamp_to_integral.h`. The
   clean fix; check a newer 0.16.x or a master build. `build.zig.zon` pins
   `minimum_zig_version`, so a bump is a deliberate change.
2. **Link the SDK's libc++** instead of Zig's, by replacing `link_libcpp` with
   a system-library link in `build.zig`. Unblocks locally, but changes the
   shared build for a local toolchain problem.
3. **An Xcode with SDK 26.x as the active developer dir**, if `xcode-select`
   can reach one.
