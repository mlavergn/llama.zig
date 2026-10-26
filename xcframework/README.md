# llama.xcframework

The llama.cpp port as a drop-in for the `llama.xcframework` upstream's
`build-xcframework.sh` builds: the same name, module, slices, architectures,
minimum OS versions, headers, module map, `Info.plist`, install names and
dSYMs. An app written against upstream's framework builds against this one
unchanged, and an app already linked against upstream's runs on this one
without relinking.

```sh
make -C xcframework            # -> xcframework/build/llama.xcframework
make -C xcframework check      # build plus every gate; see "Gates" below
make -C xcframework upstream compat   # upstream's own, and ours held to it
make -C xcframework zip        # -> build/llama.xcframework.zip + .sha256
```

From the root, `make xcframework` and `make xcframework-check` do the first
two.

## Slices

Upstream's seven, with upstream's minimums (`build-xcframework.sh:7-10`):

| Library | Platform | Minimum | arm64 CPU | x86_64 CPU |
|---|---|---|---|---|
| `macos-arm64_x86_64` | macOS | 13.3 | `apple_m1` | `x86_64_v3` |
| `ios-arm64` | iOS | 16.4 | `apple_a7` | — |
| `ios-arm64_x86_64-simulator` | iOS Simulator | 16.4 | `apple_m1` | `x86_64_v3` |
| `tvos-arm64` | tvOS | 16.4 | `apple_a7` | — |
| `tvos-arm64_x86_64-simulator` | tvOS Simulator | 16.4 | `apple_m1` | `x86_64_v3` |
| `xros-arm64` | visionOS | 1.0 | `apple_a12` | — |
| `xros-arm64_x86_64-simulator` | visionOS Simulator | 1.0 | `apple_m1` | `x86_64_v3` |

The arm64 CPU of each slice is the one Apple clang picks for that target when
nothing names one, which is what upstream's CMake does with `GGML_NATIVE`
off: measured with `clang -###`. So every device upstream's framework runs on,
this one runs on -- `apple_a7` covers every iPhone iOS 16.4 supports and every
Apple TV tvOS 16.4 does.

## Using it

Exactly as upstream's: add `llama.xcframework` to the target with **Embed &
Sign**, and

```swift
import llama

llama_backend_init()
var params = llama_model_default_params()
let model = llama_model_load_from_file(path, params)
```

or `#include <llama/llama.h>` from C, C++ or Objective-C. Upstream's own
sample, `examples/llama.swiftui`, builds against it untouched (`make
swiftui`).

**SwiftPM.** A local path package, product `llama`:

```swift
.package(path: "<repo>/xcframework"),
// ...
.target(name: "App", dependencies: [.product(name: "llama", package: "xcframework")]),
```

Run `make -C xcframework` first: the package points at
`build/llama.xcframework`, which is build output.

## Where it is not upstream's

None of these changes what an app writes or links against.

- **The CPU path is the port's**, and on macOS and the simulators it is the
  NEON one. Upstream's universal builds name two architectures in
  `CMAKE_OSX_ARCHITECTURES`, which `ggml_get_system_arch` matches as
  neither ARM nor x86, so its macOS and simulator slices compile ggml-cpu
  with `GGML_CPU_GENERIC` -- the scalar `_generic` kernels for both
  architectures. Ours are built per architecture for the CPU above: the
  same bits as the port's gated libraries, faster than upstream's there.
  The device slices compile the same kernel arms as upstream's.
- **x86_64 is `x86_64_v3` (AVX2)**, the floor of the port's x86 kernels;
  upstream's is `penryn`/`core2` running the generic kernels. Every Intel
  Mac macOS 13.3 supports has AVX2; under Rosetta the x86_64 slices need a
  Rosetta that translates AVX2. Neither has run here -- this Mac has no
  Rosetta installed -- so the x86_64 slices are built, linked and held to
  upstream's structurally, and their kernels are gated on Intel hardware
  under Windows (`make -C .. x86-gates`).
- **No C++ runtime**: the binary does not link `libc++.1.dylib`. The module
  map is upstream's and still autolinks it for the app.
- **Fewer exports**: every function the shipped headers declare that
  upstream's binary exports, ours exports (`verify.sh compat`: 993 per
  slice and architecture), and so are `ggml-quants.c`'s `quantize_*`,
  `dequantize_*`, `iq2xs_*` and `iq3xs_*`, which upstream's dylib exports
  though no shipped header declares them. Upstream's also exports its other
  internals -- C++ template instantiations, `stbi_*`, `XXH*`, `sha256_*`,
  `unicode_*` -- which the port implements in Zig under no such names.
- **Zig's compiler_rt supplies `expf`, `logf`, `sinf`, `cosf` and `exp2f`**
  inside the binary, as every bit-exact gate links them, so the CPU path
  computes the gated port's bits rather than Apple's libm's.

## Things an app must handle

- **Memory.** On Metal a 2B model's footprint is ~130 MiB. On the CPU
  the `apple_a7` slice keeps the weights memory-mapped (repacking needs
  `dotprod`), ~120 MiB on an iPhone 15; a larger model, or one built for a
  newer CPU, can need the `com.apple.developer.kernel.increased-memory-limit`
  entitlement.
- **Use the CPU on the simulator.** The simulator's GPU reports
  `MTLGPUFamilyApple2` without simdgroup reduction or matrix multiply, and
  ggml's Metal kernels produce garbage on it -- upstream's framework too.
  Select the CPU device (`ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)`
  in `llama_model_params.devices`), which `make smoke-sim` shows is
  bit-identical to the port on macOS.

## What is inside

One dylib per slice architecture, `lipo`-joined where a slice has two:
`libggml.a` (with the BLAS backend upstream's Apple build turns on) and
`libmtmd.a` (libmtmd's port, with libllama compiled into it) from `zig build
xcframework`, plus Zig's own compiler_rt, prelinked with `ld -r` so that only what
`exports.txt` names is global -- `llama_*`, `ggml_*`, `gguf_*`, `mtmd_*` and
`ggml-quants.c`'s four families -- then linked
`-dynamiclib` against Foundation, Metal and Accelerate with upstream's
install name. Device binaries are re-marked with `vtool -set-build-version`
as upstream's are; every slice ships a dSYM.

---

# Maintaining it

## Decisions

| | Decision | Why |
|---|---|---|
| Form | Upstream's: a dynamic framework per slice, dSYMs beside | drop-in: an app embeds and signs it as it does upstream's, and one linked against upstream's loads ours by the same install name |
| Name | `llama` throughout | upstream's framework, module and binary name |
| Slices | upstream's seven, arm64 and x86_64 where upstream has both | an app's every destination must find its slice |
| Minimums | upstream's: iOS 16.4, macOS 13.3, tvOS 16.4, visionOS 1.0 | a higher one would refuse to load on an OS upstream's supports |
| arm64 CPU | Apple clang's default per target | the oldest device upstream's runs on; each is gated (`GATE_CPU=apple_a7`, `apple_a12`; `apple_m1` is the baseline) |
| x86_64 CPU | `x86_64_v3` | the port's x86 floor (AVX2, gated on Intel hardware) |
| Headers, module map, Info.plist | upstream's, byte for byte (`bundle.sh`) | `verify.sh compat` holds them to upstream's own build |
| Contents | libggml + BLAS, libllama, libmtmd | what upstream's has: `LLAMA_BUILD_MTMD=ON`, `GGML_BLAS_DEFAULT=ON` |
| Toolchain | `../zig17`, Zig 0.17.0 patched for Apple's mobile SDKs | it finds every SDK and links for the simulators, which the release cannot |
| Optimize | `--release=fast` | what the bit-exact gates build |

## How a slice is built

For each architecture of each build (`make part BUILD=<build> ARCH=<arch>`):

1. `zig build xcframework -Dblas -Dtarget=<arch>-<os>.<min>[-simulator]
   -Dcpu=<cpu> --release=fast` into `build/<build>/<arch>/zig/`: `libggml.a`
   and `libmtmd.a`.
2. A probe C link with `--verbose-link` names the `libcompiler_rt.a` Zig
   built for that target; it is copied to `rt/`. The build API does not
   expose it.
3. `xcrun ld -r` with `-exported_symbols_list exports.txt` prelinks the two
   archives and compiler_rt into `llama.o`; everything not listed becomes
   local. `clang -dynamiclib` links it with upstream's install name.

Then per build: `bundle.sh` lays out the framework as upstream's
`setup_framework_structure` does, `lipo` joins the architectures,
`vtool` re-marks a device binary, `dsymutil` writes the dSYM, `strip -S`.
`xcodebuild -create-xcframework` joins the seven with their dSYMs.

## Gates (`make check`)

| Target | What it proves | Negative test |
|---|---|---|
| `verify` | per slice architecture: every object has the slice's platform and minos; the prelinked object exports `exports.txt`'s prefixes only, 0 private externs, 0 `std::__1`, compiler_rt math inside, the entry points of every library present; `test/stub.m` links with nothing but `-framework llama`; the binary needs no libc++; per slice: `Info.plist` lints, the dSYM's UUIDs match the binary's and it has DWARF; exactly upstream's seven libraries | `make negative`: wrong platform, wrong minos, no exports list, `-keep_private_externs`, no compiler_rt, no libmtmd, malformed plist, another slice's dSYM, a two-slice xcframework |
| `compat` | against `make upstream`'s build: the same libraries; every slice's bundle tree and every file but the binary byte-identical; each binary's architectures, type, install name, platform, minos and SDK identical, no dependency upstream's lacks; every header-declared function upstream exports, exported by ours, per slice and architecture | a header edited, the module renamed, `mtmd_*` dropped from one architecture, a device binary not re-marked |
| `smoke-macos` | `test/smoke.swift` through the framework is byte-identical to `test/driver.c` on the port's own libraries, six prompts, GPU and CPU; `test/coexist.cpp` runs app C++ beside it | one reference token altered |
| `dropin-macos` | the same Swift binary, linked against upstream's framework, run with upstream's and then ours in its `Frameworks/` -- unrelinked: Metal identical to upstream's, and to the port; CPU identical to the port | — |
| `smoke-sim` | the iOS, tvOS and visionOS simulators, CPU, against the port's `apple_m1` libraries | — |
| `smoke-device` | a physical iPhone (`DEVICE_ID`, signed for `TEAM`), CPU **and Metal**: `test/device`, an iOS app embedding the framework, against `test/driver.c` on the port's `apple_a7` libraries here. Not in `check`, which needs no phone | one device token altered |
| `swiftui` | upstream's `examples/llama.swiftui`, untouched, builds against ours for the simulator, and builds, installs and is still running on `DEVICE_ID` after launch. Not in `check` | — |
| `pkg-test` | `Package.swift` builds and runs on macOS, builds for the simulator | no xcframework: the error names it |
| `zip` | `.zip` and its SwiftPM checksum | — |
| `port-gates` | the root `make validate ops-diff` | — |

**On an iPhone 15** (A16, iOS 27.0.1), `smoke-device` passes 12 / 12 --
the `apple_a7` slice on the CPU and Metal byte-identical to the port here --
and `swiftui` builds upstream's sample against ours, installs it and finds
it running. Neither is in `check`: they need a phone registered in the
signing team.

`check` runs `compat` and `dropin-macos` too when `build/upstream` exists;
`make upstream` builds it with CMake from `../.tools` (`make -C .. cmake`)
over a `git archive` of the pinned `../llama.cpp`, which it never writes to.

**The device slices' CPUs are gated from the root**, not here: `smoke-device`
proves the framework computes what the port computes at `apple_a7`; that the
port computes what upstream does there is `make -C .. ops-diff repack-diff
node-diff parity-port-cpu GATE_CPU=apple_a7` (and `apple_a12` for visionOS).

## Things that were learned the hard way

- **Upstream's universal slices run generic CPU kernels.**
  `ggml_get_system_arch` (`ggml/cmake/common.cmake:28`) takes
  `CMAKE_OSX_ARCHITECTURES` only when it is exactly `arm64` or `x86_64`;
  `arm64;x86_64` falls through to `UNKNOWN` and `GGML_CPU_GENERIC`.
- **The CPU model's vectorizer shape is not one per vendor.** The port's
  `simd.strict_loop` keyed every `apple_*` model to the A14's interleave of 4;
  LLVM sets `FeatureMaxInterleaveFactor4` only from the A14 on, and the A7
  to A13 vectorize as `generic` does. `ops-diff` at `GATE_CPU=apple_a7`
  failed on `ssm_conv`, `ssm_scan` and `conv_3d` until it keyed on the
  feature (`NOTES.md`, "llama.xcframework").
- **Upstream's module map does not link the framework.** Swift autolinks
  only what a module map names, so a bare `swiftc -F <slice>` resolves the
  headers and fails at link time; name `-framework llama`, as Xcode does
  when the framework is in the target. The static framework's own module
  map linked itself; drop-in means upstream's text.
- **Zig 0.17 binds `readdir$INODE64` on every x86_64 Darwin target**, and
  only macOS's libSystem has it: the simulator SDKs are 64-bit-inode only.
  The x86_64 simulator slices failed to link until `backend_reg.zig` bound
  plain `readdir` there.
- **libm decides the CPU path's bits.** A `zig cc` link -- every bit-exact
  gate and `make port` -- defines `expf`, `logf`, `sinf`, `cosf`, `exp2f`
  inside the binary from compiler_rt; an Xcode or `swiftc` link takes Apple's
  libm. Without compiler_rt in the prelink, 3 of 6 CPU prompts diverged from
  the port while Metal stayed 6/6. `verify.sh symbols` requires the five.
- **Zig 0.17 cannot compile `std.debug.print` for iOS** (`std.Io.Threaded`'s
  iOS `NullFile` has no `fd`). Library code writes to C's `stderr` through
  `impl.writeStderr` on every target instead; `make validate` builds the iOS
  device slice to keep it so.
- **No `| grep -q` under `pipefail`** in the gates: its early exit SIGPIPEs
  the writer and the pipeline reads as a miss. It failed a real gate once.
- **An app that works before it finishes launching is killed at 20 s.**
  The first device app ran the prompts in `main()` and died with SIGKILL
  partway through, which read as memory pressure and is not: the report
  says `0x8BADF00D`, "process-launch watchdog transgression". The work runs
  on a thread started after launch. Read the termination reason (`xcrun
  devicectl device info files --domain-type systemCrashLogs`) before
  guessing from a signal 9.
- **Repacked weights are anonymous memory.** The static framework's
  `apple_a14` slice repacked a 2B model's weights for the CPU, ~1.3 GiB,
  and without the increased-memory-limit entitlement the app was jetsammed
  on the first prompt. The `apple_a7` slice does not repack; the test app
  keeps the entitlement.
- **The first launch after an install can return none of the app's console
  output** though the app ran to completion; `smoke-device` launches again
  when no `SMOKE-DEVICE` line came back.
- **`nm -m` prints `non-external (was a private external)`** for a symbol
  `ld -r` *did* localize. Grep for `private external` alone and you count
  thousands of false alarms.
