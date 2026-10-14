# llamazig.xcframework

The llama.cpp port as a static framework for macOS and iOS apps: the `llama.h`
C API, callable from Swift as `import llamazig`.

```sh
make -C xcfwk          # -> xcfwk/build/llamazig.xcframework
make -C xcfwk check    # build plus every gate; see "Gates" below
make -C xcfwk zip      # -> build/llamazig.xcframework.zip + .sha256
```

## Slices

| Slice | Platform | Minimum | CPU |
|---|---|---|---|
| `macos-arm64` | macOS | 26.0 | Apple M1 and later |
| `ios-arm64` | iOS | 18.0 | **A14 and later** (iPhone 12+) |
| `ios-arm64-simulator` | iOS Simulator | 18.0 | Apple silicon Macs |

No x86_64: neither Intel Macs nor the Intel simulator.

## Using it

**Xcode.** Add `llamazig.xcframework` to the target and set it to **Do Not
Embed**. It is static: there is nothing to embed or sign. The module map links
Metal, MetalKit, Accelerate and Foundation itself. Do **not** add `-lc++`; the
framework carries its own C++ runtime, hidden.

**SwiftPM.** A local path package:

```swift
.package(path: "<repo>/xcfwk"),
// ...
.target(name: "App", dependencies: [.product(name: "llamazig", package: "xcfwk")]),
```

Run `make -C xcfwk` first: the package points at
`build/llamazig.xcframework`, which is build output.

```swift
import llamazig

llama_backend_init()
var params = llama_model_default_params()
let model = llama_model_load_from_file(path, params)
```

## Moving from upstream's `llama.framework`

Same C API and the same headers, minus `mtmd.h`, `mtmd-helper.h` and
`ggml-blas.h` (text only). Change:

- `import llama` → `import llamazig`
- `#include <llama/llama.h>` → `#include <llamazig/llama.h>`
- *Embed & Sign* → *Do Not Embed* (static, where upstream's is dynamic)

## Things an app must handle

- **Pre-A14 iPhones are unsupported.** iOS 18 still runs on A12/A13 (iPhone
  XS, XR, 11, SE 2); the code is built for A14 and may crash there with an
  illegal instruction. Gate devices in the app.
- **Memory.** A 2B-parameter model needs the
  `com.apple.developer.kernel.increased-memory-limit` entitlement on iOS.
- **Use the CPU on the simulator.** The simulator's GPU reports
  `MTLGPUFamilyApple2` without simdgroup reduction or matrix multiply, and
  ggml's Metal kernels produce garbage on it. Select the CPU device
  (`ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)` in
  `llama_model_params.devices`), which `make smoke-sim` shows is bit-identical
  to the port on macOS.

## What is inside

One object per slice: `libggml.a` and `libllama.a` from `zig build reference`,
plus Zig's own libc++, libc++abi and compiler_rt, prelinked with `ld -r` so
that only `llama_*`, `ggml_*` and `gguf_*` symbols are exported. The bundled
compiler_rt matters: it supplies `expf`, `logf`, `sinf`, `cosf` and `exp2f`
exactly as every bit-exact gate links them, so the CPU path computes the same
bits as the gated port rather than Apple's libm's.

---

# Maintaining it

## Decisions

| | Decision | Why |
|---|---|---|
| Form | Static framework, one `ld -r`-prelinked object | keeps Zig's libc++ and compiler_rt -- the gated runtime -- without an app ever linking or meeting them |
| Slices | `macos-arm64`, `ios-arm64`, `ios-arm64-simulator` | x86 CPU kernels are not ported |
| Minimums | iOS 18.0, macOS 26.0 | chosen; Zig targets `aarch64-ios.18.0`, `aarch64-macos.26.0` |
| iOS CPU | `apple_a14` | the `apple_m1` feature set every gate runs; A12/A13 left out, documented |
| Name | `llamazig` throughout | a framework module's name must equal its framework's, so `import llama` was not an option |
| Headers | upstream's set minus `mtmd*.h`, `ggml-blas.h` | text only |
| SwiftPM | `Package.swift` here, `binaryTarget(path:)` | local path package; the zip and checksum are a hand-off artifact |
| Optimize | `--release=fast` | what the bit-exact gates build |

## How a slice is built

1. `zig build reference -Dtarget=<slice> -Dcpu=<cpu> --release=fast` into
   `build/<slice>/zig/`. macOS is `aarch64-macos.26.0`, baseline (`apple_m1`);
   iOS and the simulator are `aarch64-ios.18.0[-simulator]`, `apple_a14`, the
   same feature set.
2. A probe C++ link with `--verbose-link` names the `libc++.a`,
   `libc++abi.a` and `libcompiler_rt.a` Zig built for that target; they are
   copied to `build/<slice>/cxx/`. The build API does not expose them.
3. `xcrun ld -r` with `-exported_symbols_list exports.txt` prelinks the two
   archives and the three runtime archives into `llamazig.o`; everything
   not listed becomes local. `strip -S`, `libtool -static`.
4. The bundle: flat for iOS, `Versions/A` for macOS; headers copied from
   `llama.cpp/`, `framework.modulemap` as `Modules/module.modulemap`,
   `Info.plist.in` filled in. `xcodebuild -create-xcframework` joins them.

## Gates (`make check`)

| Target | What it proves | Negative test |
|---|---|---|
| `verify` | every object in every archive and the prelinked object has the slice's platform and minos; 1,438 exports, all `llama_`/`ggml_`/`gguf_`; 0 private externs; 0 `std::__1`; compiler_rt math inside; `Info.plist` lints; `test/stub.m` links per slice, device included, with no `-lc++` and no frameworks named; exactly the three slices | `make negative`: wrong platform, wrong minos, no exports list, `-keep_private_externs`, no compiler_rt, no libc++abi, module map without `link`, malformed plist, two-slice xcframework |
| `smoke-macos` | `test/smoke.swift` through the framework is byte-identical to `test/driver.c` on the port's own libraries, six prompts, GPU and CPU; `test/coexist.cpp` runs app C++ beside the hidden libc++ | one reference token altered |
| `smoke-sim` | the same on the iOS simulator, CPU | — |
| `pkg-test` | `Package.swift` builds and runs on macOS, builds for the simulator | no xcframework: the error names it |
| `zip` | `.zip` and its SwiftPM checksum | — |
| `port-gates` | the root `make validate ops-diff` | — |

Not automated: running on a device, which needs A14+ hardware.

## Things that were learned the hard way

- **libm decides the CPU path's bits.** A `zig cc` link -- every bit-exact
  gate and `make port` -- defines `expf`, `logf`, `sinf`, `cosf`, `exp2f`
  inside the binary from compiler_rt; an Xcode or `swiftc` link takes Apple's
  libm. Without compiler_rt in the prelink, 3 of 6 CPU prompts diverged from
  the port while Metal stayed 6/6. `verify.sh symbols` requires the five.
- **Zig 0.17 cannot compile `std.debug.print` for iOS** (`std.Io.Threaded`'s
  iOS `NullFile` has no `fd`). `grammar/parser.zig`, the one site in
  `libllama.a`, writes to fd 2 inline on iOS only.
- **Zig 0.17's linker misreads the simulator SDK version** from
  `MinimalDisplayName` (`"Simulator - 27.0"`). Only linking hits it; the
  probe links against a shim SDK whose `SDKSettings.json` fixes the key.
- **Swift autolinks only what a module map names**, not a framework module's
  own binary -- hence `link framework "llamazig"`.
- **The template is `framework.modulemap`, never `module.modulemap`**: this
  directory is a SwiftPM package root, and clang took a `module.modulemap`
  here for the module, resolved `umbrella "Headers"` against it, and imported
  nothing.
- **No `| grep -q` under `pipefail`** in the gates: its early exit SIGPIPEs
  the writer and the pipeline reads as a miss. It failed a real gate once.
- **`nm -m` prints `non-external (was a private external)`** for a symbol
  `ld -r` *did* localize. Grep for `private external` alone and you count
  12,620 false alarms.

