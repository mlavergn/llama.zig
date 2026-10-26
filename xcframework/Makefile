# llama.xcframework: the port as a drop-in for the xcframework upstream's
# build-xcframework.sh produces. Same name, module, slices, architectures,
# minimum OS versions, headers, module map, Info.plist, install names and
# dSYMs; see README.md for consumers, the decisions behind it, and the gates.
#
#   make              build/llama.xcframework
#   make verify       the structural gates, on every slice and the xcframework
#   make upstream     upstream's own llama.xcframework, built by its script
#                     from the pinned sources copied out of ../llama.cpp
#   make compat       ours against upstream's: every file but the binaries
#                     byte-identical, the binaries' architectures, platforms,
#                     install names and every API symbol upstream exports
#   make negative     every gate fed a broken input, each required to fail
#   make smoke-macos  Swift through the framework vs the port's own libraries,
#                     six prompts, GPU and CPU; plus app C++ beside it
#   make dropin-macos an app linked against upstream's framework, run with
#                     ours swapped into its bundle, unrelinked
#   make smoke-sim    the iOS, tvOS and visionOS simulators, CPU
#   make smoke-device a physical iPhone (DEVICE_ID), CPU and Metal, signed for
#                     TEAM; not part of `check`, which needs no phone
#   make swiftui      upstream's examples/llama.swiftui, unmodified, built
#                     against ours for the simulator, and for and launched on
#                     DEVICE_ID; not part of `check`
#   make pkg-test     Package.swift consumed by a package, macOS and simulator
#   make zip          build/llama.xcframework.zip and its SwiftPM checksum
#   make port-gates   the root `make validate ops-diff`, for the port changes
#   make check        verify negative smoke-macos smoke-sim pkg-test zip
#                     port-gates; compat and dropin-macos too when
#                     build/upstream exists
#
# Each slice architecture is `zig build xcframework` for its target -- libggml
# with the BLAS backend upstream's Apple build has, and libmtmd with libllama
# inside it -- then `ld -r` of those and Zig's own compiler_rt into one object
# that exports only what exports.txt names, linked as the framework's dylib.
# compiler_rt goes in so the CPU path computes with the same math the gated
# port does; no C++ runtime goes in, since neither library references one.

SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c

ROOT       := $(abspath $(CURDIR)/..)
B          := $(CURDIR)/build
XCF        := $(B)/llama.xcframework
UPSTREAM   := $(B)/upstream/build-apple/llama.xcframework
# The patched Zig 0.17 (../zig17): it finds every Apple SDK itself and links
# for the simulators, where the release misreads SDKSettings.json.
ZIG        ?= $(ROOT)/../zig17/stage/zig/zig
# The root gates and the reference drivers build with the `zig` on PATH.
HOST_ZIG   ?= zig

# Upstream's minimums (build-xcframework.sh:7-10).
IOS_MIN      := 16.4
MACOS_MIN    := 13.3
VISIONOS_MIN := 1.0
TVOS_MIN     := 16.4

MODEL      ?= $(ROOT)/Qwen3.5-2B-Q4_K_M.gguf
NPRED      ?= 32
SIM_DEVICE ?= iPhone 17 Pro
TV_DEVICE  ?= Apple TV 4K (3rd generation)
XR_DEVICE  ?= Apple Vision Pro
# smoke-device and swiftui: the phone (a UDID from `xcrun devicectl list
# devices`) and the development team its apps are signed for. The phone must
# be registered in that team.
DEVICE_ID  ?= 00008120-00112894262A601E
TEAM       ?= Q6H2FB9YW2
DEVICE_APP := com.mlavergn.llamazig.devicesmoke
SWIFTUI_APP := com.mlavergn.llamazig.swiftui

# upstream's build order (build-xcframework.sh:55)
BUILDS := ios-sim ios-device macos visionos visionos-sim tvos-sim tvos-device

# Per build: architectures; the Zig OS and ABI; the arm64 CPU; SDK; the
# `ld -platform_version` name; Mach-O platform number; the clang OS for
# `-target`; the bundle platform bundle.sh takes; minimum OS; xcframework
# library identifier; the binary's place in the bundle and its install name;
# and the `vtool -set-build-version` platform upstream re-marks device
# binaries with.
#
# The arm64 CPU is the one Apple clang picks for that target when upstream's
# CMake names none (GGML_NATIVE is off): measured with `clang -###`. It
# decides which kernel arms compile, and the oldest device each slice runs
# on: apple_a7 runs on everything iOS 16.4 and tvOS 16.4 do. x86_64 is
# x86_64_v3 (AVX2), the floor of the port's x86 kernels; every Mac macOS 13.3
# supports has it.
ios-sim_ARCHS        := arm64 x86_64
ios-sim_OS           := ios.$(IOS_MIN)-simulator
ios-sim_CPU          := apple_m1
ios-sim_SDK          := iphonesimulator
ios-sim_LD           := ios-simulator
ios-sim_PLAT         := 7
ios-sim_CLANG_OS     := ios$(IOS_MIN)-simulator
ios-sim_BUNDLE       := ios
ios-sim_MIN          := $(IOS_MIN)
ios-sim_XCF_ID       := ios-arm64_x86_64-simulator

ios-device_ARCHS     := arm64
ios-device_OS        := ios.$(IOS_MIN)
ios-device_CPU       := apple_a7
ios-device_SDK       := iphoneos
ios-device_LD        := ios
ios-device_PLAT      := 2
ios-device_CLANG_OS  := ios$(IOS_MIN)
ios-device_BUNDLE    := ios
ios-device_MIN       := $(IOS_MIN)
ios-device_XCF_ID    := ios-arm64
ios-device_VTOOL     := ios

macos_ARCHS          := arm64 x86_64
macos_OS             := macos.$(MACOS_MIN)
macos_CPU            := apple_m1
macos_SDK            := macosx
macos_LD             := macos
macos_PLAT           := 1
macos_CLANG_OS       := macos$(MACOS_MIN)
macos_BUNDLE         := macos
macos_MIN            := $(MACOS_MIN)
macos_XCF_ID         := macos-arm64_x86_64

visionos_ARCHS       := arm64
visionos_OS          := visionos.$(VISIONOS_MIN)
visionos_CPU         := apple_a12
visionos_SDK         := xros
visionos_LD          := xros
visionos_PLAT        := 11
visionos_CLANG_OS    := xros$(VISIONOS_MIN)
visionos_BUNDLE      := visionos
visionos_MIN         := $(VISIONOS_MIN)
visionos_XCF_ID      := xros-arm64
visionos_VTOOL       := visionos

visionos-sim_ARCHS   := arm64 x86_64
visionos-sim_OS      := visionos.$(VISIONOS_MIN)-simulator
visionos-sim_CPU     := apple_m1
visionos-sim_SDK     := xrsimulator
visionos-sim_LD      := xros-simulator
visionos-sim_PLAT    := 12
visionos-sim_CLANG_OS := xros$(VISIONOS_MIN)-simulator
visionos-sim_BUNDLE  := visionos
visionos-sim_MIN     := $(VISIONOS_MIN)
visionos-sim_XCF_ID  := xros-arm64_x86_64-simulator

tvos-sim_ARCHS       := arm64 x86_64
tvos-sim_OS          := tvos.$(TVOS_MIN)-simulator
tvos-sim_CPU         := apple_m1
tvos-sim_SDK         := appletvsimulator
tvos-sim_LD          := tvos-simulator
tvos-sim_PLAT        := 8
tvos-sim_CLANG_OS    := tvos$(TVOS_MIN)-simulator
tvos-sim_BUNDLE      := tvos
tvos-sim_MIN         := $(TVOS_MIN)
tvos-sim_XCF_ID      := tvos-arm64_x86_64-simulator

tvos-device_ARCHS    := arm64
tvos-device_OS       := tvos.$(TVOS_MIN)
tvos-device_CPU      := apple_a7
tvos-device_SDK      := appletvos
tvos-device_LD       := tvos
tvos-device_PLAT     := 3
tvos-device_CLANG_OS := tvos$(TVOS_MIN)
tvos-device_BUNDLE   := tvos
tvos-device_MIN      := $(TVOS_MIN)
tvos-device_XCF_ID   := tvos-arm64
tvos-device_VTOOL    := tvos

# The binary's path inside the bundle and its install name, as upstream's
# `combine_static_libraries` sets them.
bin_path     = $(if $(filter macos,$(1)),Versions/A/llama,llama)
install_name = $(if $(filter macos,$(1)),@rpath/llama.framework/Versions/Current/llama,@rpath/llama.framework/llama)

zig_arch = $(if $(filter arm64,$(1)),aarch64,x86_64)
# $(1): build, $(2): arch
part_cpu = $(if $(filter arm64,$(2)),$($(1)_CPU),x86_64_v3)

# Runs a command that must fail, and fails if it does not.
# $(1): the command; $(2): what the injected fault is.
expect_fail = if ( $(1) ) > $(B)/neg/last.log 2>&1; then \
        echo "NEGATIVE TEST ESCAPED: $(2)"; cat $(B)/neg/last.log; exit 1; \
    else echo "  caught: $(2) -- $$(grep -m1 -E 'FAIL|error|differ' $(B)/neg/last.log)"; fi

.PHONY: all xcframework $(BUILDS) part verify $(addprefix verify-,$(BUILDS)) upstream compat negative \
        ref-driver smoke-macos dropin-macos smoke-sim smoke-device swiftui pkg-test zip port-gates check clean

all: xcframework

# -----------------------------------------------------------------------------
# Slices

# One architecture of one build: `make part BUILD=<build> ARCH=<arch>`.
P = $(B)/$(BUILD)/$(ARCH)
part:
	@printf '\n== %s %s: zig build xcframework -Dtarget=%s-%s -Dcpu=%s ==\n' $(BUILD) $(ARCH) \
	    $(call zig_arch,$(ARCH)) $($(BUILD)_OS) $(call part_cpu,$(BUILD),$(ARCH))
	cd $(ROOT) && $(ZIG) build xcframework -Dblas -Dtarget=$(call zig_arch,$(ARCH))-$($(BUILD)_OS) \
	    -Dcpu=$(call part_cpu,$(BUILD),$(ARCH)) --release=fast --prefix $(P)/zig
	@# A probe link with --verbose-link names the libcompiler_rt.a Zig built
	@# for this target; the build API does not expose it. It is not
	@# optional: a `zig cc` link -- every bit-exact gate, and `make port` --
	@# takes expf, logf, sinf, cosf, exp2f from it rather than libSystem, and
	@# with Apple's libm instead 3 of 6 CPU prompts diverged from the port.
	@# A fresh cache dir forces the link to run, so the line is always
	@# printed.
	rm -rf $(P)/rt && mkdir -p $(P)/rt
	printf 'int main(void) { return 0; }\n' > $(P)/rt/probe.c
	cd $(P)/rt && $(ZIG) build-exe probe.c -lc -target $(call zig_arch,$(ARCH))-$($(BUILD)_OS) \
	    -mcpu $(call part_cpu,$(BUILD),$(ARCH)) -OReleaseFast --cache-dir cache --verbose-link 2>&1 \
	    | tr ' ' '\n' | grep -E '/libcompiler_rt\.a$$' > paths.txt
	test "$$(wc -l < $(P)/rt/paths.txt)" -eq 1 || { echo "FAIL: probe found: $$(cat $(P)/rt/paths.txt)"; exit 1; }
	cp $$(cat $(P)/rt/paths.txt) $(P)/rt/
	@printf -- '-- prelink and link\n'
	xcrun ld -r -arch $(ARCH) -platform_version $($(BUILD)_LD) $($(BUILD)_MIN) $$(xcrun --sdk $($(BUILD)_SDK) --show-sdk-version) \
	    -exported_symbols_list exports.txt \
	    -force_load $(P)/zig/lib/libggml.a -force_load $(P)/zig/lib/libmtmd.a \
	    $(P)/rt/libcompiler_rt.a -o $(P)/llama.o
	xcrun --sdk $($(BUILD)_SDK) clang -dynamiclib -target $(ARCH)-apple-$($(BUILD)_CLANG_OS) \
	    -isysroot "$$(xcrun --sdk $($(BUILD)_SDK) --show-sdk-path)" $(P)/llama.o \
	    -framework Foundation -framework Metal -framework Accelerate \
	    -install_name $(call install_name,$(BUILD)) -o $(P)/llama.dylib

# A build: its architectures, joined, in upstream's bundle, with its dSYM.
$(BUILDS): %:
	@for a in $($*_ARCHS); do $(MAKE) --no-print-directory part BUILD=$* ARCH=$$a; done
	@printf '\n== %s: framework ==\n' $*
	rm -rf $(B)/$*/framework $(B)/$*/dSYMs && mkdir -p $(B)/$*/framework $(B)/$*/dSYMs
	./bundle.sh $(ROOT)/llama.cpp $(B)/$*/framework $($*_MIN) $($*_BUNDLE)
	BIN=$(B)/$*/framework/llama.framework/$(call bin_path,$*); \
	xcrun lipo -create $(foreach a,$($*_ARCHS),$(B)/$*/$(a)/llama.dylib) -output $$BIN; \
	if [ -n "$($*_VTOOL)" ]; then \
	    xcrun vtool -set-build-version $($*_VTOOL) $($*_MIN) $($*_MIN) -replace -output $$BIN $$BIN; \
	fi; \
	xcrun dsymutil $$BIN -o $(B)/$*/dSYMs/llama.dSYM; \
	xcrun strip -S $$BIN -o $(B)/$*/stripped && mv $(B)/$*/stripped $$BIN

xcframework: $(BUILDS)
	@printf '\n== xcframework ==\n'
	rm -rf $(XCF)
	xcrun xcodebuild -create-xcframework \
	    $(foreach s,$(BUILDS),-framework $(B)/$(s)/framework/llama.framework -debug-symbols $(B)/$(s)/dSYMs/llama.dSYM) \
	    -output $(XCF)

# -----------------------------------------------------------------------------
# Gates

verify: xcframework
	@printf '\n== verify ==\n'
	@$(MAKE) --no-print-directory $(addprefix verify-,$(BUILDS))
	@printf -- '-- xcframework\n'
	./verify.sh xcframework $(XCF)

$(addprefix verify-,$(BUILDS)): verify-%:
	@printf -- '-- %s\n' $*
	@for a in $($*_ARCHS); do \
	    ./verify.sh platform $($*_PLAT) $($*_MIN) $(B)/$*/$$a/zig/lib/libggml.a $(B)/$*/$$a/zig/lib/libmtmd.a \
	        $(B)/$*/$$a/rt/libcompiler_rt.a $(B)/$*/$$a/llama.o; \
	    ./verify.sh symbols $(B)/$*/$$a/llama.o; \
	    ./verify.sh link $($*_SDK) $$a-apple-$($*_CLANG_OS) $(XCF)/$($*_XCF_ID) $(B)/test/stub-$*-$$a; \
	done
	./verify.sh plist $(XCF)/$($*_XCF_ID)/llama.framework/$(if $(filter macos,$*),Versions/A/Resources/Info.plist,Info.plist)
	./verify.sh dsym $(XCF)/$($*_XCF_ID)/llama.framework/$(call bin_path,$*) $(XCF)/$($*_XCF_ID)/dSYMs/llama.dSYM

# Upstream's own xcframework, by its own script, over a copy of the pinned
# sources -- `git archive`, so nothing is written under ../llama.cpp. It
# needs CMake: `make -C .. cmake` puts one in ../.tools.
CMAKE_BIN := $(ROOT)/.tools/cmake-4.4.3-macos-universal/CMake.app/Contents/bin
upstream:
	@printf '\n== upstream: build-xcframework.sh at %s ==\n' $$(git -C $(ROOT)/llama.cpp rev-parse --short HEAD)
	rm -rf $(B)/upstream && mkdir -p $(B)/upstream
	git -C $(ROOT)/llama.cpp archive HEAD | tar -x -C $(B)/upstream
	cd $(B)/upstream && PATH="$(CMAKE_BIN):$$PATH" ./build-xcframework.sh > $(B)/upstream.log 2>&1 \
	    || { tail -30 $(B)/upstream.log; exit 1; }
	@ls -d $(UPSTREAM)

compat:
	@printf '\n== compat: ours against upstream ==\n'
	@test -d $(UPSTREAM) || { echo "FAIL: no $(UPSTREAM) -- run 'make upstream'"; exit 1; }
	./verify.sh compat $(XCF) $(UPSTREAM)

# Every structural gate, fed an input with one fault injected.
NEG_O := $(B)/macos/arm64
negative: xcframework
	@printf '\n== negative tests ==\n'
	rm -rf $(B)/neg && mkdir -p $(B)/neg
	@# Fault: wrong platform -- the simulator object checked as a device one.
	@$(call expect_fail,./verify.sh platform $(ios-device_PLAT) $(IOS_MIN) $(B)/ios-sim/arm64/llama.o,simulator object passed as iOS device)
	@# Fault: wrong minimum OS.
	@$(call expect_fail,./verify.sh platform $(macos_PLAT) 15.0 $(NEG_O)/llama.o,macOS $(MACOS_MIN) object passed as minos 15.0)
	@# Fault: no exports list, so every internal symbol stays global.
	xcrun ld -r -arch arm64 -platform_version macos $(MACOS_MIN) $$(xcrun --sdk macosx --show-sdk-version) \
	    -force_load $(NEG_O)/zig/lib/libggml.a -force_load $(NEG_O)/zig/lib/libmtmd.a \
	    $(NEG_O)/rt/libcompiler_rt.a -o $(B)/neg/noexports.o
	@$(call expect_fail,./verify.sh symbols $(B)/neg/noexports.o,prelink without -exported_symbols_list)
	@# Fault: -keep_private_externs, so nothing is localized.
	xcrun ld -r -arch arm64 -platform_version macos $(MACOS_MIN) $$(xcrun --sdk macosx --show-sdk-version) \
	    -keep_private_externs -exported_symbols_list exports.txt \
	    -force_load $(NEG_O)/zig/lib/libggml.a -force_load $(NEG_O)/zig/lib/libmtmd.a \
	    $(NEG_O)/rt/libcompiler_rt.a -o $(B)/neg/private.o
	@$(call expect_fail,./verify.sh symbols $(B)/neg/private.o,prelink with -keep_private_externs)
	@# Fault: compiler_rt left out, so expf and friends come from Apple's libm.
	xcrun ld -r -arch arm64 -platform_version macos $(MACOS_MIN) $$(xcrun --sdk macosx --show-sdk-version) \
	    -exported_symbols_list exports.txt \
	    -force_load $(NEG_O)/zig/lib/libggml.a -force_load $(NEG_O)/zig/lib/libmtmd.a \
	    -o $(B)/neg/nocrt.o
	@$(call expect_fail,./verify.sh symbols $(B)/neg/nocrt.o,prelink without compiler_rt)
	@# Fault: libmtmd left out, so the mtmd_* API is missing.
	xcrun ld -r -arch arm64 -platform_version macos $(MACOS_MIN) $$(xcrun --sdk macosx --show-sdk-version) \
	    -exported_symbols_list exports.txt -force_load $(NEG_O)/zig/lib/libggml.a \
	    $(NEG_O)/rt/libcompiler_rt.a -o $(B)/neg/nomtmd.o
	@$(call expect_fail,./verify.sh symbols $(B)/neg/nomtmd.o,prelink without libmtmd)
	@# Fault: a malformed Info.plist.
	sed 's#</dict>##' $(XCF)/ios-arm64/llama.framework/Info.plist > $(B)/neg/Info.plist
	@$(call expect_fail,./verify.sh plist $(B)/neg/Info.plist,Info.plist missing </dict>)
	@# Fault: a dSYM from another build.
	@$(call expect_fail,./verify.sh dsym $(XCF)/ios-arm64/llama.framework/llama $(XCF)/tvos-arm64/dSYMs/llama.dSYM,dSYM of the tvOS binary for the iOS one)
	@# Fault: an xcframework with a slice missing.
	xcrun xcodebuild -create-xcframework -framework $(B)/macos/framework/llama.framework -framework $(B)/ios-device/framework/llama.framework \
	    -output $(B)/neg/two.xcframework > /dev/null
	@$(call expect_fail,./verify.sh xcframework $(B)/neg/two.xcframework,xcframework without five of its slices)
	@if [ -d $(UPSTREAM) ]; then $(MAKE) --no-print-directory negative-compat; fi
	@echo "  PASS: every gate failed on its injected fault"

# compat's own faults, against a copy of ours with one thing changed.
.PHONY: negative-compat
negative-compat:
	@# Fault: one header differs.
	rm -rf $(B)/neg/hdr && cp -R $(XCF) $(B)/neg/hdr
	chmod u+w $(B)/neg/hdr/ios-arm64/llama.framework/Headers/llama.h
	echo '// edited' >> $(B)/neg/hdr/ios-arm64/llama.framework/Headers/llama.h
	@$(call expect_fail,./verify.sh compat $(B)/neg/hdr $(UPSTREAM),a header that is not upstream's)
	@# Fault: our own module name, as the static framework had.
	rm -rf $(B)/neg/mod && cp -R $(XCF) $(B)/neg/mod
	sed -i '' 's/^framework module llama/framework module llamazig/' $(B)/neg/mod/tvos-arm64/llama.framework/Modules/module.modulemap
	@$(call expect_fail,./verify.sh compat $(B)/neg/mod $(UPSTREAM),module renamed)
	@# Fault: an API function no longer exported.
	rm -rf $(B)/neg/api && cp -R $(XCF) $(B)/neg/api
	grep -v '^_mtmd_' exports.txt > $(B)/neg/exports-nomtmd.txt
	xcrun clang -dynamiclib -target arm64-apple-macos$(MACOS_MIN) $(NEG_O)/llama.o \
	    -framework Foundation -framework Metal -framework Accelerate -Wl,-exported_symbols_list,$(B)/neg/exports-nomtmd.txt \
	    -install_name @rpath/llama.framework/Versions/Current/llama -o $(B)/neg/arm64.dylib
	xcrun lipo -create $(B)/neg/arm64.dylib $(B)/macos/x86_64/llama.dylib -output $(B)/neg/api/macos-arm64_x86_64/llama.framework/Versions/A/llama
	@$(call expect_fail,./verify.sh compat $(B)/neg/api $(UPSTREAM),mtmd_* not exported on macOS arm64)
	@# Fault: the device binary not re-marked by vtool, so its SDK field differs.
	rm -rf $(B)/neg/vt && cp -R $(XCF) $(B)/neg/vt
	cp $(B)/tvos-device/arm64/llama.dylib $(B)/neg/vt/tvos-arm64/llama.framework/llama
	@$(call expect_fail,./verify.sh compat $(B)/neg/vt $(UPSTREAM),tvOS binary without vtool)

# -----------------------------------------------------------------------------
# Smoke tests

# The reference: test/driver.c against the port's own libraries, as
# scripts/parity-port links its port side, with the BLAS backend the framework
# has -- one per CPU a slice is built for, because the CPU comparison is only
# exact when both sides compiled the same kernel arms. Each is installed to a
# prefix of its own rather than zig-out/, which the root gates rebuild for
# whatever CPU they are run at. driver-<cpu>: apple_m1 for macOS and the
# simulators, apple_a7 for the iOS device.
DRIVER_CPUS := apple_m1 apple_a7
ref-driver: $(addprefix $(B)/test/driver-,$(DRIVER_CPUS))

$(B)/test/driver-%: test/driver.c
	@printf '\n== reference driver for %s (zig build reference -Dblas -Dcpu=%s) ==\n' $* $*
	cd $(ROOT) && $(HOST_ZIG) build reference --release=fast -Dblas -Dcpu=$* --prefix $(B)/ref-$*
	mkdir -p $(B)/test
	cd $(ROOT) && $(HOST_ZIG) cc -mcpu=$* $(CURDIR)/test/driver.c -o $@ -std=c11 \
	    -I llama.cpp/include -I llama.cpp/ggml/include $(B)/ref-$*/lib/libllama.a $(B)/ref-$*/lib/libggml.a \
	    -lc++ -framework Foundation -framework Metal -framework MetalKit -framework Accelerate \
	    -F "$$(xcrun --sdk macosx --show-sdk-path)/System/Library/Frameworks"

MACOS_FW := $(XCF)/$(macos_XCF_ID)

smoke-macos: xcframework $(B)/test/driver-apple_m1
	@printf '\n== smoke: macOS ==\n'
	xcrun --sdk macosx swiftc -O -target arm64-apple-macos$(MACOS_MIN) -F $(MACOS_FW) -framework llama \
	    -Xlinker -rpath -Xlinker $(MACOS_FW) \
	    -module-cache-path $(B)/test/modules test/smoke.swift -o $(B)/test/smoke-macos
	./smoke.sh compare macos gpu $(B)/test/out $(MODEL) $(NPRED) $(B)/test/driver-apple_m1 $(B)/test/smoke-macos
	./smoke.sh compare macos cpu $(B)/test/out $(MODEL) $(NPRED) $(B)/test/driver-apple_m1 $(B)/test/smoke-macos
	@printf -- '-- negative: one token of the reference altered\n'
	mkdir -p $(B)/neg
	sed '1s/./X/' $(B)/test/out/macos-gpu-1.ref.txt > $(B)/neg/altered.txt
	@$(call expect_fail,cmp $(B)/neg/altered.txt $(B)/test/out/macos-gpu-1.test.txt,altered reference token)
	@printf -- '-- app C++ beside the framework\n'
	xcrun --sdk macosx clang++ -std=c++17 -target arm64-apple-macos$(MACOS_MIN) -fmodules -fcxx-modules \
	    -fmodules-cache-path=$(B)/test/modules -F $(MACOS_FW) -framework llama -Wl,-rpath,$(MACOS_FW) \
	    test/coexist.cpp -o $(B)/test/coexist
	$(B)/test/coexist 2>/dev/null

# The binary drop-in: test/smoke.swift linked against *upstream's*
# framework, in an app-style layout with the framework beside it under
# @rpath, run once with upstream's framework and once with ours copied over
# it -- no relink. Metal must be identical between the two (the GPU kernels
# and their host layer are upstream's own source on both sides) and ours
# must equal the port's driver. Upstream's CPU path is not ours -- its macOS
# slice compiles ggml-cpu generic, see README -- so CPU is reported only.
dropin-macos: xcframework $(B)/test/driver-apple_m1
	@printf '\n== drop-in: an app linked against upstream'"'"'s framework, run on ours ==\n'
	@test -d $(UPSTREAM) || { echo "FAIL: no $(UPSTREAM) -- run 'make upstream'"; exit 1; }
	rm -rf $(B)/dropin && mkdir -p $(B)/dropin/up/Frameworks $(B)/dropin/ours/Frameworks
	xcrun --sdk macosx swiftc -O -target arm64-apple-macos$(MACOS_MIN) -F $(UPSTREAM)/$(macos_XCF_ID) -framework llama \
	    -Xlinker -rpath -Xlinker @executable_path/Frameworks \
	    -module-cache-path $(B)/dropin/modules test/smoke.swift -o $(B)/dropin/smoke
	cp $(B)/dropin/smoke $(B)/dropin/up/smoke && cp $(B)/dropin/smoke $(B)/dropin/ours/smoke
	cp -R $(UPSTREAM)/$(macos_XCF_ID)/llama.framework $(B)/dropin/up/Frameworks/
	cp -R $(MACOS_FW)/llama.framework $(B)/dropin/ours/Frameworks/
	cmp $(B)/dropin/up/smoke $(B)/dropin/ours/smoke
	./smoke.sh compare upstream gpu $(B)/dropin/out $(MODEL) $(NPRED) $(B)/dropin/up/smoke $(B)/dropin/ours/smoke
	./smoke.sh compare port gpu $(B)/dropin/out $(MODEL) $(NPRED) $(B)/test/driver-apple_m1 $(B)/dropin/ours/smoke
	./smoke.sh compare port cpu $(B)/dropin/out $(MODEL) $(NPRED) $(B)/test/driver-apple_m1 $(B)/dropin/ours/smoke
	@printf -- '-- CPU, upstream'"'"'s framework against ours (reported, not a gate)\n'
	./smoke.sh compare upstream cpu $(B)/dropin/out $(MODEL) $(NPRED) $(B)/dropin/up/smoke $(B)/dropin/ours/smoke || true

# One simulator: $(1) label, $(2) build, $(3) device name.
define smoke_sim
	xcrun --sdk $($(2)_SDK) swiftc -O -target arm64-apple-$($(2)_CLANG_OS) -F $(XCF)/$($(2)_XCF_ID) -framework llama \
	    -Xlinker -rpath -Xlinker $(XCF)/$($(2)_XCF_ID) \
	    -module-cache-path $(B)/test/modules test/smoke.swift -o $(B)/test/smoke-$(1)
	UDID=$$(./smoke.sh sim-udid "$(3)"); \
	test -n "$$UDID" || { echo "FAIL: no available simulator named $(3)"; exit 1; }; \
	STATE=$$(xcrun simctl list devices | grep "$$UDID" | sed -E 's/.*\((Booted|Shutdown)\).*/\1/'); \
	xcrun simctl bootstatus $$UDID -b > /dev/null; \
	trap '[ "$$STATE" = Booted ] || xcrun simctl shutdown '$$UDID EXIT; \
	./smoke.sh compare $(1) cpu $(B)/test/out $(MODEL) $(NPRED) $(B)/test/driver-apple_m1 "xcrun simctl spawn $$UDID $(B)/test/smoke-$(1)"
endef

smoke-sim: xcframework $(B)/test/driver-apple_m1
	@printf '\n== smoke: iOS simulator (%s) ==\n' "$(SIM_DEVICE)"
	$(call smoke_sim,ios-sim,ios-sim,$(SIM_DEVICE))
	@printf '\n== smoke: tvOS simulator (%s) ==\n' "$(TV_DEVICE)"
	$(call smoke_sim,tvos-sim,tvos-sim,$(TV_DEVICE))
	@printf '\n== smoke: visionOS simulator (%s) ==\n' "$(XR_DEVICE)"
	$(call smoke_sim,xros-sim,visionos-sim,$(XR_DEVICE))

# On a phone: `test/device`, an iOS app running `smoke.swift`'s loop over the
# six prompts on the CPU and on Metal, staged into build/device with the model
# linked into its bundle and the framework beside it, embedded and signed by
# xcodebuild, installed and run by devicectl. The app's output comes back on
# the console and is compared with test/driver.c on the port's own apple_a7
# libraries here -- CPU and Metal both, since the Mac's Metal computes the
# same bits as the phone's (measured on an iPhone 15). The app carries the
# increased-memory-limit entitlement a 2B model needs (README, "Memory").
smoke-device: xcframework $(B)/test/driver-apple_a7
	@printf '\n== smoke: iOS device (%s) ==\n' "$(DEVICE_ID)"
	rm -rf $(B)/device && mkdir -p $(B)/device
	cp -R test/device/DeviceSmoke.xcodeproj test/device/App test/device/DeviceSmoke.entitlements $(B)/device/
	ln $(MODEL) $(B)/device/App/model.gguf 2>/dev/null || cp $(MODEL) $(B)/device/App/model.gguf
	cp -R $(XCF) $(B)/device/
	cd $(B)/device && xcrun xcodebuild build -quiet -project DeviceSmoke.xcodeproj -scheme DeviceSmoke -configuration Release \
	    -destination 'platform=iOS,id=$(DEVICE_ID)' -derivedDataPath $(B)/device-dd -allowProvisioningUpdates \
	    DEVELOPMENT_TEAM=$(TEAM) > $(B)/device-build.log 2>&1 \
	    || { grep -m5 'error:' $(B)/device-build.log; exit 1; }
	@# Embedding signs the copy, so its bytes are not ours; its UUID is.
	test "$$(xcrun dwarfdump --uuid $(B)/device-dd/Build/Products/Release-iphoneos/DeviceSmoke.app/Frameworks/llama.framework/llama | cut -d' ' -f2)" = \
	     "$$(xcrun dwarfdump --uuid $(XCF)/ios-arm64/llama.framework/llama | cut -d' ' -f2)" \
	    || { echo "FAIL: our llama.framework is not embedded in the app"; exit 1; }
	xcrun devicectl device install app --device $(DEVICE_ID) $(B)/device-dd/Build/Products/Release-iphoneos/DeviceSmoke.app > /dev/null
	@# The first launch after an install has come back with none of the
	@# app's output on the console while the app ran to completion; one more
	@# launch, if so.
	for attempt in 1 2; do \
	    xcrun devicectl device process launch --device $(DEVICE_ID) --console --terminate-existing $(DEVICE_APP) \
	        > $(B)/device-run.log 2>&1 || true; \
	    grep -q '^SMOKE-DEVICE' $(B)/device-run.log && break; \
	done
	./smoke.sh device device $(B)/test/out $(MODEL) $(B)/test/driver-apple_a7 $(B)/device-run.log
	@printf -- '-- negative: one token of the device output altered\n'
	mkdir -p $(B)/neg
	sed '1s/./X/' $(B)/test/out/device-gpu-1.test.txt > $(B)/neg/device-altered.txt
	@$(call expect_fail,cmp $(B)/test/out/device-gpu-1.ref.txt $(B)/neg/device-altered.txt,altered device token)

# Upstream's own consumer, examples/llama.swiftui, untouched: copied with its
# project, which names ../../build-apple/llama.xcframework, and that path made
# ours. It must build for the simulator, and build, install and launch on the
# phone and still be running after LAUNCH_WAIT seconds -- a framework that
# cannot load, or an API symbol the app binds that ours lacks, kills it in
# dyld before its first frame. Only the signing team and bundle id are
# overridden, on the command line.
LAUNCH_WAIT ?= 8
swiftui: xcframework
	@printf '\n== swiftui: upstream'"'"'s examples/llama.swiftui against ours ==\n'
	rm -rf $(B)/swiftui && mkdir -p $(B)/swiftui/examples $(B)/swiftui/build-apple
	git -C $(ROOT)/llama.cpp archive HEAD examples/llama.swiftui | tar -x -C $(B)/swiftui
	ln -s $(XCF) $(B)/swiftui/build-apple/llama.xcframework
	cd $(B)/swiftui/examples/llama.swiftui && xcrun xcodebuild build -quiet -project llama.swiftui.xcodeproj \
	    -scheme llama.swiftui -configuration Release -destination 'generic/platform=iOS Simulator' \
	    -derivedDataPath $(B)/swiftui-dd CODE_SIGNING_ALLOWED=NO > $(B)/swiftui-sim.log 2>&1 \
	    || { grep -m5 'error:' $(B)/swiftui-sim.log; exit 1; }
	@echo "  built for the iOS simulator"
	cd $(B)/swiftui/examples/llama.swiftui && xcrun xcodebuild build -quiet -project llama.swiftui.xcodeproj \
	    -scheme llama.swiftui -configuration Release -destination 'platform=iOS,id=$(DEVICE_ID)' \
	    -derivedDataPath $(B)/swiftui-dd -allowProvisioningUpdates \
	    DEVELOPMENT_TEAM=$(TEAM) PRODUCT_BUNDLE_IDENTIFIER=$(SWIFTUI_APP) > $(B)/swiftui-device.log 2>&1 \
	    || { grep -m5 'error:' $(B)/swiftui-device.log; exit 1; }
	APP=$(B)/swiftui-dd/Build/Products/Release-iphoneos/llama.swiftui.app; \
	test "$$(xcrun dwarfdump --uuid $$APP/Frameworks/llama.framework/llama | cut -d' ' -f2)" = \
	     "$$(xcrun dwarfdump --uuid $(XCF)/ios-arm64/llama.framework/llama | cut -d' ' -f2)" \
	    || { echo "FAIL: the app did not embed our framework's binary"; exit 1; }; \
	xcrun devicectl device install app --device $(DEVICE_ID) $$APP > /dev/null
	xcrun devicectl device process launch --device $(DEVICE_ID) --terminate-existing $(SWIFTUI_APP) > $(B)/swiftui-launch.log 2>&1
	sleep $(LAUNCH_WAIT)
	xcrun devicectl device info processes --device $(DEVICE_ID) > $(B)/swiftui-procs.txt 2>&1
	grep -q 'llama.swiftui.app/llama.swiftui' $(B)/swiftui-procs.txt \
	    || { echo "FAIL: llama.swiftui is not running $(LAUNCH_WAIT) s after launch"; exit 1; }
	@echo "  built, installed and running on $(DEVICE_ID) after $(LAUNCH_WAIT) s"

# -----------------------------------------------------------------------------
# Distribution

pkg-test: xcframework
	@printf '\n== package: macOS ==\n'
	swift build --package-path test/pkg --scratch-path $(B)/pkg/swiftpm -c release
	$(B)/pkg/swiftpm/release/pkgtest-run
	@printf '\n== package: iOS simulator ==\n'
	cd test/pkg && xcrun xcodebuild build -quiet -scheme PkgTest -destination 'generic/platform=iOS Simulator' \
	    -derivedDataPath $(B)/pkg/dd
	@echo "  built PkgTest for the iOS simulator"
	@printf -- '-- negative: the package before the xcframework is built\n'
	rm -rf $(B)/neg/pkg && mkdir -p $(B)/neg/pkg/xcframework/test/pkg
	cp Package.swift $(B)/neg/pkg/xcframework/
	cp -R test/pkg/Package.swift test/pkg/Sources $(B)/neg/pkg/xcframework/test/pkg/
	@$(call expect_fail,swift build --package-path $(B)/neg/pkg/xcframework/test/pkg --scratch-path $(B)/neg/pkg/swiftpm,package resolved with no xcframework built)
	grep -q 'llama.xcframework' $(B)/neg/last.log || { echo "FAIL: the error does not name the xcframework"; exit 1; }

zip: xcframework
	@printf '\n== zip ==\n'
	rm -f $(XCF).zip
	cd $(B) && ditto -c -k --keepParent llama.xcframework llama.xcframework.zip
	swift package compute-checksum $(XCF).zip | tee $(XCF).zip.sha256
	@ls -la $(XCF).zip

# -----------------------------------------------------------------------------

port-gates:
	@printf '\n== port gates: make validate ops-diff ==\n'
	$(MAKE) -C $(ROOT) validate ops-diff

check: verify negative smoke-macos smoke-sim pkg-test zip port-gates
	@if [ -d $(UPSTREAM) ]; then $(MAKE) --no-print-directory compat dropin-macos; \
	else echo "  (no build/upstream: compat and dropin-macos skipped -- 'make upstream' builds it)"; fi
	@printf '\n== check: all passed ==\n'

clean:
	rm -rf $(B) .build .swiftpm test/pkg/.build test/pkg/.swiftpm
