# llamazig.xcframework: the port as a static framework for macOS and iOS apps.
# See README.md for consumers, the decisions behind it, and the gates.
#
#   make              build/llamazig.xcframework
#   make verify       the structural gates, on every slice and the xcframework
#   make negative     every gate fed a broken input, each required to fail
#   make smoke-macos  Swift through the framework vs the port's own libraries,
#                     six prompts, GPU and CPU; plus app C++ beside it
#   make smoke-sim    the same on the iOS simulator, CPU; one Metal run reported
#   make smoke-device the same on a physical iPhone (DEVICE_ID), CPU and Metal,
#                     signed for TEAM; not part of `check`, which needs no phone
#   make pkg-test     Package.swift consumed by a package, macOS and simulator
#   make zip          build/llamazig.xcframework.zip and its SwiftPM checksum
#   make port-gates   the root `make validate ops-diff`, for the port changes
#   make check        all of the above, in that order
#
# Each slice is `zig build reference` for its target, then `ld -r` of libggml,
# libllama and Zig's own compiler_rt into one object that exports only
# llama_/ggml_/gguf_ -- so the CPU path computes with the same math the gated
# port does. No C++ runtime goes in: since libllama swapped, neither library
# references one, and prelinking libc++ and libc++abi beside them produced a
# byte-identical object.

SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c

ROOT       := $(abspath $(CURDIR)/..)
B          := $(CURDIR)/build
NAME       := llamazig
XCF        := $(B)/$(NAME).xcframework
ZIG        ?= zig

IOS_MIN    ?= 18.0
MACOS_MIN  ?= 26.0
# iOS: A14 and later. Its feature set is the M1's -- the arms every gate
# runs at the baseline; iOS 18 also runs on A12/A13, which this excludes.
# macOS: M3 and later. apple_m3 adds i8mm, so the CPU path takes the
# two-row `smmla` dot products and the i8mm gemms (src/ggml/cpu/quants/arm/
# mmla.zig, src/ggml/cpu/repack/arm/mmla.zig) and drops llamafile's sgemm,
# as upstream does; those arms are gated with `make -C .. ops-diff
# repack-diff node-diff GATE_CPU=apple_m3`. An M1 cannot run this slice:
# i8mm is an illegal instruction there. An M2 has i8mm and Zig's apple_m3
# adds only `hcx` over apple_m2, so it should run there, but is untested.
IOS_CPU    ?= apple_a14
MACOS_CPU  ?= apple_m3
BUNDLE_ID  ?= ai.cloneable.llamazig
VERSION    := $(shell sed -n 's/^ *\.version = "\(.*\)",/\1/p' $(ROOT)/build.zig.zon)

MODEL      ?= $(ROOT)/Qwen3.5-2B-Q4_K_M.gguf
NPRED      ?= 32
SIM_DEVICE ?= iPhone 17 Pro
# smoke-device: the phone (a UDID from `xcrun devicectl list devices`) and the
# development team its app is signed for. The phone must be registered in
# that team; `-allowProvisioningDeviceRegistration` would register it.
DEVICE_ID  ?= 00008120-00112894262A601E
TEAM       ?= Q6H2FB9YW2
DEVICE_APP := com.mlavergn.llamazig.devicesmoke

SLICES := macos ios ios-sim

HEADERS := $(ROOT)/llama.cpp/include/llama.h \
           $(addprefix $(ROOT)/llama.cpp/ggml/include/,ggml.h ggml-opt.h ggml-alloc.h ggml-backend.h ggml-metal.h ggml-cpu.h gguf.h)

# Per slice: Zig target and CPU, SDK, `ld -platform_version` name, Mach-O
# platform number, clang target, Info.plist platform and minimum-OS key,
# xcframework library identifier, and the Info.plist's place in the bundle.
macos_TARGET     := aarch64-macos.$(MACOS_MIN)
macos_CPU        := $(MACOS_CPU)
macos_MIN        := $(MACOS_MIN)
macos_SDK        := macosx
macos_LD         := macos
macos_PLAT       := 1
macos_CLANG      := arm64-apple-macos$(MACOS_MIN)
macos_PLIST_PLAT := MacOSX
macos_MIN_KEY    := LSMinimumSystemVersion
macos_XCF_ID     := macos-arm64
macos_PLIST      := Versions/A/Resources/Info.plist

ios_TARGET       := aarch64-ios.$(IOS_MIN)
ios_CPU          := $(IOS_CPU)
ios_MIN          := $(IOS_MIN)
ios_SDK          := iphoneos
ios_LD           := ios
ios_PLAT         := 2
ios_CLANG        := arm64-apple-ios$(IOS_MIN)
ios_PLIST_PLAT   := iPhoneOS
ios_MIN_KEY      := MinimumOSVersion
ios_XCF_ID       := ios-arm64
ios_PLIST        := Info.plist

ios-sim_TARGET     := aarch64-ios.$(IOS_MIN)-simulator
ios-sim_CPU        := $(IOS_CPU)
ios-sim_MIN        := $(IOS_MIN)
ios-sim_SDK        := iphonesimulator
ios-sim_LD         := ios-simulator
ios-sim_PLAT       := 7
ios-sim_CLANG      := arm64-apple-ios$(IOS_MIN)-simulator
ios-sim_PLIST_PLAT := iPhoneSimulator
ios-sim_MIN_KEY    := MinimumOSVersion
ios-sim_XCF_ID     := ios-arm64-simulator
ios-sim_PLIST      := Info.plist

# Runs a command that must fail, and fails if it does not.
# $(1): the command; $(2): what the injected fault is.
expect_fail = if ( $(1) ) > $(B)/neg/last.log 2>&1; then \
        echo "NEGATIVE TEST ESCAPED: $(2)"; cat $(B)/neg/last.log; exit 1; \
    else echo "  caught: $(2) -- $$(grep -m1 -E 'FAIL|error|differ' $(B)/neg/last.log)"; fi

.PHONY: all xcframework $(SLICES) verify $(addprefix verify-,$(SLICES)) negative \
        ref-driver ref-driver-macos ref-driver-sim smoke-macos smoke-sim smoke-device pkg-test zip port-gates check clean

all: xcframework

# -----------------------------------------------------------------------------
# Slices

$(SLICES): %:
	@printf '\n== %s: zig build reference -Dtarget=%s -Dcpu=%s ==\n' $* $($*_TARGET) $($*_CPU)
	cd $(ROOT) && $(ZIG) build reference -Dtarget=$($*_TARGET) -Dcpu=$($*_CPU) --release=fast --prefix $(B)/$*/zig
	@printf '\n== %s: Zig compiler_rt for %s ==\n' $* $($*_TARGET)
	@# A probe link with --verbose-link names the libcompiler_rt.a Zig built
	@# for this target; the build API does not expose it. It is not
	@# optional: a `zig cc` link -- every
	@# bit-exact gate, and `make port` -- takes expf, logf, sinf, cosf, exp2f
	@# from it rather than libSystem, and with Apple's libm instead 3 of 6 CPU
	@# prompts diverged from the port. A fresh cache dir forces the link to
	@# run, so the line is always printed.
	@#
	@# Zig 0.17's linker reads the SDK version from SDKSettings.json's
	@# MinimalDisplayName, which the simulator SDK sets to "Simulator - 27.0"
	@# and the link fails. The probe links against a shim: the SDK's entries
	@# symlinked, and SDKSettings.json with MinimalDisplayName set to Version.
	rm -rf $(B)/$*/rt && mkdir -p $(B)/$*/rt/sdk
	SDK=$$(xcrun --sdk $($*_SDK) --show-sdk-path); \
	for e in "$$SDK"/*; do [ "$${e##*/}" = SDKSettings.json ] || ln -s "$$e" $(B)/$*/rt/sdk/; done; \
	cp "$$SDK/SDKSettings.json" $(B)/$*/rt/sdk/; \
	plutil -replace MinimalDisplayName -string "$$(plutil -extract Version raw "$$SDK/SDKSettings.json")" -- $(B)/$*/rt/sdk/SDKSettings.json; \
	plutil -convert json $(B)/$*/rt/sdk/SDKSettings.json; \
	$(ZIG) libc -target $($*_TARGET) | sed "s#$$SDK#$(B)/$*/rt/sdk#" > $(B)/$*/rt/libc.txt
	printf 'int main(void) { return 0; }\n' > $(B)/$*/rt/probe.c
	cd $(B)/$*/rt && $(ZIG) build-exe probe.c -lc -target $($*_TARGET) -mcpu $($*_CPU) -OReleaseFast \
	    --libc libc.txt --cache-dir cache --verbose-link 2>&1 | tr ' ' '\n' | grep -E '/libcompiler_rt\.a$$' > paths.txt
	test "$$(wc -l < $(B)/$*/rt/paths.txt)" -eq 1 || { echo "FAIL: probe found: $$(cat $(B)/$*/rt/paths.txt)"; exit 1; }
	cp $$(cat $(B)/$*/rt/paths.txt) $(B)/$*/rt/
	@printf '\n== %s: prelink ==\n' $*
	xcrun ld -r -arch arm64 -platform_version $($*_LD) $($*_MIN) $$(xcrun --sdk $($*_SDK) --show-sdk-version) \
	    -exported_symbols_list exports.txt \
	    -force_load $(B)/$*/zig/lib/libggml.a -force_load $(B)/$*/zig/lib/libllama.a \
	    $(B)/$*/rt/libcompiler_rt.a -o $(B)/$*/$(NAME).o
	xcrun strip -S $(B)/$*/$(NAME).o
	rm -f $(B)/$*/$(NAME).a && xcrun libtool -static -o $(B)/$*/$(NAME).a $(B)/$*/$(NAME).o
	@printf '\n== %s: framework ==\n' $*
	rm -rf $(B)/$*/fw
	F=$(B)/$*/fw/$(NAME).framework; \
	if [ $* = macos ]; then \
	    D=$$F/Versions/A; mkdir -p $$D/Headers $$D/Modules $$D/Resources; \
	    ln -s A $$F/Versions/Current; \
	    for l in $(NAME) Headers Modules Resources; do ln -s Versions/Current/$$l $$F/$$l; done; \
	else \
	    D=$$F; mkdir -p $$D/Headers $$D/Modules; \
	fi; \
	cp $(B)/$*/$(NAME).a $$D/$(NAME); \
	cp $(HEADERS) $$D/Headers/; \
	cp framework.modulemap $$D/Modules/module.modulemap; \
	sed -e 's/@NAME@/$(NAME)/g' -e 's/@BUNDLE_ID@/$(BUNDLE_ID)/g' -e 's/@VERSION@/$(VERSION)/g' \
	    -e 's/@PLATFORM@/$($*_PLIST_PLAT)/g' -e 's/@MIN_KEY@/$($*_MIN_KEY)/g' -e 's/@MIN_OS@/$($*_MIN)/g' \
	    Info.plist.in > $$F/$($*_PLIST)

xcframework: $(SLICES)
	@printf '\n== xcframework ==\n'
	rm -rf $(XCF)
	xcodebuild -create-xcframework $(foreach s,$(SLICES),-framework $(B)/$(s)/fw/$(NAME).framework) -output $(XCF)

# -----------------------------------------------------------------------------
# Gates

verify: xcframework
	@printf '\n== verify ==\n'
	@$(MAKE) --no-print-directory $(addprefix verify-,$(SLICES))
	@printf -- '-- xcframework\n'
	./verify.sh xcframework $(XCF)

$(addprefix verify-,$(SLICES)): verify-%:
	@printf -- '-- %s\n' $*
	./verify.sh platform $($*_PLAT) $($*_MIN) $(B)/$*/zig/lib/libggml.a $(B)/$*/zig/lib/libllama.a \
	    $(B)/$*/rt/libcompiler_rt.a $(B)/$*/$(NAME).o
	./verify.sh symbols $(B)/$*/$(NAME).o
	./verify.sh plist $(XCF)/$($*_XCF_ID)/$(NAME).framework/$($*_PLIST)
	./verify.sh link $($*_SDK) $($*_CLANG) $(XCF)/$($*_XCF_ID) $(B)/test/stub-$*

# Every structural gate, fed an input with one fault injected.
negative: xcframework
	@printf '\n== negative tests ==\n'
	rm -rf $(B)/neg && mkdir -p $(B)/neg
	@# Fault: wrong platform -- the simulator object checked as a device one.
	@$(call expect_fail,./verify.sh platform $(ios_PLAT) $(ios_MIN) $(B)/ios-sim/$(NAME).o,simulator object passed as iOS device)
	@# Fault: wrong minimum OS.
	@$(call expect_fail,./verify.sh platform $(macos_PLAT) 15.0 $(B)/macos/$(NAME).o,macOS $(MACOS_MIN) object passed as minos 15.0)
	@# Fault: no exports list, so every internal symbol stays global.
	xcrun ld -r -arch arm64 -platform_version macos $(MACOS_MIN) $$(xcrun --sdk macosx --show-sdk-version) \
	    -force_load $(B)/macos/zig/lib/libggml.a -force_load $(B)/macos/zig/lib/libllama.a \
	    $(B)/macos/rt/libcompiler_rt.a -o $(B)/neg/noexports.o
	@$(call expect_fail,./verify.sh symbols $(B)/neg/noexports.o,prelink without -exported_symbols_list)
	@# Fault: -keep_private_externs, so nothing is localized.
	xcrun ld -r -arch arm64 -platform_version macos $(MACOS_MIN) $$(xcrun --sdk macosx --show-sdk-version) \
	    -keep_private_externs -exported_symbols_list exports.txt \
	    -force_load $(B)/macos/zig/lib/libggml.a -force_load $(B)/macos/zig/lib/libllama.a \
	    $(B)/macos/rt/libcompiler_rt.a -o $(B)/neg/private.o
	@$(call expect_fail,./verify.sh symbols $(B)/neg/private.o,prelink with -keep_private_externs)
	@# Fault: compiler_rt left out, so expf and friends come from Apple's libm.
	xcrun ld -r -arch arm64 -platform_version macos $(MACOS_MIN) $$(xcrun --sdk macosx --show-sdk-version) \
	    -exported_symbols_list exports.txt \
	    -force_load $(B)/macos/zig/lib/libggml.a -force_load $(B)/macos/zig/lib/libllama.a \
	    -o $(B)/neg/nocrt.o
	@$(call expect_fail,./verify.sh symbols $(B)/neg/nocrt.o,prelink without compiler_rt)
	@# Fault: the module map's autolinking removed.
	mkdir -p $(B)/neg/nolink && cp -R $(B)/macos/fw/$(NAME).framework $(B)/neg/nolink/
	rm $(B)/neg/nolink/$(NAME).framework/Versions/A/Modules/module.modulemap
	sed '/link framework/d' framework.modulemap > $(B)/neg/nolink/$(NAME).framework/Versions/A/Modules/module.modulemap
	@$(call expect_fail,./verify.sh link macosx $(macos_CLANG) $(B)/neg/nolink $(B)/neg/stub-nolink,module map without link framework)
	@# Fault: a malformed Info.plist.
	sed 's#</dict>##' $(B)/macos/fw/$(NAME).framework/$(macos_PLIST) > $(B)/neg/Info.plist
	@$(call expect_fail,./verify.sh plist $(B)/neg/Info.plist,Info.plist missing </dict>)
	@# Fault: an xcframework with a slice missing.
	xcodebuild -create-xcframework -framework $(B)/macos/fw/$(NAME).framework -framework $(B)/ios/fw/$(NAME).framework \
	    -output $(B)/neg/two.xcframework > /dev/null
	@$(call expect_fail,./verify.sh xcframework $(B)/neg/two.xcframework,xcframework without the simulator slice)
	@echo "  PASS: every gate failed on its injected fault"

# -----------------------------------------------------------------------------
# Smoke tests

# The reference: test/driver.c against the port's own libraries, exactly as
# scripts/parity-port links its port side -- one per slice CPU, because the
# CPU comparison is only exact when both sides compiled the same kernel arms.
# The macOS slice is apple_m3 (i8mm: the two-row dot products and the i8mm
# gemms); the simulator slice is apple_a14 (neither). Each is installed to a
# prefix of its own rather than zig-out/, which the root gates rebuild for
# whatever CPU they are run at.
#   ref-driver-macos  -> $(B)/test/driver-macos, MACOS_CPU
#   ref-driver-sim    -> $(B)/test/driver-sim,   IOS_CPU
ref-driver: ref-driver-macos ref-driver-sim

ref-driver-macos: DRIVER_CPU = $(MACOS_CPU)
ref-driver-sim:   DRIVER_CPU = $(IOS_CPU)
ref-driver-macos ref-driver-sim: ref-driver-%:
	@printf '\n== reference driver for %s (zig build reference -Dcpu=%s) ==\n' $* $(DRIVER_CPU)
	cd $(ROOT) && $(ZIG) build reference --release=fast -Dcpu=$(DRIVER_CPU) --prefix $(B)/ref-$(DRIVER_CPU)
	mkdir -p $(B)/test
	cd $(ROOT) && $(ZIG) cc -mcpu=$(DRIVER_CPU) $(CURDIR)/test/driver.c -o $(B)/test/driver-$* -std=c11 \
	    -I llama.cpp/include -I llama.cpp/ggml/include $(B)/ref-$(DRIVER_CPU)/lib/libllama.a $(B)/ref-$(DRIVER_CPU)/lib/libggml.a \
	    -lc++ -framework Foundation -framework Metal -framework MetalKit -framework Accelerate \
	    -F "$$(xcrun --sdk macosx --show-sdk-path)/System/Library/Frameworks"

smoke-macos: xcframework ref-driver-macos
	@printf '\n== smoke: macOS ==\n'
	xcrun --sdk macosx swiftc -O -target $(macos_CLANG) -F $(XCF)/$(macos_XCF_ID) \
	    -module-cache-path $(B)/test/modules test/smoke.swift -o $(B)/test/smoke-macos
	! otool -L $(B)/test/smoke-macos | grep 'libc++'
	./smoke.sh compare macos gpu $(B)/test/out $(MODEL) $(NPRED) $(B)/test/driver-macos $(B)/test/smoke-macos
	./smoke.sh compare macos cpu $(B)/test/out $(MODEL) $(NPRED) $(B)/test/driver-macos $(B)/test/smoke-macos
	@printf -- '-- negative: one token of the reference altered\n'
	mkdir -p $(B)/neg
	sed '1s/./X/' $(B)/test/out/macos-gpu-1.ref.txt > $(B)/neg/altered.txt
	@$(call expect_fail,cmp $(B)/neg/altered.txt $(B)/test/out/macos-gpu-1.test.txt,altered reference token)
	@printf -- '-- app C++ beside the framework\n'
	xcrun --sdk macosx clang++ -std=c++17 -target $(macos_CLANG) -fmodules -fcxx-modules \
	    -fmodules-cache-path=$(B)/test/modules -F $(XCF)/$(macos_XCF_ID) -framework $(NAME) \
	    test/coexist.cpp -o $(B)/test/coexist
	$(B)/test/coexist 2>/dev/null

smoke-sim: xcframework ref-driver-sim
	@printf '\n== smoke: iOS simulator (%s) ==\n' "$(SIM_DEVICE)"
	xcrun --sdk iphonesimulator swiftc -O -target $(ios-sim_CLANG) -F $(XCF)/$(ios-sim_XCF_ID) \
	    -module-cache-path $(B)/test/modules test/smoke.swift -o $(B)/test/smoke-sim
	UDID=$$(./smoke.sh sim-udid "$(SIM_DEVICE)"); \
	test -n "$$UDID" || { echo "FAIL: no available simulator named $(SIM_DEVICE)"; exit 1; }; \
	STATE=$$(xcrun simctl list devices | grep "$$UDID" | sed -E 's/.*\((Booted|Shutdown)\).*/\1/'); \
	xcrun simctl bootstatus $$UDID -b > /dev/null; \
	trap '[ "$$STATE" = Booted ] || xcrun simctl shutdown '$$UDID EXIT; \
	./smoke.sh compare sim cpu $(B)/test/out $(MODEL) $(NPRED) $(B)/test/driver-sim "xcrun simctl spawn $$UDID $(B)/test/smoke-sim"; \
	printf -- '-- Metal on the simulator (reported, not a gate)\n'; \
	if xcrun simctl spawn $$UDID $(B)/test/smoke-sim $(MODEL) "The capital of France is" $(NPRED) gpu \
	        > $(B)/test/out/sim-metal.txt 2> $(B)/test/out/sim-metal.log; then \
	    echo "  ran: $$(head -c 120 $(B)/test/out/sim-metal.txt | tr '\n' ' ')"; \
	    if cmp -s $(B)/test/out/sim-metal.txt $(B)/test/out/macos-gpu-1.ref.txt; then echo "  identical to macOS Metal"; \
	    else echo "  differs from macOS Metal"; fi; \
	else echo "  exited $$? -- $$(grep -m3 -iE 'error|fail|abort' $(B)/test/out/sim-metal.log | tr '\n' ' ')"; fi

# On a phone: `test/device`, an iOS app running `smoke.swift`'s loop over the
# six prompts on the CPU and on Metal, staged into build/device with the model
# linked into its bundle and the framework beside it, built and signed by
# xcodebuild, installed and run by devicectl. The app's output comes back on
# the console and is compared with test/driver.c on the port's own apple_a14
# libraries here -- CPU and Metal both, since the Mac's Metal computes the
# same bits as the phone's (measured on an iPhone 15). The app carries the
# increased-memory-limit entitlement a 2B model needs (README, "Memory").
smoke-device: xcframework ref-driver-sim
	@printf '\n== smoke: iOS device (%s) ==\n' "$(DEVICE_ID)"
	rm -rf $(B)/device && mkdir -p $(B)/device
	cp -R test/device/DeviceSmoke.xcodeproj test/device/App test/device/DeviceSmoke.entitlements $(B)/device/
	ln $(MODEL) $(B)/device/App/model.gguf 2>/dev/null || cp $(MODEL) $(B)/device/App/model.gguf
	cp -R $(XCF) $(B)/device/
	cd $(B)/device && xcodebuild build -quiet -project DeviceSmoke.xcodeproj -scheme DeviceSmoke -configuration Release \
	    -destination 'platform=iOS,id=$(DEVICE_ID)' -derivedDataPath $(B)/device-dd -allowProvisioningUpdates \
	    DEVELOPMENT_TEAM=$(TEAM) > $(B)/device-build.log 2>&1 \
	    || { grep -m5 'error:' $(B)/device-build.log; exit 1; }
	xcrun devicectl device install app --device $(DEVICE_ID) $(B)/device-dd/Build/Products/Release-iphoneos/DeviceSmoke.app > /dev/null
	@# The first launch after an install has come back with none of the
	@# app's output on the console while the app ran to completion; one more
	@# launch, if so.
	for attempt in 1 2; do \
	    xcrun devicectl device process launch --device $(DEVICE_ID) --console --terminate-existing $(DEVICE_APP) \
	        > $(B)/device-run.log 2>&1 || true; \
	    grep -q '^SMOKE-DEVICE' $(B)/device-run.log && break; \
	done
	./smoke.sh device device $(B)/test/out $(MODEL) $(B)/test/driver-sim $(B)/device-run.log
	@printf -- '-- negative: one token of the device output altered\n'
	mkdir -p $(B)/neg
	sed '1s/./X/' $(B)/test/out/device-gpu-1.test.txt > $(B)/neg/device-altered.txt
	@$(call expect_fail,cmp $(B)/test/out/device-gpu-1.ref.txt $(B)/neg/device-altered.txt,altered device token)

# -----------------------------------------------------------------------------
# Distribution

pkg-test: xcframework
	@printf '\n== package: macOS ==\n'
	swift build --package-path test/pkg --scratch-path $(B)/pkg/swiftpm -c release
	$(B)/pkg/swiftpm/release/pkgtest-run
	@printf '\n== package: iOS simulator ==\n'
	cd test/pkg && xcodebuild build -quiet -scheme PkgTest -destination 'generic/platform=iOS Simulator' \
	    -derivedDataPath $(B)/pkg/dd
	@echo "  built PkgTest for the iOS simulator"
	@printf -- '-- negative: the package before the xcframework is built\n'
	rm -rf $(B)/neg/pkg && mkdir -p $(B)/neg/pkg/xcfwk/test/pkg
	cp Package.swift $(B)/neg/pkg/xcfwk/
	cp -R test/pkg/Package.swift test/pkg/Sources $(B)/neg/pkg/xcfwk/test/pkg/
	@$(call expect_fail,swift build --package-path $(B)/neg/pkg/xcfwk/test/pkg --scratch-path $(B)/neg/pkg/swiftpm,package resolved with no xcframework built)
	grep -q 'llamazig.xcframework' $(B)/neg/last.log || { echo "FAIL: the error does not name the xcframework"; exit 1; }

zip: xcframework
	@printf '\n== zip ==\n'
	rm -f $(XCF).zip
	cd $(B) && ditto -c -k --keepParent $(NAME).xcframework $(NAME).xcframework.zip
	swift package compute-checksum $(XCF).zip | tee $(XCF).zip.sha256
	@ls -la $(XCF).zip

# -----------------------------------------------------------------------------

port-gates:
	@printf '\n== port gates: make validate ops-diff ==\n'
	$(MAKE) -C $(ROOT) validate ops-diff

check: verify negative smoke-macos smoke-sim pkg-test zip port-gates
	@printf '\n== check: all passed ==\n'

clean:
	rm -rf $(B) .build .swiftpm test/pkg/.build test/pkg/.swiftpm
