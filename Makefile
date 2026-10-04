###############################################
#
# Makefile
#
###############################################

.DEFAULT_GOAL := build

.PHONY: build cli llama.cpp probe graph-diff sched-diff ops-diff repack-diff backend-ops parity-cli parity-port parity-port-cpu node-diff port ref ref-chat validate

CMAKE ?= $(firstword $(wildcard $(CURDIR)/.tools/cmake-*/CMake.app/Contents/bin/cmake) cmake)

MACOS_FLAGS = -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_OSX_SYSROOT=macosx -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_ACCELERATE=ON -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF

ZIG_CC  = $(CURDIR)/scripts/zigcc
ZIG_CXX = $(CURDIR)/scripts/zigcxx

MODEL   ?= Qwen3.5-2B-Q4_K_M.gguf
PROMPT  ?= The capital of France is
NPRED   ?= 512
TEMP    ?= 0.8
SEED    ?= 42
SCRATCH ?= /tmp

PARITY_NPRED ?= 32

PORT_CLI := ./zig-out/bin/llama-cli
REF_CLI  := llama.cpp.zmake/zig-out/macos/bin/llama-cli

#
# Setup
#

# Download CMake into .tools/.
cmake:
	mkdir -p .tools
	cd .tools; curl -sSL -o cmake.tar.gz https://github.com/Kitware/CMake/releases/download/v4.4.3/cmake-4.4.3-macos-universal.tar.gz
	cd .tools; tar xzf cmake.tar.gz && rm cmake.tar.gz
	$(CMAKE) --version

# Clone the pinned llama.cpp reference checkout.
clone:
	git clone https://github.com/ggerganov/llama.cpp.git
	cd llama.cpp; git checkout v0.3.0

# Delete all build output.
clean:
	rm -rf cmake-build zig-out .zig-cache

#
# Reference builds
#

# Build llama.cpp for iOS with CMake.
buildios:
	-mkdir cmake-build
	cd cmake-build; cmake ../llama.cpp -G "Unix Makefiles" -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=18.0 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_ACCELERATE=ON -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
	cd cmake-build; make -j

# Build llama.cpp for macOS with Apple clang; the control build.
buildmacos:
	-mkdir -p cmake-build/apple
	cd cmake-build/apple; $(CMAKE) $(CURDIR)/llama.cpp -G "Unix Makefiles" $(MACOS_FLAGS)
	cd cmake-build/apple; make -j

# Build the same sources with zig cc.
buildmacos-zig:
	-mkdir -p cmake-build/zig
	cd cmake-build/zig; $(CMAKE) $(CURDIR)/llama.cpp -G "Unix Makefiles" $(MACOS_FLAGS) -DCMAKE_C_COMPILER=$(ZIG_CC) -DCMAKE_CXX_COMPILER=$(ZIG_CXX) -DCMAKE_ASM_COMPILER=$(ZIG_CC)
	cd cmake-build/zig; make -j

# Build the llama.cpp reference libraries with the Zig toolchain.
reference:
	zig build reference

# Build upstream's llama-cli through llama.cpp.zmake.
zmake:
	cd llama.cpp.zmake; $(MAKE) cli

#
# Build
#

# Build the library and CLI.
build:
	zig build

# Build the CLI.
buildcli:
	zig build cli

# Build optimized for release.
dist:
	zig build --release=fast

#
# Checks
#

# Run every check that needs no model or reference build.
validate:
	@printf '\n== formatting ==\n'
	zig fmt --check build.zig build/*.zig harness/*.zig src/*.zig src/ggml/*.zig src/ggml/quants/*.zig src/ggml/cpu/*.zig src/ggml/cpu/ops/*.zig src/ggml/cpu/quants/*.zig src/ggml/cpu/quants/arm/*.zig src/ggml/cpu/repack/*.zig src/ggml/cpu/repack/arm/*.zig cli/*.zig
	@printf '\n== scaffold ==\n'
	zig build
	@printf '\n== unit tests ==\n'
	zig build test --summary all
	@printf '\n== ported ggml (debug) ==\n'
	zig build test-port --summary all
	@printf '\n== ported ggml (release) ==\n'
	zig build test-port --release=fast --summary all
	@printf '\n== ported symbol coverage ==\n'
	./scripts/port-coverage
	@printf '\n== upstream citations ==\n'
	./scripts/port-links
	@printf '\n== scheduler diff vs reference ==\n'
	@./scripts/sched-diff || { [ -f cmake-build/apple/ggml/src/libggml.a ] || echo "  skipped: no reference build (run 'make buildmacos')"; }
	@printf '\n== graph diff vs reference ==\n'
	@if [ -f cmake-build/apple/ggml/src/libggml.a ]; then zig build ported-lib --release=fast && ./scripts/graph-diff; else echo "  skipped: no reference build (run 'make buildmacos')"; fi
	@printf '\nvalidate: OK\n'

# Diff the token streams of the Apple-clang and zig cc builds.
parity:
	./scripts/parity $(MODEL) "$(PROMPT)"

# Prove ported code is on the execution path.
probe:
	./scripts/probe-ported $(MODEL)

# Check every upstream citation still points at the C it names.
port-links:
	./scripts/port-links

# Diff the graphs the ported constructors build against the reference ones.
graph-diff:
	zig build ported-lib --release=fast
	./scripts/graph-diff

# Diff the scheduler's backend assignments against the reference ones.
# Nothing else covers them: see the header of scripts/sched-diff.
sched-diff:
	zig build reference
	./scripts/sched-diff

# Diff the CPU op kernels' computed output against the reference C, on bits.
# The only value-level oracle for ggml-cpu -- see the header of scripts/ops-diff.
#
# Both sides must be --release=fast: the reference is the stock C built by the
# same Zig toolchain, and an optimization difference moves the last bit. The
# reference lives in llama.cpp.zmake, which is a separate repository.
ops-diff:
	cd llama.cpp.zmake && zig build lib --release=fast
	zig build reference --release=fast
	./scripts/ops-diff

# Diff every interleaved repack kernel against the reference C++, on bits.
# The only oracle those 36 kernels have: test-backend-ops never allocates a
# CPU_REPACK buffer, ops-diff's tensors live in a plain CPU buffer, and
# `make port` runs on Metal. node-diff reaches them but stops at the first
# divergent node. See the header of scripts/repack-diff.
repack-diff:
	zig build reference --release=fast
	./scripts/repack-diff

# Diff our CLI's output against the C reference, end to end.
parity-cli: buildcli
	./scripts/parity-cli $(MODEL) $(PARITY_NPRED)

# Run upstream's test-backend-ops against the ported ggml.
backend-ops:
	zig build reference --release=fast
	./scripts/backend-ops $(ARGS)

# Diff inference output between the ported and reference libraries.
# Both sides --release=fast, against llama.cpp.zmake: see scripts/parity-port.
parity-port:
	cd llama.cpp.zmake && zig build lib --release=fast
	zig build reference --release=fast
	./scripts/parity-port $(MODEL)

# Every graph node of a real model's decode, ported vs reference, on bits. CPU
# by default; ARGS=--gpu for Metal. See the header of scripts/node-diff.
node-diff:
	cd llama.cpp.zmake && zig build lib --release=fast
	zig build reference --release=fast
	./scripts/node-diff $(ARGS) $(MODEL)

# The same, with the model on the CPU device alone, so inference runs through
# the ported ggml-cpu kernels -- which on a Metal machine it otherwise never
# does.
parity-port-cpu:
	cd llama.cpp.zmake && zig build lib --release=fast
	zig build reference --release=fast
	./scripts/parity-port --cpu $(MODEL)

#
# Run
#

# Load a model through the reference build and generate.
smoke:
	zig build smoke -- "$(MODEL)" "$(PROMPT)"

# Launch our CLI in interactive conversation mode.
# No -n: a reply runs to end-of-generation, as upstream's conversation mode
# does. NPRED is for the port/ref pair, where 512 tokens keeps a diff short;
# here it cut replies off mid-answer. ARGS="-n N" still caps it.
cli: buildcli
	@$(PORT_CLI) -m "$(MODEL)" -cnv --temp $(TEMP) -s $(SEED) $(ARGS)

# Run our CLI on a raw completion.
port: buildcli
	@printf '\n$$ %s -m %s -p %s -n %s --temp %s -s %s %s\n' "$(PORT_CLI)" "$(MODEL)" "'$(PROMPT)'" "$(NPRED)" "$(TEMP)" "$(SEED)" "$(ARGS)"
	@$(PORT_CLI) -m "$(MODEL)" -p "$(PROMPT)" -n $(NPRED) --temp $(TEMP) -s $(SEED) -st $(ARGS)
	@printf '\n'

# Run the same completion loop linked against the stock C libraries.
#
# Linked with Apple's clang, not `zig cc`. These objects come from the
# Apple-clang CMake build, and since Xcode 27 Zig's linker cannot consume them
# -- "failed to resolve relocations and write atoms: Overflow" on
# llama-model.cpp.o and friends. Apple's own linker takes them without
# complaint, and this target is the Apple-clang reference side anyway, so the
# compiler confound it documents is unchanged.
ref:
	@test -f cmake-build/apple/src/libllama.a || { echo "missing the reference libraries -- run 'make buildmacos'" >&2; exit 2; }
	@clang harness/raw_completion.c -o $(SCRATCH)/raw_completion -std=c11 -w -I llama.cpp/include -I llama.cpp/ggml/include cmake-build/apple/src/libllama.a cmake-build/apple/ggml/src/libggml.a cmake-build/apple/ggml/src/ggml-metal/libggml-metal.a cmake-build/apple/ggml/src/libggml-cpu.a cmake-build/apple/ggml/src/libggml-base.a -lc++ -framework Foundation -framework Metal -framework MetalKit -framework Accelerate -F "$$(xcrun --sdk macosx --show-sdk-path)/System/Library/Frameworks"
	@printf '\n$$ raw_completion %s %s %s --temp %s -s %s\n' "$(MODEL)" "'$(PROMPT)'" "$(NPRED)" "$(TEMP)" "$(SEED)"
	@$(SCRATCH)/raw_completion "$(MODEL)" "$(PROMPT)" $(NPRED) show $(TEMP) $(SEED) 2>/dev/null
	@printf '\n'

# Run upstream's llama-cli as a chat REPL; output will not match `make port`.
ref-chat:
	@test -x "$(REF_CLI)" || { echo "missing $(REF_CLI) -- run 'make zmake'" >&2; exit 2; }
	@printf '\n$$ %s -m %s -p %s -n %s -st %s\n' "$(REF_CLI)" "$(MODEL)" "'$(PROMPT)'" "$(NPRED)" "$(ARGS)"
	@$(REF_CLI) -m "$(MODEL)" -p "$(PROMPT)" -n $(NPRED) -st $(ARGS) </dev/null
	@printf '\n'

#
# Models
#

# Download Qwen3.5-2B Q4_K_M; standard (1.28GB).
qwen35:
	curl -L -o Qwen3.5-2B-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf

# Download Qwen3.5-2B IQ4_XS; smaller (1.17GB).
qwen35xs:
	curl -L -o Qwen3.5-2B-IQ4_XS.gguf https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-IQ4_XS.gguf

# Download Qwen3.5-2B UD-Q4_K_XL; dynamic quant for better quality (1.34GB).
qwen35xl:
	curl -L -o Qwen3.5-2B-UD-Q4_K_XL.gguf https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-UD-Q4_K_XL.gguf
