###############################################
#
# Makefile
#
###############################################

.DEFAULT_GOAL := build

.PHONY: build llama.cpp probe graph-diff backend-ops parity-cli port ref ref-chat

# CMake is only needed for the stage-1 reference builds. Prefer a local install
# under .tools (see `make cmake`), fall back to whatever is on PATH.
CMAKE ?= $(firstword $(wildcard $(CURDIR)/.tools/cmake-*/CMake.app/Contents/bin/cmake) cmake)

# Shared flag set for the macOS arm64 target. The stage-1 builds differ only in
# which compiler drives them, so the options live in one place.
MACOS_FLAGS = -DCMAKE_SYSTEM_NAME=Darwin \
              -DCMAKE_OSX_SYSROOT=macosx \
              -DCMAKE_OSX_ARCHITECTURES=arm64 \
              -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
              -DCMAKE_BUILD_TYPE=Release \
              -DGGML_METAL=ON \
              -DGGML_METAL_EMBED_LIBRARY=ON \
              -DGGML_ACCELERATE=ON \
              -DGGML_BLAS=OFF \
              -DGGML_OPENMP=OFF \
              -DGGML_NATIVE=OFF \
              -DBUILD_SHARED_LIBS=OFF \
              -DLLAMA_CURL=OFF

ZIG_CC  = $(CURDIR)/scripts/zigcc
ZIG_CXX = $(CURDIR)/scripts/zigcxx

# Download CMake locally. Not on PATH on this machine, and /usr/local is
# root-owned, so it lands under .tools/ rather than being installed system-wide.
cmake:
	mkdir -p .tools
	cd .tools; curl -sSL -o cmake.tar.gz https://github.com/Kitware/CMake/releases/download/v4.4.3/cmake-4.4.3-macos-universal.tar.gz
	cd .tools; tar xzf cmake.tar.gz && rm cmake.tar.gz
	$(CMAKE) --version

clone:
	git clone https://github.com/ggerganov/llama.cpp.git
	cd llama.cpp; git checkout v0.3.0	

clean:
	rm -rf cmake-build zig-out .zig-cache

# com.apple.developer.kernel.increased-memory-limit
buildios:
	-mkdir cmake-build
	cd cmake-build; cmake ../llama.cpp -G "Unix Makefiles" -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=18.0 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_ACCELERATE=ON -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
	cd cmake-build; make -j

# Stage 1 reference build: llama.cpp compiled by Apple clang.
#
# This is the control. Its binary is what a Zig-compiled or Zig-ported binary
# gets diffed against, so it must NOT use the Zig toolchain.
buildmacos:
	-mkdir -p cmake-build/apple
	cd cmake-build/apple; $(CMAKE) $(CURDIR)/llama.cpp -G "Unix Makefiles" $(MACOS_FLAGS)
	cd cmake-build/apple; make -j

# Stage 1 proper: the same sources and the same flags, compiled by zig cc.
#
# CMake still drives the build; only the compiler changes. The Xcode generator
# cannot be used here -- it shells out to xcodebuild, which selects Apple clang
# and ignores CMAKE_C_COMPILER.
buildmacos-zig:
	-mkdir -p cmake-build/zig
	cd cmake-build/zig; $(CMAKE) $(CURDIR)/llama.cpp -G "Unix Makefiles" $(MACOS_FLAGS) \
	    -DCMAKE_C_COMPILER=$(ZIG_CC) \
	    -DCMAKE_CXX_COMPILER=$(ZIG_CXX) \
	    -DCMAKE_ASM_COMPILER=$(ZIG_CC)
	cd cmake-build/zig; make -j

# NOTE: this runs *upstream's* llama-cli from the CMake reference build, not
# ours, and the model path is a leftover that this repository does not contain.
# For our CLI, use `make demo`. Kept only because the reference binary is still
# occasionally worth running by hand.
cli:
	./cmake-build/apple/bin/llama-cli -m models/Qwen2.5-72B-Instruct-Q4_K_M.gguf -ngl 99 -p "You are an assistant. How are you?"

#
# Zig
#

# ---------------------------------------------
# Build
# ---------------------------------------------

# Build the library and CLI.
build:
	zig build

# Build optimized for release.
dist:
	zig build --release=fast

# Everything that can be checked without a model or a reference build:
# formatting, the scaffold, the unit tests, and the ported ggml.
#
# The ported code is tested in both modes on purpose. A translation unit cannot
# be linked beside the C it replaces until it is complete, so `test-port` is the
# only check these files get -- and some of them behave differently under
# NDEBUG, where ggml returns an error instead of aborting. A debug-only run
# would skip those paths entirely.
#
# This does not prove the port is correct, only that it is consistent. Use
# `make parity-port` for correctness; it needs a model and a reference build.
.PHONY: validate
validate:
	@printf '\n== formatting ==\n'
	zig fmt --check build.zig build/*.zig harness/*.zig src/*.zig src/ggml/*.zig src/ggml/quants/*.zig src/ggml/cpu/*.zig src/ggml/cpu/quants/*.zig src/ggml/cpu/quants/arm/*.zig cli/*.zig
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
	@printf '\n== graph diff vs reference ==\n'
	@if [ -f cmake-build/apple/ggml/src/libggml.a ]; then \
		zig build ported-lib --release=fast && ./scripts/graph-diff; \
	else \
		echo "  skipped: no reference build (run 'make buildmacos')"; \
	fi
	@printf '\nvalidate: OK\n'

# Diff the token stream of the Apple-clang and zig cc builds. Greedy sampling
# with a fixed seed makes the output a function of the arithmetic alone, so a
# difference here is a real difference in what the code computes.
# Requires both `make buildmacos` and `make buildmacos-zig`.
parity:
	./scripts/parity $(MODEL) "$(PROMPT)"

# Build llama-cli with the Zig build system instead of CMake. Lives in its own
# repository at llama.cpp.zmake, which reads the sources from a checkout nested
# inside it and does not modify them. See llama.cpp.zmake/README.md.
zmake:
	cd llama.cpp.zmake; $(MAKE) cli

# Prove ported code is on the execution path. Output comparison cannot tell
# correct code from code that never runs; this can.
probe:
	./scripts/probe-ported $(MODEL)

# Check every `Ports X (file:line)` citation still points at the C it names,
# and that every file citing upstream declares the pinned commit. These are the
# port's map back to llama.cpp: when the pin moves, they say which of our
# functions an upstream change lands in.
port-links:
	./scripts/port-links

# Diff the graphs the ported constructors build against the reference ones.
# Needs no model: graph construction only. Catches a wrong shape or a value in
# the wrong op_params slot, which token parity is far too coarse to see.
graph-diff:
	zig build ported-lib --release=fast
	./scripts/graph-diff

# Our llamazig CLI against the C reference, end to end. Covers what
# parity-port cannot: argument parsing, tokenizer flags, the sampler chain and
# the decode loop all live in the binary, not the library.
# Its own token count, deliberately bounded: this is a gate that runs six
# prompts twice, not a demo, and it should not inherit NPRED's -1.
PARITY_NPRED ?= 32
parity-cli:
	zig build cli --release=fast
	./scripts/parity-cli $(MODEL) $(PARITY_NPRED)

# Upstream's test-backend-ops against the ported ggml: ~21k op configurations,
# Metal against CPU. The broadest exercise of the ported constructors, and the
# only gate that *executes* them at this scale. It cannot catch a constructor
# that is consistently wrong -- both backends read the same op_params -- so it
# complements graph-diff rather than replacing it. `--diff` also runs the C
# reference and compares output.
backend-ops:
	zig build reference --release=fast
	./scripts/backend-ops $(ARGS)

# Diff inference output between the ported libraries and the Apple-clang
# reference libraries, using one driver linked against each. This is the gate
# for stages 3 and 4: `parity` catches toolchain differences, this catches
# porting bugs.
# Requires: make buildmacos && zig build reference --release=fast
parity-port:
	./scripts/parity-port $(MODEL)

# Build the llama.cpp reference libraries (ggml + libllama) with the Zig
# toolchain. No CMake involved.
reference:
	zig build reference

# Load a model through the reference build and generate, to prove the Metal
# path works end to end. Override with: make smoke MODEL=... PROMPT=...
MODEL  ?= Qwen3.5-2B-Q4_K_M.gguf
PROMPT ?= The capital of France is

# Bounded on purpose. Upstream's default for -n is -1 (generate until the model
# stops or the context fills), and matching that made these targets run for an
# unpredictable length -- a reasoning model with a greedy sampler can fail to
# emit an end token and run to the context limit, which looks like a hang.
#
# 512 is large enough that a normal answer is not cut off, which a smaller cap
# was. Pass NPRED=-1 for upstream's real behaviour, or NPRED=N for a tighter
# cap.
NPRED  ?= 512

# Sampling, shared by `port` and `ref` so the two are comparable.
#
# The seed is fixed rather than random -- upstream's default is a random one,
# but a pair of targets meant to be diffed has to be reproducible, and at
# temp 0.80 an unlucky seed sends a 2B model into a repetition loop. Pass
# TEMP=0 for greedy, or SEED=$$RANDOM to sample freely.
TEMP   ?= 0.8
SEED   ?= 42
smoke:
	zig build smoke -- "$(MODEL)" "$(PROMPT)"

# Run our ported CLI, and the reference CLI, on the same arguments.
#
# `make port` and `make ref` are a matched pair: same MODEL, PROMPT, NPRED and
# ARGS, so the two can be run back to back.
#
# **They are not doing quite the same thing, and the outputs will differ.**
# Upstream's llama-cli has no raw-completion mode -- `-st` runs a *single turn
# of a conversation*, which applies the model's chat template and so feeds it a
# different prompt from the one we send. You will see it in the numbers: the
# prompt token count, and therefore the prompt t/s, do not match. Use these to
# eyeball behaviour, not to judge correctness.
#
# `make parity-cli` is the mechanical comparison, and it deliberately does not
# use llama-cli for exactly this reason: it links its own raw C driver so both
# sides tokenize the same bytes.
#
#   make port ARGS="--temp 0 -s 42"          # reproducible; default is temp 0.80
#   make ref  ARGS="--temp 0 -s 42"
#   make port PROMPT="def fibonacci(n):" NPRED=128
#
# Sampling defaults are upstream's, so successive runs differ unless you pass
# --temp 0.

PORT_CLI := ./zig-out/bin/llamazig
REF_CLI  := llama.cpp.zmake/zig-out/macos/bin/llama-cli

port:
	@zig build cli --release=fast
	@printf '\n$$ %s -m %s -p %s -n %s --temp %s -s %s %s\n' \
		"$(PORT_CLI)" "$(MODEL)" "'$(PROMPT)'" "$(NPRED)" "$(TEMP)" "$(SEED)" "$(ARGS)"
	@$(PORT_CLI) -m "$(MODEL)" -p "$(PROMPT)" -n $(NPRED) --temp $(TEMP) -s $(SEED) -st $(ARGS)
	@printf '\n'

# The reference half of the pair: the same tokenize/decode/sample loop our CLI
# runs, linked against the **stock C libraries**.
#
# This is `ref` rather than upstream's `llama-cli` because `llama-cli` has no
# raw-completion mode -- its `-st` is a single turn of a *conversation*, so it
# applies the model's chat template and is fed a different prompt entirely.
# Comparing against that measures the template, not the port. `make ref-chat`
# runs it if you want to see it.
#
# `make port` and `make ref` should produce identical text. If they ever do
# not, that is a real porting bug.
ref:
	@test -f cmake-build/apple/src/libllama.a || { echo "missing the reference libraries -- run 'make buildmacos'" >&2; exit 2; }
	@zig cc harness/raw_completion.c -o $(SCRATCH)/raw_completion -std=c11 -w \
		-I llama.cpp/include -I llama.cpp/ggml/include \
		cmake-build/apple/src/libllama.a \
		cmake-build/apple/ggml/src/libggml.a \
		cmake-build/apple/ggml/src/ggml-metal/libggml-metal.a \
		cmake-build/apple/ggml/src/libggml-cpu.a \
		cmake-build/apple/ggml/src/libggml-base.a \
		-lc++ -framework Foundation -framework Metal -framework MetalKit -framework Accelerate \
		-F "$$(xcrun --sdk macosx --show-sdk-path)/System/Library/Frameworks"
	@printf '\n$$ raw_completion %s %s %s --temp %s -s %s\n' \
		"$(MODEL)" "'$(PROMPT)'" "$(NPRED)" "$(TEMP)" "$(SEED)"
	@$(SCRATCH)/raw_completion "$(MODEL)" "$(PROMPT)" $(NPRED) show $(TEMP) $(SEED) 2>/dev/null
	@printf '\n'

# Where ref-raw puts its driver.
SCRATCH ?= /tmp

# Upstream's `llama-cli` from llama.cpp.zmake -- stock C and C++ sources, Zig
# build system, no port involved. Build it with `make zmake`.
#
# **This is a chat REPL, not a completion.** `-st` runs a single turn of a
# conversation: it prints a banner and a command list, applies the model's chat
# template, and answers as an assistant. Its output will not match `make port`
# and is not meant to -- the model is being asked a different question. Use
# `make ref` for a comparable reference.
ref-chat:
	@test -x "$(REF_CLI)" || { echo "missing $(REF_CLI) -- run 'make zmake'" >&2; exit 2; }
	@printf '\n$$ %s -m %s -p %s -n %s -st %s\n' \
		"$(REF_CLI)" "$(MODEL)" "'$(PROMPT)'" "$(NPRED)" "$(ARGS)"
	@$(REF_CLI) -m "$(MODEL)" -p "$(PROMPT)" -n $(NPRED) -st $(ARGS) </dev/null
	@printf '\n'

#
# Models
#

# standard (1.28GB)
qwen35:
	curl -L -o Qwen3.5-2B-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf

# smaller (1.17GB)
qwen35xs:
	curl -L -o Qwen3.5-2B-IQ4_XS.gguf https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-IQ4_XS.gguf

# dynamic quant for better quality (1.34GB)
qwen35xl:
	curl -L -o Qwen3.5-2B-UD-Q4_K_XL.gguf https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-UD-Q4_K_XL.gguf