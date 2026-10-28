###############################################
#
# Makefile
#
###############################################

# Optional include to extend the makefile targets
-include ../styleguide/Makefile.in

# The include's validate is `clean format lint build test`; `test` below is
# this project's part of it. These are the directories format and lint walk.
ZIG_DIRS := src cli tools harness build

# Lint covers only what is clean so far: build.zig, build/, cli/ and tools/,
# less the files below. src/ and harness/ are not linted yet -- the exported
# C names alone are 1,805 case-convention hits. Shrink the list as files are
# cleaned; a new file in a linted directory is linted from the start.
LINT_DIRS := build cli tools
LINT_EXCLUDE := build/llamacpp.zig build/vulkan_shaders.zig \
	cli/chat.zig cli/fit.zig cli/main.zig cli/session.zig \
	tools/common/arg_options.zig tools/common/arg.zig tools/common/console.zig \
	tools/common/init.zig tools/cvector_generator.zig tools/fit_params.zig \
	tools/llama_bench.zig tools/quantize.zig
LINT_FILES := $(filter-out $(LINT_EXCLUDE),build.zig $(shell find $(LINT_DIRS) -name '*.zig' | sort))

.DEFAULT_GOAL := build

.PHONY: test distclean build cli llama.cpp probe graph-diff sched-diff ops-diff repack-diff sampler-diff host-arms-diff vk-toolchain vk-repro vk-builds vk-backend-diff vk-node-diff vk-ops-diff vk-push-layout vk-parity-cli blas-builds blas-diff blas-node-diff x86-quants-diff x86-repack-diff x86-ops-diff x86-node-diff x86-gates win-host-diffs libcxx-diff regex-diff unicode-diff vocab-diff arch-diff loader-diff batch-diff chat-diff jinja-diff context-diff arch-fixtures arch-model-diff gemma4-assistant-diff adapter-diff api-diff abi-check tuning-diff struct-layout metal-diff backend-ops parity-cli cnv-diff fit-diff tools-diff parity-port parity-port-cpu node-diff cuda-ops-diff cuda-node-diff cuda-backend-diff cuda-parity-cli cuda-require-linux port oneshot chat ref ref-chat validate xcframework xcframework-check perf perf-cpu

CMAKE ?= $(firstword $(wildcard $(CURDIR)/.tools/cmake-*/CMake.app/Contents/bin/cmake) cmake)

MACOS_FLAGS = -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_OSX_SYSROOT=macosx -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_ACCELERATE=ON -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF

ZIG_CC  = $(CURDIR)/scripts/zigcc
ZIG_CXX = $(CURDIR)/scripts/zigcxx

MODEL   ?= Qwen3.5-2B-Q4_K_M.gguf
# `make oneshot` / `make chat` models: the zigtrain base and SFT models
# `zigtrain export` writes.
TRAIN_GGUF       ?= $(or $(ZIGTRAIN_BASE_DIR),$(HOME)/.cache/zigtrain)/gguf
TRAIN_MODEL      ?= $(TRAIN_GGUF)/plain-d6_base_005000.gguf
TRAIN_CHAT_MODEL ?= $(TRAIN_GGUF)/plain-d6_sft_001499.gguf
PROMPT  ?= The capital of France is
NPRED   ?= 512
TEMP    ?= 0.8
SEED    ?= 42
SCRATCH ?= /tmp

PARITY_NPRED ?= 32

# The CPU the bit-exact gates build both libraries for: ops-diff,
# repack-diff, node-diff, parity-port. The kernel arms that exist depend on
# it -- i8mm's two-row dot products and gemms compile only for a CPU that
# has i8mm -- so a target that ships for apple_m3 is gated at apple_m3 too:
#   make ops-diff repack-diff node-diff GATE_CPU=apple_m3
# See scripts/gate-cpu.
GATE_CPU ?= baseline
export GATE_CPU

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

# Delete the Zig build output. validate runs this first, so it leaves the
# CMake reference alone: without cmake-build/ sched-diff and graph-diff skip.
clean:
	rm -rf zig-out .zig-cache

# Delete all build output, the CMake reference builds included.
distclean: clean
	rm -rf cmake-build

#
# Reference builds
#

# Build the port as llama.xcframework, a drop-in for the one upstream's
# build-xcframework.sh produces: macOS, iOS, tvOS and visionOS, devices and
# simulators. Everything lives in xcframework/; see xcframework/README.md.
xcframework:
	$(MAKE) -C xcframework

# Performance of the stock C against the port; see metrics/README.md.
perf:
	$(MAKE) -C metrics

# The same on the CPU device alone, so the ported CPU kernels carry the model.
perf-cpu:
	$(MAKE) -C metrics cpu

# The xcframework plus every gate on it, and `validate ops-diff`.
xcframework-check:
	$(MAKE) -C xcframework check

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

# Without the styleguide checkout (the gate machines have none), validate is
# the format check, the scaffold and the checks below.
ifeq ($(wildcard ../styleguide/Makefile.in),)
validate:
	@printf '\n== formatting ==\n'
	zig fmt --check build.zig $(ZIG_DIRS)
	@printf '\n== scaffold ==\n'
	@$(MAKE) --no-print-directory build test
	@printf '\nvalidate: OK\n'
endif

# Run every check that needs no model or reference build.
test:
	@printf '\n== unit tests ==\n'
	zig build test --summary all
	@printf '\n== ported ggml (debug) ==\n'
	zig build test-port --summary all
	@printf '\n== ported ggml (release) ==\n'
	zig build test-port --release=fast --summary all
	@printf '\n== ported libllama (debug) ==\n'
	zig build test-llama --summary all
	@printf '\n== ported libmtmd (debug) ==\n'
	zig build test-mtmd --summary all
	@printf '\n== libc++ algorithms vs the SDK templates ==\n'
	./scripts/libcxx-diff
	@printf '\n== std::regex vs libc++ ==\n'
	zig build reference --release=fast
	./scripts/regex-diff
	@printf '\n== unicode.cpp vs the port ==\n'
	./scripts/unicode-diff
	@printf '\n== llama-vocab.cpp vs the port ==\n'
	./scripts/vocab-diff
	@printf '\n== llama-arch.cpp vs the port ==\n'
	./scripts/arch-diff
	@printf '\n== llama-model-loader.cpp vs the port ==\n'
	./scripts/loader-diff
	@printf '\n== llama-batch.cpp vs the port ==\n'
	./scripts/batch-diff
	@printf '\n== llama-chat.cpp vs the port ==\n'
	./scripts/chat-diff
	@printf '\n== llama.h at the C boundary vs the port ==\n'
	./scripts/api-diff
	@printf '\n== ported symbol coverage ==\n'
	./scripts/port-coverage
	@printf '\n== libmtmd exports ==\n'
	zig build mtmd-ref --release=fast
	./scripts/mtmd-exports
	@printf '\n== the libraries for iOS (no std.debug.print or @panic in library code) ==\n'
	@# Zig 0.17 cannot compile std.Io.Threaded for iOS, which std.debug.print
	@# reaches; nothing else would notice until the xcframework next builds.
	@# The xcframework's iOS device slice: libggml with BLAS, libmtmd with
	@# libllama inside, for the A7 upstream's framework is built for.
	zig build xcframework -Dblas -Dtarget=aarch64-ios.16.4 -Dcpu=apple_a7 --release=fast --prefix zig-out/ios-check
	@printf '\n== upstream citations ==\n'
	./scripts/port-links
	@printf '\n== scheduler diff vs reference ==\n'
	@if [ -f cmake-build/apple/ggml/src/libggml.a ]; then ./scripts/sched-diff; else echo "  skipped: no reference build (run 'make buildmacos')"; fi
	@printf '\n== graph diff vs reference ==\n'
	@if [ -f cmake-build/apple/ggml/src/libggml.a ]; then zig build ported-lib --release=fast && ./scripts/graph-diff; else echo "  skipped: no reference build (run 'make buildmacos')"; fi
	@printf '\ntest: OK\n'

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
	cd llama.cpp.zmake && zig build lib --release=fast -Dcpu=$(GATE_CPU)
	zig build reference --release=fast -Dcpu=$(GATE_CPU)
	./scripts/ops-diff

# Compare the ported Metal vtables against the reference's, directly.
# node-diff --gpu runs the graph and sees almost none of ggml-metal.cpp:
# measured, dropping the FLASH_ATTN_EXT scratch terms from get_alloc_size
# passed it. See the header of scripts/metal-diff.
metal-diff:
	zig build reference --release=fast
	./scripts/metal-diff

# Compare the hand-declared Metal structs against the real headers.
# device_c.zig hand-declares ggml_metal_device_props rather than widening
# impl.zig's cImport; this asks the C compiler whether the transcription is
# right, and whether the generated metal/impl_c.zig and metal/kargs.zig
# agree with ggml-metal-impl.h -- 65 kargs structs and 936 fields the
# Metal kernels read by offset. Regenerate those with scripts/gen-kargs.
# See the header of scripts/struct-layout.
struct-layout:
	zig build reference --release=fast
	./scripts/struct-layout

# Sweep the flash-attention tuning lookup against the reference.
# tuning_table.zig is 936 transcribed rows and no other gate can tell a
# dropped row from a kept one. See the header of scripts/tuning-diff.
tuning-diff:
	zig build reference --release=fast
	./scripts/tuning-diff

# Call the port's by-value-struct entry points as a C caller does.
# The gate for a toolchain bug: Zig 0.16 miscompiles an extern struct
# received by value across the C ABI unless its size is a multiple of 4
# with no padding. See the header of scripts/abi-check.
abi-check:
	zig build reference --release=fast
	./scripts/abi-check

# Diff every interleaved repack kernel against the reference C++, on bits.
# The only oracle those 36 kernels have: test-backend-ops never allocates a
# CPU_REPACK buffer, ops-diff's tensors live in a plain CPU buffer, and
# `make port` runs on Metal. node-diff reaches them but stops at the first
# divergent node. See the header of scripts/repack-diff.
repack-diff:
	zig build reference --release=fast -Dcpu=$(GATE_CPU)
	./scripts/repack-diff

# Every public sampler and the grammar, ported vs reference, on bits.
# parity-cli builds one chain and reads one token a step; this prints every
# candidate's id, logit and probability after every call, and reaches the
# samplers, grammars and backend-sampling path the CLI never configures. See
# the header of scripts/sampler-diff.
sampler-diff:
	cd llama.cpp.zmake && zig build lib --release=fast
	zig build reference --release=fast
	./scripts/sampler-diff $(MODEL)

# Stage 7's first measurement: is the reference Vulkan backend reproducible
# run to run, per node and fused? Builds llama.cpp.zmake with its Vulkan
# backend with the pinned toolchain in .tools/vulkan (`make vk-toolchain`;
# PLAN.md, Decision 55), so it runs on the GB10 and not on this Mac's
# Metal. See the header of scripts/vk-repro.
VK_TOOLS = $(CURDIR)/.tools/vulkan
VK_FLAGS = -Dvulkan -Dglslc=$(VK_TOOLS)/bin/glslc -Dvulkan-include=$(VK_TOOLS)/include

vk-toolchain:
	./scripts/vk-toolchain

vk-repro:
	cd llama.cpp.zmake && zig build lib --release=fast $(VK_FLAGS)
	./scripts/vk-repro $(MODEL)

# The Vulkan gates (PLAN.md, "Stage 7 — Vulkan"): this repository's libggml
# with upstream's ggml-vulkan.cpp (vk-upstream) and with the Zig host layer
# (vk-port), CUDA off so the only GPU device is Vulkan's, and upstream's
# backend with its exports renamed for the in-process gates. vk-backend-diff
# asks both registries the same questions; vk-ops-diff runs test-backend-ops
# with the reference where the CPU goes, on bits; vk-node-diff diffs every
# node of a decode, per node and fused, on every arm of the GB10;
# vk-push-layout checks the generated push constants against the C++'s.
# See each script.
VK_GATE_FLAGS = --release=fast -Dvulkan -Dcuda=false

vk-builds:
	zig build reference $(VK_GATE_FLAGS) -Dvulkan-port=false --prefix zig-out/vk-upstream
	zig build reference vk-ref $(VK_GATE_FLAGS) --prefix zig-out/vk-port

vk-backend-diff: vk-builds
	./scripts/vk-backend-diff

vk-node-diff: vk-builds
	./scripts/vk-node-diff $(MODEL)

# Our llama-cli on the Vulkan device against raw_completion.c over
# upstream's libllama and Vulkan host layer, greedy, six prompts.
vk-parity-cli: vk-builds
	zig build cli $(VK_GATE_FLAGS) --prefix zig-out/vk-cli
	./scripts/vk-parity-cli $(MODEL) $(PARITY_NPRED)

vk-ops-diff: vk-builds
	./scripts/vk-ops-diff $(ARGS)

vk-push-layout:
	zig build vk-push-layout $(VK_GATE_FLAGS)

# The BLAS backend (PLAN.md, Decision 60): Apple only, over Accelerate.
# blas-diff runs upstream's ggml-blas.cpp, its exports renamed, beside the
# port in one process -- the vtables, supports_op swept, every type's
# computation on bits; blas-node-diff diffs a decode node by node with BLAS
# in the scheduler. See each script.
blas-builds:
	zig build reference --release=fast -Dblas -Dblas-port=false --prefix zig-out/blas-upstream
	zig build reference --release=fast -Dblas --prefix zig-out/blas-port
	zig build reference --release=fast --prefix zig-out/noblas

blas-diff: blas-builds
	./scripts/blas-diff

blas-node-diff: blas-builds
	./scripts/blas-node-diff

# The x86 gates, run on the Windows laptop (Decision 57) through
# zigupdater's link: each cross-builds both sides for
# x86_64-windows-gnu at X86_CPU (x86_64_v3), pushes the programs and reads
# back the verdict. The agent's window must be open there. AVX-512 --
# X86_CPU=x86_64_v4 or tigerlake -- runs on X86_AGENT=JWHLPTOP, a Tiger Lake. x86-quants-diff
# and x86-repack-diff are the per-kernel oracles; x86-ops-diff and
# x86-node-diff are scripts/ops-diff and scripts/node-diff on Intel silicon,
# the second with a model on the laptop (X86_MODEL). See the header of each
# script, and scripts/win-run.
x86-quants-diff:
	./scripts/x86-quants-diff

x86-repack-diff:
	./scripts/x86-repack-diff

x86-ops-diff:
	./scripts/x86-ops-diff

x86-node-diff:
	./scripts/x86-node-diff

x86-gates: x86-quants-diff x86-repack-diff x86-ops-diff x86-node-diff

# The Windows pickup's host gates (PLAN.md, "Windows pickup"), on the laptop:
# host-arms-diff (priority, power throttling, affinity, the clock seed),
# libcxx-diff, regex-diff and unicode-diff, each cross-built by its own
# script under WIN_X86=1 (scripts/win-target). host-arms-diff goes first: it
# builds the Windows port the other three link.
win-host-diffs:
	WIN_X86=1 ./scripts/host-arms-diff
	WIN_X86=1 ./scripts/libcxx-diff
	WIN_X86=1 ./scripts/regex-diff
	WIN_X86=1 ./scripts/unicode-diff

# The CPU backend's host arms -- thread priority, CPU affinity, NUMA, the
# sampler's default seed -- read back from the OS on every worker, ported
# against reference, one process per scenario. Needs no model. The gate for
# the Linux arms on Linux; on macOS it covers the Apple ones. See the header
# of scripts/host-arms-diff.
host-arms-diff:
	cd llama.cpp.zmake && zig build lib --release=fast
	zig build reference --release=fast
	./scripts/host-arms-diff

# The libc++ sort, heap, mt19937 and distributions src/llama/libcxx.zig
# reproduces, against the real templates, in one process. The sampler's tie
# order and draws depend on them. See the header of scripts/libcxx-diff.
libcxx-diff:
	./scripts/libcxx-diff

# src/llama/regex.zig against libc++'s std::regex, which the lazy grammar's
# trigger patterns use. See the header of scripts/regex-diff.
regex-diff:
	zig build reference --release=fast
	./scripts/regex-diff

# src/llama/unicode.zig against the unmodified unicode.cpp, in one process:
# every code point, UTF-8 decoding, the byte map, and every pre-tokenizer
# pattern set in llama-vocab.cpp. See the header of scripts/unicode-diff.
unicode-diff:
	zig build reference --release=fast
	./scripts/unicode-diff

# src/llama/vocab/ against the C++ vocabulary inside libllama.a, in one
# process: every vocabulary file upstream ships, the local Qwen3.5 model,
# and synthesized files for the tokenizer types and error paths no real
# file reaches. See the header of scripts/vocab-diff.
vocab-diff:
	zig build reference --release=fast
	./scripts/vocab-diff

# llama-arch.cpp's tables and lookups, Zig against the C++ in one process:
# every architecture, every metadata key for every architecture, every
# tensor name and info. The tables are generated by scripts/gen-arch.
arch-diff:
	zig build reference --release=fast
	./scripts/arch-diff

# The loader, hyperparameters and model metadata, Zig against the C++ in one
# process: the metadata half of llama_model_load on every upstream vocabulary
# file, the local Qwen3.5 model, and synthesized files.
loader-diff:
	zig build reference --release=fast
	./scripts/loader-diff

# The batch splitter, Zig against the C++ in one process.
batch-diff:
	zig build reference --release=fast
	./scripts/batch-diff

# The built-in chat templates, Zig against the C++ in one process.
chat-diff:
	zig build reference --release=fast
	./scripts/chat-diff

# The CLI's Jinja chat templates (cli/chat.zig over zigjinja) against
# upstream's common/jinja, on every template in llama.cpp/models/templates:
# the text, which side raises, and that ours never trusts a byte upstream
# marks as input. See the header of scripts/jinja-diff.
jinja-diff:
	zig build reference --release=fast
	./scripts/jinja-diff

# The context, the memory modules, the graph and Qwen3.5's graph: a decode
# script run through llama.h and through the port, every graph node, every
# output's bits and the state API compared; on the GPU, the CPU, a partial
# offload, a device without the fused Gated Delta Net, and tensor
# parallelism. ARGS=--quick for the GPU's multi-sequence cases only.
context-diff:
	zig build reference --release=fast
	./scripts/context-diff $(ARGS)

# Every llama.h entry point the context does not own -- model, vocabulary,
# loading, chat, batch, logging -- at the C boundary, Zig against the C++.
api-diff:
	zig build reference --release=fast
	./scripts/api-diff

# LoRA and control-vector adapters, Zig against the C++ in one process.
adapter-diff:
	zig build reference --release=fast
	./scripts/adapter-diff $(ARGS)

# One small random model per architecture, written by upstream's own
# tests/test-llama-archs.cpp, for arch-model-diff. See scripts/gen-arch-fixtures.
arch-fixtures:
	zig build reference --release=fast
	./scripts/gen-arch-fixtures

# Every architecture on its fixture, the port against the C++: context-diff
# across architectures. Counts the architectures not yet ported rather than
# failing on them. ARGS=<substring...> narrows it. See scripts/arch-model-diff.
arch-model-diff:
	zig build reference --release=fast
	@[ -d zig-out/arch-fixtures ] || ./scripts/gen-arch-fixtures
	./scripts/arch-model-diff zig-out/arch-fixtures $(ARGS)

# The Gemma 4 assistant over a Gemma 4 target, which arch-model-diff cannot
# reach. See scripts/gemma4-assistant-diff.
gemma4-assistant-diff:
	zig build reference --release=fast
	./scripts/gemma4-assistant-diff

# Diff our CLI's output against the C reference, end to end.
parity-cli: buildcli
	./scripts/parity-cli $(MODEL) $(PARITY_NPRED)

# Our CLI's conversation mode (-cnv) against a reference conversation over
# upstream's Jinja engine and libllama: replies, stderr and the cache each
# turn kept. Needs both models: make qwen35 qwen3. See scripts/cnv-diff.
cnv-diff:
	zig build cli --release=fast
	./scripts/cnv-diff $(ARGS)

# cli/fit.zig -- what --fit decides from free memory -- against upstream's
# common/fit.cpp, on fake GPUs with chosen memory, the CPU alone and Metal.
# Needs make qwen35 qwen3 and make arch-fixtures. See scripts/fit-diff.
fit-diff:
	zig build reference --release=fast
	./scripts/fit-diff $(ARGS)

# Upstream's other tools (PLAN.md, Decisions 61 and 62): each of our twelve ports run
# beside upstream's own binary on harness/tools/<tool>.cases, stdout,
# stderr, exit code and every file written compared. TOOL=<name> runs one.
TOOLS ?= gguf-split quantize tokenize fit-params export-lora batched-bench cvector-generator imatrix perplexity bench tts mtmd-cli
tools-diff:
	@set -e; for t in $(if $(TOOL),$(TOOL),$(TOOLS)); do ./scripts/tools-diff $$t; done

# libmtmd (Stage 8, PLAN.md Decision 62). mmproj-fixtures writes one small
# random mmproj per projector type and checks each through upstream's
# libmtmd; the gates compare the port against it.
.PHONY: mmproj-fixtures mtmd-diff mtmd-exports mtmd-image-diff mtmd-audio-diff mtmd-decode-diff mtmd-gates
mmproj-fixtures:
	./scripts/gen-mmproj-fixtures $(ARGS)

# libmtmd through mtmd.h and mtmd-helper.h, port against upstream, on every
# fixture: logs, every encoder node, chunks, embeddings and logits on bits,
# TTS output. DEVS=cpu|gpu|both, FILTER=<projector substring>.
mtmd-diff:
	zig build reference mtmd-ref --release=fast
	./scripts/mtmd-diff $(if $(DEVS),$(DEVS),both) $(FILTER)

# The 82 mtmd_* exports, equal to the reference's.
mtmd-exports:
	zig build mtmd-ref --release=fast
	./scripts/mtmd-exports

# The image and audio preprocessors and the vendored decoders, each against
# upstream's translation unit compiled unmodified, in one process.
mtmd-image-diff:
	./scripts/mtmd-image-diff $(ARGS)
mtmd-audio-diff:
	./scripts/mtmd-audio-diff $(ARGS)
mtmd-decode-diff:
	./scripts/mtmd-decode-diff $(ARGS)

mtmd-gates: mtmd-exports mtmd-image-diff mtmd-audio-diff mtmd-decode-diff mtmd-diff

# Run upstream's test-backend-ops against the ported ggml.
backend-ops:
	zig build reference --release=fast
	./scripts/backend-ops $(ARGS)

# Diff inference output between the ported and reference libraries.
# Both sides --release=fast, against llama.cpp.zmake: see scripts/parity-port.
parity-port:
	cd llama.cpp.zmake && zig build lib --release=fast -Dcpu=$(GATE_CPU)
	zig build reference --release=fast -Dcpu=$(GATE_CPU)
	./scripts/parity-port $(MODEL)

# Every graph node of a real model's decode, ported vs reference, on bits. CPU
# by default; ARGS=--gpu for Metal. See the header of scripts/node-diff.
node-diff:
	cd llama.cpp.zmake && zig build lib --release=fast -Dcpu=$(GATE_CPU)
	zig build reference --release=fast -Dcpu=$(GATE_CPU)
	./scripts/node-diff $(ARGS) $(MODEL)

# The same, with the model on the CPU device alone, so inference runs through
# the ported ggml-cpu kernels -- which on a Metal machine it otherwise never
# does.
parity-port-cpu:
	cd llama.cpp.zmake && zig build lib --release=fast -Dcpu=$(GATE_CPU)
	zig build reference --release=fast -Dcpu=$(GATE_CPU)
	./scripts/parity-port --cpu $(MODEL)

# The CUDA gates (PLAN.md, "Stage 6 -- CUDA"). Linux with nvcc and an NVIDIA
# GPU only: the Mac has neither, and Metal's gates cannot run on the CUDA
# machine (Decision 43). Each builds what its script's `Requires:` line names
# into its own prefix, so the port and upstream's host layer never share an
# archive; the prefixes are the ones the scripts read. See the header of
# each script.

cuda-require-linux:
	@[ "$$(uname -s)" = Linux ] || { echo "the CUDA gates run on the Linux CUDA machine only (PLAN.md, Decision 43)" >&2; exit 2; }

# test-backend-ops on bits, the ported CUDA backend against upstream's.
cuda-ops-diff: cuda-require-linux
	zig build reference cuda-ref --release=fast --prefix zig-out/cuda-port
	./scripts/cuda-ops-diff $(ARGS)

# Every node of a decode, then the fused graph, ported host layer vs upstream's.
cuda-node-diff: cuda-require-linux
	zig build reference -Dcuda-port=false --release=fast --prefix zig-out/cuda-upstream
	zig build reference --release=fast --prefix zig-out/cuda-port
	./scripts/cuda-node-diff $(MODEL)

# The device, buffer-type and registry vtables: metal-diff for CUDA.
cuda-backend-diff: cuda-require-linux
	zig build reference cuda-ref --release=fast --prefix zig-out/cuda-port
	./scripts/cuda-backend-diff

# parity-cli for the CUDA machine.
cuda-parity-cli: cuda-require-linux
	zig build cli --release=fast
	zig build reference -Dcuda-port=false --release=fast --prefix zig-out/cuda-upstream
	./scripts/cuda-parity-cli $(MODEL) $(PARITY_NPRED)

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
# --jinja renders the model's own chat template, so a template's literal
# control tokens (a leading <|bos|>) are honoured.
cli: buildcli
	@$(PORT_CLI) -m "$(MODEL)" -cnv --jinja --temp $(TEMP) -s $(SEED) $(ARGS)

# Run our CLI on a raw completion.
port: buildcli
	@printf '\n$$ %s -m %s -p %s -n %s --temp %s -s %s %s\n' "$(PORT_CLI)" "$(MODEL)" "'$(PROMPT)'" "$(NPRED)" "$(TEMP)" "$(SEED)" "$(ARGS)"
	@$(PORT_CLI) -m "$(MODEL)" -p "$(PROMPT)" -n $(NPRED) --temp $(TEMP) -s $(SEED) -st $(ARGS)
	@printf '\n'

# Run `make port` on the zigtrain base model; TRAIN_MODEL picks another.
oneshot: MODEL = $(TRAIN_MODEL)
oneshot: port

# Run `make cli` on the zigtrain SFT model; TRAIN_CHAT_MODEL picks another.
chat: MODEL = $(TRAIN_CHAT_MODEL)
chat: cli

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

# Download Qwen3-0.6B Q8_0 (0.64GB): a plain KV cache, for cnv-diff's
# seq_rm path, which Qwen3.5's hybrid memory refuses.
qwen3:
	curl -L -o Qwen3-0.6B-Q8_0.gguf https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf

# Download Qwen3.5-2B IQ4_XS; smaller (1.17GB).
qwen35xs:
	curl -L -o Qwen3.5-2B-IQ4_XS.gguf https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-IQ4_XS.gguf

# Download Qwen3.5-2B UD-Q4_K_XL; dynamic quant for better quality (1.34GB).
qwen35xl:
	curl -L -o Qwen3.5-2B-UD-Q4_K_XL.gguf https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-UD-Q4_K_XL.gguf

# ---------------------------------------------
# Package
# ---------------------------------------------

.PHONY: pkg

# zigpkg's `zpkg` (../zigpkg) packages the CLI as its recipe in zigpkg's store
# says (the name and version come from build.zig.zon), then signs, notarizes
# and publishes it as the recipe says.
ZPKG ?= zpkg

# What the package installs (the recipe's .pkg root).
PKG_ROOT := zig-out/pkg/root

# Build the CLI into zig-out/pkg/root, then the package with zpkg.
pkg:
	@command -v $(ZPKG) >/dev/null || { echo "❌ $(ZPKG) not found: build ../zigpkg, or set ZPKG"; exit 1; }
	rm -rf $(PKG_ROOT)
	zig build cli --release=fast --prefix $(PKG_ROOT)
	$(ZPKG) --package
