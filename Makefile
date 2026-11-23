###############################################
#
# Makefile
#
###############################################

# Optional include to extend the makefile targets
-include ../styleguide/Makefile.in

# The include's validate is `clean format lint build test`; `test` below is
# this project's part of it. These are the directories format and lint walk.
ZIG_DIRS := src cli tools testing build

# Lint covers only what is clean so far: build.zig, build/, testing/build/, cli/ and tools/,
# less the files below. src/ and testing/harness/ are not linted yet -- the exported
# C names alone are 1,805 case-convention hits. Shrink the list as files are
# cleaned; a new file in a linted directory is linted from the start.
LINT_DIRS := build testing/build cli tools
LINT_EXCLUDE := build/llamacpp.zig build/vulkan_shaders.zig \
	cli/chat.zig cli/fit.zig cli/main.zig cli/session.zig \
	tools/common/arg_options.zig tools/common/arg.zig tools/common/console.zig \
	tools/common/init.zig tools/cvector_generator.zig tools/fit_params.zig \
	tools/llama_bench.zig tools/quantize.zig
LINT_FILES := $(filter-out $(LINT_EXCLUDE),build.zig $(shell find $(LINT_DIRS) -name '*.zig' | sort))

.DEFAULT_GOAL := build

.PHONY: test distclean build cli llama.cpp graph-diff sched-diff ops-diff repack-diff sampler-diff host-arms-diff vk-toolchain vk-repro vk-builds vk-backend-diff vk-node-diff vk-ops-diff vk-push-layout vk-parity-cli blas-builds blas-diff blas-node-diff x86-quants-diff x86-repack-diff x86-ops-diff x86-node-diff x86-gates win-host-diffs libcxx-diff regex-diff json-diff schema-diff peg-diff chat-msg-diff chat-parse-diff spec-diff server-diff server-task-diff server-core-consumer http-diff unicode-diff vocab-diff arch-diff loader-diff batch-diff chat-diff jinja-diff context-diff moe-slru-diff arch-fixtures arch-model-diff mtp-diff gemma4-assistant-diff adapter-diff api-diff abi-check tuning-diff struct-layout metal-diff metal-ops-diff backend-ops parity-cli cnv-diff fit-diff tools-diff parity-port parity-port-cpu node-diff cuda-ops-diff cuda-node-diff cuda-backend-diff cuda-parity-cli cuda-require-linux port oneshot chat ref ref-chat web ui validate xcframework xcframework-check perf perf-cpu

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

PORT_CLI := ./zig-out/bin/llama-cli
REF_CLI  := llama.cpp.zmake/zig-out/macos/bin/llama-cli

#
# Setup
#

# Clone the pinned llama.cpp reference checkout.
# The release the port is synced to, from build/upstream.zig.
UPSTREAM_TAG := $(shell sed -n 's/^pub const tag = "\(.*\)";$$/\1/p' build/upstream.zig)

clone:
	git clone https://github.com/ggml-org/llama.cpp.git
	cd llama.cpp; git checkout $(UPSTREAM_TAG)

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
# simulators. Everything lives in xcframework/.
xcframework:
	$(MAKE) -C xcframework

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

# The web UI bundle both llama-servers embed (-Dui; build/ui.zig): upstream's
# tools/ui, built with npm by llama.cpp.zmake's `make ui` into its .ui/dist,
# once. Delete that directory to rebuild it. `npm run build` is not
# reproducible, so both sides of a gate always embed this one bundle.
UI_DIST ?= llama.cpp.zmake/.ui/dist
ui:
	@test -f $(UI_DIST)/index.html || $(MAKE) -C llama.cpp.zmake ui

# The CUDA gates. Linux with nvcc and an NVIDIA
# GPU only: the Mac has neither, and Metal's gates cannot run on the CUDA
# machine. Each builds what its script's `Requires:` line names
# into its own prefix, so the port and upstream's host layer never share an
# archive; the prefixes are the ones the scripts read. See the header of
# each script.

#
# Run
#

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

# Our llama-server on $(MODEL), port $(PORT), with the web UI embedded (`ui`)
# and the default browser opened on it once /health answers 200 (503 while
# the model loads). The server runs in the foreground; Ctrl-C stops it. The
# waiter gives up after five minutes or as soon as the server has exited.
# `llama.cpp.zmake`'s `make web` is the same over upstream's server.
PORT ?= 8080
OPEN ?= $(if $(filter Darwin,$(shell uname -s)),open,xdg-open)

web: ui
	zig build tools --release=fast -Dui=$(UI_DIST)
	@( for i in $$(seq 1 600); do \
	     curl -sf -o /dev/null http://127.0.0.1:$(PORT)/health && { $(OPEN) http://127.0.0.1:$(PORT)/; exit 0; }; \
	     kill -0 $$$$ 2>/dev/null || exit 0; sleep 0.5; \
	   done ) & \
	exec ./zig-out/bin/llama-server -m $(MODEL) --port $(PORT) $(ARGS)

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

# Gates, oracle builds and the checks that run them: testing/Makefile.
# Included, not delegated, so `make ops-diff` and the rest run from here.
include testing/Makefile

# ---------------------------------------------
# Package
# ---------------------------------------------

# `make pkg` (DRY_RUN=1 stops before Apple and R2) and `make pkginstall` come
# from ../styleguide/Makefile.in: the CLI as a .pkg, built by zigpkg's zpkg
# from this project's recipe in zigpkg's store.
