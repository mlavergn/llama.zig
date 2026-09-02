###############################################
#
# Makefile
#
###############################################

.DEFAULT_GOAL := build

.PHONY: build llama.cpp

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

# Diff the token stream of the Apple-clang and zig cc builds. Greedy sampling
# with a fixed seed makes the output a function of the arithmetic alone, so a
# difference here is a real difference in what the code computes.
# Requires both `make buildmacos` and `make buildmacos-zig`.
parity:
	./scripts/parity $(MODEL) "$(PROMPT)"

# Build the llama.cpp reference libraries (ggml + libllama) with the Zig
# toolchain. No CMake involved.
reference:
	zig build reference

# Load a model through the reference build and generate, to prove the Metal
# path works end to end. Override with: make smoke MODEL=... PROMPT=...
MODEL  ?= Qwen3.5-2B-Q4_K_M.gguf
PROMPT ?= The capital of France is
smoke:
	zig build smoke -- "$(MODEL)" "$(PROMPT)"

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