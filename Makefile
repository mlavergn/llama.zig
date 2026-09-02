###############################################
#
# Makefile
#
###############################################

.DEFAULT_GOAL := build

.PHONY: build llama.cpp

clone:
	git clone https://github.com/ggerganov/llama.cpp.git
	cd llama.cpp; git checkout v0.3.0	

clean:
	rm -r build

# com.apple.developer.kernel.increased-memory-limit
buildios:
	-mkdir build
	cd build; cmake ../llama.cpp -G Xcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=18.0 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_ACCELERATE=ON -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO
	cd build; make 

buildmacos:
	-mkdir build
	cd build; cmake ../llama.cpp -G Xcode -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_OSX_SYSROOT=macosx -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_ACCELERATE=ON -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO
	cd build; make 

cli:
	./build/bin/llama-cli -m models/Qwen2.5-72B-Instruct-Q4_K_M.gguf -ngl 99 -p "You are an assistant. How are you?"

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