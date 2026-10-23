/* The Vulkan C API for the Zig host layer (src/ggml/vulkan/), translated by
 * build/llamacpp.zig's `addVulkanPort` into the `vulkan_c` module. Not a
 * port of anything.
 *
 * `ggml-vulkan.cpp` is written against `vulkan.hpp`, the C++ binding; that is
 * a header-only wrapper over this C API, so the port calls the C API
 * directly. No ggml header is included here: a second translation of one
 * would produce incompatible `ggml_tensor` types (CLAUDE.md, `impl.zig`). */
#define VK_NO_PROTOTYPES 1
#include <vulkan/vulkan_core.h>

/* SPIR-V's opcodes and enumerants, for the three passes `ggml-vulkan.cpp`
 * runs over a module before creating it (`pipeline.zig`). The C++ includes
 * `spirv.hpp`; this is the same header's C spelling. */
#include <spirv/unified1/spirv.h>
