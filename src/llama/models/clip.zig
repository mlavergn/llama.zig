//! CLIP: not an architecture llama.cpp runs. The class exists so that
//! `llama-quantize` can open a multimodal projector's GGUF; loading one as
//! a model stops in the generic loader, and each override below aborts if
//! anything reaches it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/clip.cpp` at v1.0.0 (`b1b770d90`):
//! `llama_model_clip`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, all three stubs.

const std = @import("std");
const llama = @import("../module.zig");

/// Ports `llama_model_clip`'s `load_arch_hparams` (src/models/clip.cpp:6 @b1b770d90).
pub fn loadArchHparams(_: *llama.model.Model, _: *const llama.loader.Loader, _: *llama.impl.Exception) llama.impl.Thrown!void {
    llama.impl.abort("models/clip.cpp", 7, "CLIP is a quant-only stub; load_arch_hparams should not be called");
}

/// Ports `llama_model_clip`'s `load_arch_tensors` (src/models/clip.cpp:11 @b1b770d90).
pub fn loadArchTensors(_: *llama.model.Model, _: *llama.loader.Loader, _: *llama.impl.Exception) llama.impl.Thrown!void {
    llama.impl.abort("models/clip.cpp", 12, "CLIP is a quant-only stub; load_arch_tensors should not be called");
}

/// Ports `llama_model_clip`'s `build_arch_graph` (src/models/clip.cpp:16 @b1b770d90).
pub fn buildArchGraph(_: *const llama.model.Model, _: *const llama.graph.Builder, _: *const llama.graph.Params) void {
    llama.impl.abort("models/clip.cpp", 17, "CLIP has no inference graph via llama_model dispatch; runtime lives in tools/mtmd");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
