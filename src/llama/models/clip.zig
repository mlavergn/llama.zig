//! CLIP: not an architecture llama.cpp runs. The class exists so that
//! `llama-quantize` can open a multimodal projector's GGUF; loading one as
//! a model stops in the generic loader, and each override below aborts if
//! anything reaches it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/models/clip.cpp` at v0.3.0 (`c1d0e7a00`):
//! `llama_model_clip`'s `load_arch_hparams`, `load_arch_tensors` and
//! `build_arch_graph`, all three stubs.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");

const Exception = impl.Exception;
const Thrown = impl.Thrown;
const Model = model_mod.Model;

/// Ports `llama_model_clip`'s `load_arch_hparams` (src/models/clip.cpp:6 @c1d0e7a00).
pub fn loadArchHparams(_: *Model, _: *const loader.Loader, _: *Exception) Thrown!void {
    impl.abort("models/clip.cpp", 7, "CLIP is a quant-only stub; load_arch_hparams should not be called");
}

/// Ports `llama_model_clip`'s `load_arch_tensors` (src/models/clip.cpp:11 @c1d0e7a00).
pub fn loadArchTensors(_: *Model, _: *loader.Loader, _: *Exception) Thrown!void {
    impl.abort("models/clip.cpp", 12, "CLIP is a quant-only stub; load_arch_tensors should not be called");
}

/// Ports `llama_model_clip`'s `build_arch_graph` (src/models/clip.cpp:16 @c1d0e7a00).
pub fn buildArchGraph(_: *const Model, _: *const graph.Builder, _: *const graph.Params) void {
    impl.abort("models/clip.cpp", 17, "CLIP has no inference graph via llama_model dispatch; runtime lives in tools/mtmd/clip.cpp");
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
