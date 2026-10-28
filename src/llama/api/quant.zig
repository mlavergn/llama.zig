//! `llama.h`'s quantization and model-saving entry points.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-quant.cpp` -- `llama_model_quantize_default_params`
//!   and `llama_model_quantize`, its two C exports
//! - `llama.cpp/src/llama.cpp` -- `llama_model_save_to_file`, which lives
//!   there rather than in `llama-model-saver.cpp`
//!
//! All at v0.4.0 (`5266f24da`). The work is `quant.zig`'s and
//! `model_saver.zig`'s.
//!
//! Each function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call these beside the C++ that defines the same
//! names, and `exports.zig` exports them once libllama swaps.

const std = @import("std");
const impl = @import("../impl.zig");
const quant = @import("../quant.zig");
const model_saver = @import("../model_saver.zig");
const ctx_api = @import("context.zig");

const c = impl.c;

fn span(p: [*c]const u8) [:0]const u8 {
    const z: [*:0]const u8 = @ptrCast(p);
    return std.mem.span(z);
}

/// Ports `llama_model_quantize_default_params` (llama-quant.cpp:1365 @5266f24da).
pub fn llama_model_quantize_default_params() callconv(.c) c.llama_model_quantize_params {
    return quant.defaultParams();
}

/// Ports `llama_model_quantize` (llama-quant.cpp:1387 @5266f24da): every
/// exception is caught, logged and reported as 1.
pub fn llama_model_quantize(fname_inp: [*c]const u8, fname_out: [*c]const u8, params: [*c]const c.llama_model_quantize_params) callconv(.c) u32 {
    var exc: impl.Exception = .{};
    quant.quantize(span(fname_inp), span(fname_out), params, &exc) catch {
        impl.logError("%s: failed to quantize: %s\n", .{ "llama_model_quantize", exc.whatZ() });
        return 1;
    };
    return 0;
}

/// Ports `llama_model_save_to_file` (src/llama.cpp:498 @5266f24da). Nothing
/// catches what the saver throws, so it terminates.
pub fn llama_model_save_to_file(model: ?*const c.struct_llama_model, path_model: [*c]const u8) callconv(.c) void {
    var ms = model_saver.Saver.init(ctx_api.modelOf(model));
    defer ms.deinit();
    var exc: impl.Exception = .{};
    ms.addKvFromModel(&exc) catch exc.terminate("llama.cpp", 500);
    ms.addTensorsFromModel();
    ms.save(@ptrCast(path_model));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
