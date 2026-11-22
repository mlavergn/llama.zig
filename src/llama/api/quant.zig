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
//! All at v1.0.0 (`b1b770d90`). The work is `quant.zig`'s and
//! `model_saver.zig`'s.
//!
//! Each function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call these beside the C++ that defines the same
//! names, and `exports.zig` exports them once libllama swaps.

const std = @import("std");
const llama = @import("../module.zig");

fn span(p: [*c]const u8) [:0]const u8 {
    const z: [*:0]const u8 = @ptrCast(p);
    return std.mem.span(z);
}

/// Ports `llama_model_quantize_default_params` (llama-quant.cpp:1468 @b1b770d90).
pub fn llama_model_quantize_default_params() callconv(.c) llama.impl.c.llama_model_quantize_params {
    return llama.quant.defaultParams();
}

/// Ports `llama_model_quantize` (llama-quant.cpp:1490 @b1b770d90): every
/// exception is caught, logged and reported as 1.
pub fn llama_model_quantize(fname_inp: [*c]const u8, fname_out: [*c]const u8, params: [*c]const llama.impl.c.llama_model_quantize_params) callconv(.c) u32 {
    var exc: llama.impl.Exception = .{};
    llama.quant.quantize(span(fname_inp), span(fname_out), params, &exc) catch {
        llama.impl.logError("%s: failed to quantize: %s\n", .{ "llama_model_quantize", exc.whatZ() });
        return 1;
    };
    return 0;
}

/// Ports `llama_model_save_to_file` (src/llama.cpp:492 @b1b770d90). Nothing
/// catches what the saver throws, so it terminates.
pub fn llama_model_save_to_file(model: ?*const llama.impl.c.struct_llama_model, path_model: [*c]const u8) callconv(.c) void {
    var ms = llama.model_saver.Saver.init(llama.api.context.modelOf(model));
    defer ms.deinit();
    var exc: llama.impl.Exception = .{};
    ms.addKvFromModel(&exc) catch exc.terminate("llama.cpp", 494);
    ms.addTensorsFromModel();
    ms.save(@ptrCast(path_model));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
