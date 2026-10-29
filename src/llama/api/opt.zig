//! `llama.h`'s training entry points.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-context.cpp` (v0.4.1, `b29c606e2`): the
//! three functions under its `// training` banner in the C API. The work is
//! `context_opt.zig`'s.
//!
//! Each function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call these beside the C++ that defines the same
//! names, and `exports.zig` exports them once libllama swaps.

const std = @import("std");
const impl = @import("../impl.zig");
const context_opt = @import("../context_opt.zig");
const api_context = @import("context.zig");

const c = impl.c;

/// Ports `llama_opt_param_filter_all` (llama-context.cpp:4298 @b29c606e2).
///
/// Parameters:
/// - `tensor`: unused.
/// - `userdata`: unused.
///
/// Return: true, for every tensor.
pub fn llama_opt_param_filter_all(tensor: [*c]const c.struct_ggml_tensor, userdata: ?*anyopaque) callconv(.c) bool {
    _ = tensor;
    _ = userdata;
    return true;
}

/// Ports `llama_opt_init` (llama-context.cpp:4304 @b29c606e2).
///
/// Parameters:
/// - `ctx`: the context; it must not have been initialised for training.
/// - `model`: the model to train.
/// - `lopt_params`: the training parameters.
///
/// Return: nothing.
pub fn llama_opt_init(ctx: ?*c.struct_llama_context, model: ?*c.struct_llama_model, lopt_params: c.struct_llama_opt_params) callconv(.c) void {
    context_opt.optInit(api_context.ctxOf(ctx), api_context.modelOf(model), lopt_params);
}

/// Ports `llama_opt_epoch` (llama-context.cpp:4308 @b29c606e2).
///
/// Parameters:
/// - `ctx`: the context, after `llama_opt_init`.
/// - `dataset`: the data, `n_ctx` tokens per datapoint.
/// - `result_train`: accumulates the training datapoints.
/// - `result_eval`: accumulates the evaluation datapoints.
/// - `idata_split`: datapoints before this train, the rest evaluate.
/// - `callback_train`: called after each training evaluation; may be null.
/// - `callback_eval`: called after each evaluation-only one; may be null.
///
/// Return: nothing.
pub fn llama_opt_epoch(
    ctx: ?*c.struct_llama_context,
    dataset: c.ggml_opt_dataset_t,
    result_train: c.ggml_opt_result_t,
    result_eval: c.ggml_opt_result_t,
    idata_split: i64,
    callback_train: c.ggml_opt_epoch_callback,
    callback_eval: c.ggml_opt_epoch_callback,
) callconv(.c) void {
    context_opt.optEpoch(api_context.ctxOf(ctx), dataset, result_train, result_eval, idata_split, callback_train, callback_eval);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
