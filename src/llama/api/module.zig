//! Barrel for the ported `llama.h` entry points.
//!
//! **Not a port.** It gathers the files that hold `llama.h`'s C API, each
//! a `pub` function with the C signature and none exported here:
//!
//! - `context.zig` -- the context, memory, decode and perf entry points of
//!   `llama.cpp/src/llama-context.cpp`
//! - `adapter.zig` -- the LoRA and control-vector entry points of
//!   `llama.cpp/src/llama-adapter.cpp` and `llama.cpp/src/llama-context.cpp`
//! - `state.zig` -- its state API
//! - `llama.zig` -- `llama.cpp/src/llama.cpp`: backend setup, capabilities,
//!   loading, chat templates and split names
//! - `model.zig` -- `llama.cpp/src/llama-model.cpp`
//! - `vocab.zig` -- `llama.cpp/src/llama-vocab.cpp`
//! - `impl.zig` -- `llama.cpp/src/llama-impl.cpp`: the logger's callback
//! - `batch.zig` -- `llama.cpp/src/llama-batch.cpp`
//! - `chat.zig` -- `llama.cpp/src/llama-chat.cpp`
//! - `cparams.zig` -- `llama.cpp/src/llama-cparams.cpp`
//! - `model_loader.zig` -- `llama.cpp/src/llama-model-loader.cpp`
//! - `quant.zig` -- `llama.cpp/src/llama-quant.cpp`'s two C exports, and
//!   `llama_model_save_to_file` from `llama.cpp/src/llama.cpp`
//! - `opt.zig` -- the training entry points of `llama.cpp/src/llama-context.cpp`

const std = @import("std");

pub const context = @import("context.zig");
pub const adapter = @import("adapter.zig");
pub const state = @import("state.zig");
pub const quant = @import("quant.zig");
pub const llama = @import("llama.zig");
pub const model = @import("model.zig");
pub const vocab = @import("vocab.zig");
pub const impl = @import("impl.zig");
pub const batch = @import("batch.zig");
pub const chat = @import("chat.zig");
pub const cparams = @import("cparams.zig");
pub const model_loader = @import("model_loader.zig");
pub const opt = @import("opt.zig");

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(context);
    std.testing.refAllDecls(adapter);
    std.testing.refAllDecls(state);
    std.testing.refAllDecls(quant);
    std.testing.refAllDecls(llama);
    std.testing.refAllDecls(model);
    std.testing.refAllDecls(vocab);
    std.testing.refAllDecls(impl);
    std.testing.refAllDecls(batch);
    std.testing.refAllDecls(chat);
    std.testing.refAllDecls(cparams);
    std.testing.refAllDecls(model_loader);
    std.testing.refAllDecls(opt);
}
