//! Barrel for the Vulkan host layer.
//!
//! **Not a port.** Everything it imports is; see each file. Built only with
//! `-Dvulkan-port` until the swap (PLAN.md, "Stage 7 — Vulkan"), and then
//! the files here replace upstream's `ggml-vulkan.cpp`, whose shaders travel
//! as the reference generator's SPIR-V (Decisions 55 and 59).

pub const common = @import("common.zig");
pub const instance = @import("instance.zig");
pub const backend = @import("backend.zig");
pub const device = @import("device.zig");
pub const pipeline = @import("pipeline.zig");
pub const shaders = @import("shaders.zig");
pub const push = @import("push.zig");
pub const buffer = @import("buffer.zig");
pub const submit = @import("submit.zig");
pub const graph = @import("graph.zig");
pub const context = @import("context.zig");
pub const perf = @import("perf.zig");
pub const iface = @import("iface.zig");
pub const nodes = @import("nodes.zig");
pub const ops = @import("ops.zig");
pub const ops_basic = @import("ops_basic.zig");
pub const ops_reduce = @import("ops_reduce.zig");
pub const ops_more = @import("ops_more.zig");
pub const ops_conv = @import("ops_conv.zig");
pub const ops_matmul = @import("ops_matmul.zig");
pub const ops_matmul_id = @import("ops_matmul_id.zig");
pub const ops_flash_attn = @import("ops_flash_attn.zig");
pub const op_pipeline = @import("op_pipeline.zig");
pub const supports = @import("supports.zig");

comptime {
    // `backend.zig` exports `ggml-vulkan.h`.
    _ = backend;
}
