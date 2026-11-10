//! The builders chatterbox's gen and speaker-encoder graphs share. This
//! file's convention is `[C, T]`.
//!
//! # Provenance
//!
//! Ports `clip_graph_chatterbox_base`, declared in
//! `llama.cpp/tools/mtmd/models/models.h` and defined in
//! `llama.cpp/tools/mtmd/models/chatterbox-gen.cpp`, at v0.9.0
//! (`cd959bae9`). Each declaration names what it ports and the line it
//! began at.
//!
//! Upstream's class is a `clip_graph` subclass with three `const` members;
//! here they are free functions over a `graph.Graph`.

const std = @import("std");
const impl = @import("../impl.zig");
const graph = @import("../graph.zig");
const c = impl.c;
const T = graph.T;
const ne = graph.ne;

/// Ports `cbx_linear` (chatterbox-gen.cpp:7 @cd959bae9), a member of
/// `clip_graph_chatterbox_base`: x `[C, T]`, y = W x + b with torch Linear
/// weights stored as `[in, out]`.
pub fn cbxLinear(g: *const graph.Graph, w: T, b: T, x: T) T {
    const ctx0 = g.ctx0;
    var y = c.ggml_mul_mat(ctx0, w, x);
    if (b != null) {
        y = c.ggml_add(ctx0, y, b);
    }
    return y;
}

/// Ports `cbx_conv1d` (chatterbox-gen.cpp:16 @cd959bae9), a member of
/// `clip_graph_chatterbox_base`: conv1d over time with asymmetric zero
/// padding, x `[C, T]` to `[OC, T_out]`, kernel `[K, IC, OC]`.
pub fn cbxConv1d(g: *const graph.Graph, k: T, b: T, x: T, stride: c_int, pad_l: c_int, pad_r: c_int) T {
    const ctx0 = g.ctx0;
    var xt = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, x)); // [T, C]
    if (pad_l > 0) {
        var z = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, pad_l, ne(xt, 1));
        z = c.ggml_scale(ctx0, z, 0.0);
        xt = c.ggml_concat(ctx0, z, xt, 0);
    }
    if (pad_r > 0) {
        var z = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, pad_r, ne(xt, 1));
        z = c.ggml_scale(ctx0, z, 0.0);
        xt = c.ggml_concat(ctx0, xt, z, 0);
    }
    var y = c.ggml_conv_1d(ctx0, k, xt, stride, 0, 1); // [T_out, OC]
    y = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, y)); // [OC, T_out]
    if (b != null) {
        y = c.ggml_add(ctx0, y, b);
    }
    return y;
}

/// Ports `cbx_conv1d_dil` (chatterbox-gen.cpp:38 @cd959bae9), a member of
/// `clip_graph_chatterbox_base`: x `[C, T]` through a symmetric-padded
/// dilated conv to `[OC, T]`.
pub fn cbxConv1dDil(g: *const graph.Graph, k: T, b: T, x: T, pad: c_int, dil: c_int) T {
    const ctx0 = g.ctx0;
    const xt = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, x));
    var y = c.ggml_conv_1d(ctx0, k, xt, 1, pad, dil);
    y = c.ggml_cont(ctx0, c.ggml_transpose(ctx0, y));
    if (b != null) {
        y = c.ggml_add(ctx0, y, b);
    }
    return y;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
