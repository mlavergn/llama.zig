//! The builders chatterbox's gen and speaker-encoder graphs share. This
//! file's convention is `[C, T]`.
//!
//! # Provenance
//!
//! Ports `mtmd_graph_chatterbox_base`, declared in
//! `llama.cpp/tools/mtmd/models/models.h` and defined in
//! `llama.cpp/tools/mtmd/models/chatterbox-gen.cpp`, at v1.0.0
//! (`b1b770d90`). Each declaration names what it ports and the line it
//! began at.
//!
//! Upstream's class is a `mtmd_graph` subclass with three `const` members;
//! here they are free functions over a `graph.Graph`.

const std = @import("std");
const mtmd = @import("../module.zig");

/// Ports `cbx_linear` (chatterbox-gen.cpp:6 @b1b770d90), a member of
/// `mtmd_graph_chatterbox_base`: x `[C, T]`, y = W x + b with torch Linear
/// weights stored as `[in, out]`.
pub fn cbxLinear(g: *const mtmd.graph.Graph, w: mtmd.graph.T, b: mtmd.graph.T, x: mtmd.graph.T) mtmd.graph.T {
    const ctx0 = g.ctx0;
    var y = mtmd.impl.c.ggml_mul_mat(ctx0, w, x);
    if (b != null) {
        y = mtmd.impl.c.ggml_add(ctx0, y, b);
    }
    return y;
}

/// Ports `cbx_conv1d` (chatterbox-gen.cpp:15 @b1b770d90), a member of
/// `mtmd_graph_chatterbox_base`: conv1d over time with asymmetric zero
/// padding, x `[C, T]` to `[OC, T_out]`, kernel `[K, IC, OC]`.
pub fn cbxConv1d(g: *const mtmd.graph.Graph, k: mtmd.graph.T, b: mtmd.graph.T, x: mtmd.graph.T, stride: c_int, pad_l: c_int, pad_r: c_int) mtmd.graph.T {
    const ctx0 = g.ctx0;
    var xt = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, x)); // [T, C]
    if (pad_l > 0) {
        var z = mtmd.impl.c.ggml_new_tensor_2d(ctx0, mtmd.impl.c.GGML_TYPE_F32, pad_l, mtmd.graph.ne(xt, 1));
        z = mtmd.impl.c.ggml_scale(ctx0, z, 0.0);
        xt = mtmd.impl.c.ggml_concat(ctx0, z, xt, 0);
    }
    if (pad_r > 0) {
        var z = mtmd.impl.c.ggml_new_tensor_2d(ctx0, mtmd.impl.c.GGML_TYPE_F32, pad_r, mtmd.graph.ne(xt, 1));
        z = mtmd.impl.c.ggml_scale(ctx0, z, 0.0);
        xt = mtmd.impl.c.ggml_concat(ctx0, xt, z, 0);
    }
    var y = mtmd.impl.c.ggml_conv_1d(ctx0, k, xt, stride, 0, 1); // [T_out, OC]
    y = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, y)); // [OC, T_out]
    if (b != null) {
        y = mtmd.impl.c.ggml_add(ctx0, y, b);
    }
    return y;
}

/// Ports `cbx_conv1d_dil` (chatterbox-gen.cpp:37 @b1b770d90), a member of
/// `mtmd_graph_chatterbox_base`: x `[C, T]` through a symmetric-padded
/// dilated conv to `[OC, T]`.
pub fn cbxConv1dDil(g: *const mtmd.graph.Graph, k: mtmd.graph.T, b: mtmd.graph.T, x: mtmd.graph.T, pad: c_int, dil: c_int) mtmd.graph.T {
    const ctx0 = g.ctx0;
    const xt = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, x));
    var y = mtmd.impl.c.ggml_conv_1d(ctx0, k, xt, 1, pad, dil);
    y = mtmd.impl.c.ggml_cont(ctx0, mtmd.impl.c.ggml_transpose(ctx0, y));
    if (b != null) {
        y = mtmd.impl.c.ggml_add(ctx0, y, b);
    }
    return y;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
