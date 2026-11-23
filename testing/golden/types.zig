//! Golden checksums for `src/ggml/types.zig`'s traits and name tables.
//!
//! **Not a port.** Lifted out of the ported file so the release tree carries
//! no oracle data.

const std = @import("std");
const ggml = @import("ggml");
const mod = @import("module.zig");

// Golden checksums of the C tables, captured from a C-only build before any
// of ggml.c was ported.
//
// These cannot be direct comparisons against `c.ggml_get_type_traits` or
// `c.ggml_op_name`: this file exports those very symbols, so such a test would
// resolve back to the code under test and pass no matter how wrong the table
// became. That is not hypothetical -- it happened, and corrupting a block size
// left the comparison green.
//
// Regenerate only from a C build: `testing/scripts/types-golden` prints the traits and
// four name checksums from the stock ggml `llama.cpp.zmake` builds.
const golden_traits: u64 = 0xbbdffe39ac19e897;
const golden_op_name: u64 = 0x0fc0e52932954250;
const golden_op_symbol: u64 = 0x889b46b6e2bd5c8e;
const golden_unary_name: u64 = 0x78d971e2f43bd703;
const golden_glu_name: u64 = 0xbcc1426328c349eb;

/// Folds a C string into the checksum, distinguishing absent from empty.
fn mixStr(h_in: u64, s: ?[*:0]const u8) u64 {
    const str = s orelse return mod.impl.mix(h_in, 0xdeadbeef);
    var h = h_in;
    var i: usize = 0;
    while (str[i] != 0) : (i += 1) h = mod.impl.mix(h, str[i]);
    return mod.impl.mix(h, 0);
}

test "traits table matches the C checksum" {
    var h: u64 = 0;
    var t: usize = 0;
    while (t < ggml.impl.ggml_c.GGML_TYPE_COUNT) : (t += 1) {
        const tr = &ggml.types.type_traits[t];
        h = mixStr(h, @ptrCast(tr.type_name));
        h = mod.impl.mix(h, @bitCast(tr.blck_size));
        h = mod.impl.mix(h, tr.type_size);
        h = mod.impl.mix(h, if (tr.is_quantized) 1 else 0);
        h = mod.impl.mix(h, if (tr.to_float != null) 1 else 0);
        h = mod.impl.mix(h, if (tr.from_float_ref != null) 1 else 0);
    }
    try std.testing.expectEqual(golden_traits, h);
}

test "op name tables match the C checksums" {
    var hn: u64 = 0;
    var hs: u64 = 0;
    for (ggml.types.op_name, ggml.types.op_symbol) |n, sym| {
        hn = mixStr(hn, n);
        hs = mixStr(hs, sym);
    }
    try std.testing.expectEqual(golden_op_name, hn);
    try std.testing.expectEqual(golden_op_symbol, hs);

    var hu: u64 = 0;
    for (ggml.types.unary_op_name) |n| hu = mixStr(hu, n);
    try std.testing.expectEqual(golden_unary_name, hu);

    var hg: u64 = 0;
    for (ggml.types.glu_op_name) |n| hg = mixStr(hg, n);
    try std.testing.expectEqual(golden_glu_name, hg);

    // The tables must also be exactly as long as the enums they index.
    try std.testing.expectEqual(@as(usize, ggml.impl.ggml_c.GGML_OP_COUNT), ggml.types.op_name.len);
    try std.testing.expectEqual(@as(usize, ggml.impl.ggml_c.GGML_OP_COUNT), ggml.types.op_symbol.len);
    try std.testing.expectEqual(@as(usize, ggml.impl.ggml_c.GGML_UNARY_OP_COUNT), ggml.types.unary_op_name.len);
    try std.testing.expectEqual(@as(usize, ggml.impl.ggml_c.GGML_GLU_OP_COUNT), ggml.types.glu_op_name.len);
}
