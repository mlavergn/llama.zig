//! Golden checksums for `src/ggml/impl.zig`'s fp16/bf16 conversions.
//!
//! **Not a port.** Lifted out of the ported file so the release tree carries
//! no oracle data.

const std = @import("std");
const ggml = @import("ggml");

// Golden values, captured from the C implementations in `ggml.c` before any
// of `ggml.c` was ported. They are checksums rather than direct comparisons on
// purpose: once `runtime.zig` exports `ggml_fp16_to_fp32` and friends, a test
// that called those symbols would be comparing this code against itself and
// would pass no matter how wrong it became.
//
// Regenerate only from a C build, never from this code. The procedure is in
// the commit that introduced them.
const golden_fp16_to_fp32: u64 = 0x5c17cf6b1ad614ad;
const golden_bf16_to_fp32: u64 = 0x4b65f6efc80cd5ba;
const golden_fp32_to_fp16: u64 = 0x3081ce06bf36ae60;
const golden_fp32_to_bf16: u64 = 0xd60a7155d0aaade5;

/// The mixing step the golden checksums were built with.
///
/// Order-dependent and avalanching, so a single wrong conversion anywhere in
/// the swept domain changes the result.
pub fn mix(h: u64, v: u64) u64 {
    return h ^ (v +% 0x9e3779b97f4a7c15 +% (h << 6) +% (h >> 2));
}

/// The fp32 sweep used by the checksums: a prime stride, so every exponent and
/// both signs are visited while the low mantissa bits keep changing.
const sweep_stride: u32 = 9973;

test "fp16 to fp32 matches the C checksum over all 65536 inputs" {
    var h: u64 = 0;
    var bits: u32 = 0;
    while (bits <= std.math.maxInt(u16)) : (bits += 1) {
        const f = ggml.impl.fp16ToFp32(@intCast(bits));
        h = mix(h, @as(u32, @bitCast(f)));
    }
    try std.testing.expectEqual(golden_fp16_to_fp32, h);
}

test "bf16 to fp32 matches the C checksum over all 65536 inputs" {
    var h: u64 = 0;
    var bits: u32 = 0;
    while (bits <= std.math.maxInt(u16)) : (bits += 1) {
        const f = ggml.impl.bf16ToFp32(@intCast(bits));
        h = mix(h, @as(u32, @bitCast(f)));
    }
    try std.testing.expectEqual(golden_bf16_to_fp32, h);
}

test "fp32 narrowing matches the C checksums across the exponent range" {
    var h16: u64 = 0;
    var hbf: u64 = 0;
    var bits: u32 = 0;
    while (true) {
        const f: f32 = @bitCast(bits);
        h16 = mix(h16, ggml.impl.fp32ToFp16(f));
        hbf = mix(hbf, ggml.impl.fp32ToBf16(f));
        if (bits > std.math.maxInt(u32) - sweep_stride) break;
        bits += sweep_stride;
    }
    try std.testing.expectEqual(golden_fp32_to_fp16, h16);
    try std.testing.expectEqual(golden_fp32_to_bf16, hbf);
}
