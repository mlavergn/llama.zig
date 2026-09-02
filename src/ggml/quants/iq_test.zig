//! Golden checks for the i-quant formats.
//!
//! # Provenance
//!
//! **Not a port.** Test scaffolding, in its own file because the i-quants are
//! split across `iq1.zig`, `iq2.zig`, `iq3.zig`, `iq4.zig` and
//! `iq_dequant.zig`, and checking them together keeps one table of what each
//! format supports rather than five.
//!
//! # Why these are checked as a group
//!
//! Each i-quant has a different combination of entry points -- some have a
//! reference quantizer, some only the chunk form, some require an importance
//! matrix -- and the golden record already records which. Driving them from
//! that record means a format cannot be silently skipped, which is exactly
//! what happened when these were first written: the i-quants had no golden
//! test at all, and a pointer-direction bug in the neighbour lookup reached
//! `test-backend-ops` before anything caught it.

const std = @import("std");
const impl = @import("../impl.zig");
const blocks = @import("blocks.zig");
const quantize = @import("../quantize.zig");
const dq = @import("iq_dequant.zig");
const iq1 = @import("iq1.zig");
const iq2 = @import("iq2.zig");
const iq3 = @import("iq3.zig");
const iq4 = @import("iq4.zig");
const t = @import("testing.zig");
const c = impl.c;

const ChunkFn = *const fn ([*c]const f32, ?*anyopaque, i64, i64, [*c]const f32) callconv(.c) usize;
const DeqFn = *const fn (?*const anyopaque, [*c]f32, i64) callconv(.c) void;

/// One format's entry points, as the C exposes them.
const Case = struct {
    name: []const u8,
    /// The ggml type, so the codebook can be initialised.
    type: c.enum_ggml_type,
    /// `quantize_row_*_ref`, wrapped to a common signature. Null when the C
    /// has none -- the imatrix-only formats.
    ref: ?*const fn ([*c]const f32, ?*anyopaque, i64) void = null,
    chunk: ChunkFn,
    dequant: DeqFn,
};

fn wrapRef(comptime f: anytype) *const fn ([*c]const f32, ?*anyopaque, i64) void {
    return struct {
        fn g(x: [*c]const f32, y: ?*anyopaque, k: i64) void {
            f(x, @ptrCast(@alignCast(y)), k);
        }
    }.g;
}

fn wrapDeq(comptime f: anytype) DeqFn {
    return @ptrCast(&f);
}

const cases = [_]Case{
    .{
        .name = "IQ2_XXS",
        .type = c.GGML_TYPE_IQ2_XXS,
        .chunk = iq2.quantize_iq2_xxs,
        .dequant = wrapDeq(dq.dequantize_row_iq2_xxs),
    },
    .{
        .name = "IQ2_XS",
        .type = c.GGML_TYPE_IQ2_XS,
        .chunk = iq2.quantize_iq2_xs,
        .dequant = wrapDeq(dq.dequantize_row_iq2_xs),
    },
    .{
        .name = "IQ2_S",
        .type = c.GGML_TYPE_IQ2_S,
        .ref = wrapRef(iq2.quantize_row_iq2_s_ref),
        .chunk = iq2.quantize_iq2_s,
        .dequant = wrapDeq(dq.dequantize_row_iq2_s),
    },
    .{
        .name = "IQ3_XXS",
        .type = c.GGML_TYPE_IQ3_XXS,
        .ref = wrapRef(iq3.quantize_row_iq3_xxs_ref),
        .chunk = iq3.quantize_iq3_xxs,
        .dequant = wrapDeq(dq.dequantize_row_iq3_xxs),
    },
    .{
        .name = "IQ3_S",
        .type = c.GGML_TYPE_IQ3_S,
        .ref = wrapRef(iq3.quantize_row_iq3_s_ref),
        .chunk = iq3.quantize_iq3_s,
        .dequant = wrapDeq(dq.dequantize_row_iq3_s),
    },
    .{
        .name = "IQ1_S",
        .type = c.GGML_TYPE_IQ1_S,
        .chunk = iq1.quantize_iq1_s,
        .dequant = wrapDeq(dq.dequantize_row_iq1_s),
    },
    .{
        .name = "IQ1_M",
        .type = c.GGML_TYPE_IQ1_M,
        .chunk = iq1.quantize_iq1_m,
        .dequant = wrapDeq(dq.dequantize_row_iq1_m),
    },
};

test "every i-quant matches the C on every pattern" {
    for (cases) |case| {
        // The codebooks are built lazily. Going through `ggml_quantize_init`
        // rather than the per-family initialiser is the point: it dispatches
        // on type, where calling `iq2xs_init_impl` directly aborts for the
        // 3-bit formats.
        quantize.ggml_quantize_init(case.type);
        for (t.all_patterns) |pattern| {
            // **Skipped where the C's own answer is unspecified.**
            //
            // The 1-bit formats solve their split exactly: sort the block,
            // then try every pair of split points. When a block's weights are
            // all *identical*, two different splits score the same in exact
            // arithmetic, and which wins comes down to the rounding of
            // `sumx`, accumulated in sorted order.
            //
            // C does not specify what order `qsort` leaves equal elements in.
            // Measured: macOS libc returns `31 1 2 ... 30 0` for 32 equal
            // elements -- it swaps the ends. glibc would differ, and so could
            // a future Apple release. We sort deterministically instead (see
            // `iq1.zig`), so these cannot be matched and should not be.
            //
            // `constant` is all-identical by construction. `outlier` is too
            // for `iq1_m`, whose blocks are 16 wide: only the block holding
            // the outlier has any variation.
            //
            // Real model weights do not produce exact ties, so this does not
            // reach a real quantization -- but it is a divergence and it is
            // recorded rather than hidden.
            const ties_unspecified = (std.mem.eql(u8, case.name, "IQ1_S") or
                std.mem.eql(u8, case.name, "IQ1_M")) and
                (pattern == .constant or pattern == .outlier);
            if (ties_unspecified) continue;

            const g = t.find(case.name, pattern);

            var src: [t.n_elem]f32 = undefined;
            var imatrix: [t.n_per_row]f32 = undefined;
            t.fillSrc(pattern, &src);
            t.fillImatrix(&imatrix);

            var buf: [t.n_elem * 4]u8 align(16) = undefined;
            // A *separate* buffer for the dequantizer's input. Slicing `buf`
            // would have worked right up until the chunk checks below
            // overwrite it, which is exactly what happened the first time.
            var deq_src: [t.n_elem * 4]u8 align(16) = undefined;
            var deq_len: usize = 0;

            if (case.ref) |ref| {
                @memset(&buf, 0);
                ref(&src, &buf, @intCast(t.n_elem));
                const used = g.row_size * t.n_rows;
                std.testing.expectEqual(g.ref.?, t.fnv(buf[0..used])) catch |e| {
                    std.debug.print("{s}: ref differs on '{s}'\n", .{ case.name, pattern.name() });
                    return e;
                };
                @memcpy(deq_src[0..used], buf[0..used]);
                deq_len = used;
            }

            if (g.chunk) |want| {
                @memset(&buf, 0);
                const n = case.chunk(&src, &buf, t.n_rows, t.n_per_row, null);
                std.testing.expectEqual(want, t.fnv(buf[0..n])) catch |e| {
                    std.debug.print("{s}: chunk differs on '{s}'\n", .{ case.name, pattern.name() });
                    return e;
                };
            }

            if (g.chunk_imatrix) |want| {
                @memset(&buf, 0);
                const n = case.chunk(&src, &buf, t.n_rows, t.n_per_row, &imatrix);
                std.testing.expectEqual(want, t.fnv(buf[0..n])) catch |e| {
                    std.debug.print("{s}: chunk+imatrix differs on '{s}'\n", .{ case.name, pattern.name() });
                    return e;
                };
                if (case.ref == null) {
                    @memcpy(deq_src[0..n], buf[0..n]);
                    deq_len = n;
                }
            }

            if (g.deq) |want| {
                var out: [t.n_elem]f32 = undefined;
                @memset(&out, 0);
                std.debug.assert(deq_len > 0);
                case.dequant(@ptrCast(&deq_src), &out, @intCast(t.n_elem));
                std.testing.expectEqual(want, t.fnv(std.mem.sliceAsBytes(out[0..]))) catch |e| {
                    std.debug.print("{s}: dequantize differs on '{s}'\n", .{ case.name, pattern.name() });
                    return e;
                };
            }
        }
    }
}
