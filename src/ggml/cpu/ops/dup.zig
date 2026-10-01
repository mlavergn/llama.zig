//! The `dup` / `cpy` / `cont` family: copying a tensor, with or without a
//! change of type or layout.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/ops.cpp` at v0.3.0 (`c1d0e7a00`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # The C++ templates
//!
//! `ggml_compute_forward_dup_flt` is `template<typename src_t, typename
//! dst_t>` and `ggml_compute_forward_dup_to_q` is `template<typename src_t>`.
//! Both become `comptime` parameters, and the `if constexpr
//! (std::is_same_v<dst_t, src_t>)` arms become `if (SrcT == DstT)` on
//! comptime-known types, which Zig folds the same way.
//!
//! # Loop index names
//!
//! The C uses `i00`, `i01`, `i02`, `i03`, `i10`, `i11`, `i12`, `i13` as loop
//! indices, and Zig reserves `i<N>` as integer type names. They are renamed
//! `j00`, `j01`, … digit for digit, the convention `cpu/mulmat.zig` already
//! uses, so the index arithmetic still reads against the C line by line. Do
//! not renumber them.

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");

const c = impl.c;
const Tensor = common.Tensor;
const ComputeParams = common.ComputeParams;

/// Ports `ggml_compute_forward_dup_same_cont` (ops.cpp:17 @c1d0e7a00).
fn dupSameCont(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(c.ggml_nelements(dst) == c.ggml_nelements(src0), "ggml_nelements(dst) == ggml_nelements(src0)");
    impl.assert(c.ggml_is_contiguous(dst) and c.ggml_is_contiguous(src0), "ggml_is_contiguous(dst) && ggml_is_contiguous(src0)");
    impl.assert(src0.type == dst.type, "src0->type == dst->type");

    const nb0 = c.ggml_type_size(src0.type);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    // parallelize by blocks
    const nk: i64 = @intCast(@divTrunc(c.ggml_nelements(src0), c.ggml_blck_size(src0.type)));
    const dr = @divTrunc(nk + nth - 1, nth);
    const k0 = dr * ith;
    const k1 = @min(k0 + dr, nk);

    if (k0 < k1) {
        const d: [*]u8 = @ptrCast(dst.data.?);
        const s: [*]const u8 = @ptrCast(src0.data.?);
        const off = @as(usize, @intCast(k0)) * nb0;
        const len = @as(usize, @intCast(k1 - k0)) * nb0;
        @memcpy(d[off..][0..len], s[off..][0..len]);
    }
}

/// Ports `ggml_compute_forward_dup_flt` (ops.cpp:47 @c1d0e7a00), the
/// `template<typename src_t, typename dst_t>` form.
fn dupFlt(comptime SrcT: type, comptime DstT: type, params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(c.ggml_nelements(dst) == c.ggml_nelements(src0), "ggml_nelements(dst) == ggml_nelements(src0)");
    impl.assert(!c.ggml_is_quantized(src0.type) and !c.ggml_is_quantized(dst.type), "!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type)");

    const l = common.UnaryLocals.of(src0, dst);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    // parallelize by rows
    const nr = l.ne01;
    const dr = @divTrunc(nr + nth - 1, nth);
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const sdata: [*]const u8 = @ptrCast(src0.data.?);
    const ddata: [*]u8 = @ptrCast(dst.data.?);

    // case: type & row size equal
    if (src0.type == dst.type and l.ne00 == l.ne0 and
        l.nb00 == c.ggml_type_size(src0.type) and l.nb0 == c.ggml_type_size(dst.type))
    {
        // copy by rows
        const rs = @as(usize, @intCast(l.ne00)) * l.nb00;
        var j03: i64 = 0;
        while (j03 < l.ne03) : (j03 += 1) {
            var j02: i64 = 0;
            while (j02 < l.ne02) : (j02 += 1) {
                var j01: i64 = ir0;
                while (j01 < ir1) : (j01 += 1) {
                    const doff = @as(usize, @intCast(j01)) * l.nb1 + @as(usize, @intCast(j02)) * l.nb2 + @as(usize, @intCast(j03)) * l.nb3;
                    const soff = @as(usize, @intCast(j01)) * l.nb01 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03;
                    @memcpy(ddata[doff..][0..rs], sdata[soff..][0..rs]);
                }
            }
        }
        return;
    }

    // case: dst tensor is contiguous
    if (c.ggml_is_contiguous(dst)) {
        if (l.nb00 == @sizeOf(SrcT)) {
            if (SrcT == DstT) {
                // same type
                var id: usize = 0;
                const rs = @as(usize, @intCast(l.ne00)) * l.nb00;
                var j03: i64 = 0;
                while (j03 < l.ne03) : (j03 += 1) {
                    var j02: i64 = 0;
                    while (j02 < l.ne02) : (j02 += 1) {
                        id += rs * @as(usize, @intCast(ir0));
                        var j01: i64 = ir0;
                        while (j01 < ir1) : (j01 += 1) {
                            const soff = @as(usize, @intCast(j01)) * l.nb01 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03;
                            @memcpy(ddata[id..][0..rs], sdata[soff..][0..rs]);
                            id += rs;
                        }
                        id += rs * @as(usize, @intCast(l.ne01 - ir1));
                    }
                }
            } else {
                // casting between non-quantized types
                var id: usize = 0;
                const dst_ptr: [*]DstT = @ptrCast(@alignCast(dst.data.?));
                var j03: i64 = 0;
                while (j03 < l.ne03) : (j03 += 1) {
                    var j02: i64 = 0;
                    while (j02 < l.ne02) : (j02 += 1) {
                        id += @as(usize, @intCast(l.ne00 * ir0));
                        var j01: i64 = ir0;
                        while (j01 < ir1) : (j01 += 1) {
                            const soff = @as(usize, @intCast(j01)) * l.nb01 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03;
                            const src0_ptr: [*]const SrcT = @ptrCast(@alignCast(sdata + soff));
                            var j00: i64 = 0;
                            while (j00 < l.ne00) : (j00 += 1) {
                                const tmp = common.toF32(SrcT, src0_ptr[@intCast(j00)]);
                                dst_ptr[id] = common.fromF32(DstT, tmp);
                                id += 1;
                            }
                        }
                        id += @as(usize, @intCast(l.ne00 * (l.ne01 - ir1)));
                    }
                }
            }
        } else {
            var id: usize = 0;
            const dst_ptr: [*]DstT = @ptrCast(@alignCast(dst.data.?));
            var j03: i64 = 0;
            while (j03 < l.ne03) : (j03 += 1) {
                var j02: i64 = 0;
                while (j02 < l.ne02) : (j02 += 1) {
                    id += @as(usize, @intCast(l.ne00 * ir0));
                    var j01: i64 = ir0;
                    while (j01 < ir1) : (j01 += 1) {
                        var j00: i64 = 0;
                        while (j00 < l.ne00) : (j00 += 1) {
                            const soff = @as(usize, @intCast(j00)) * l.nb00 + @as(usize, @intCast(j01)) * l.nb01 +
                                @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03;
                            const src0_ptr: *const SrcT = @ptrCast(@alignCast(sdata + soff));
                            const tmp = common.toF32(SrcT, src0_ptr.*);
                            dst_ptr[id] = common.fromF32(DstT, tmp);
                            id += 1;
                        }
                    }
                    id += @as(usize, @intCast(l.ne00 * (l.ne01 - ir1)));
                }
            }
        }
        return;
    }

    // dst counters
    var j10: i64 = 0;
    var j11: i64 = 0;
    var j12: i64 = 0;
    var j13: i64 = 0;

    if (SrcT == DstT) {
        var j03: i64 = 0;
        while (j03 < l.ne03) : (j03 += 1) {
            var j02: i64 = 0;
            while (j02 < l.ne02) : (j02 += 1) {
                j10 += l.ne00 * ir0;
                while (j10 >= l.ne0) {
                    j10 -= l.ne0;
                    j11 += 1;
                    if (j11 == l.ne1) {
                        j11 = 0;
                        j12 += 1;
                        if (j12 == l.ne2) {
                            j12 = 0;
                            j13 += 1;
                            if (j13 == l.ne3) j13 = 0;
                        }
                    }
                }
                var j01: i64 = ir0;
                while (j01 < ir1) : (j01 += 1) {
                    var j00: i64 = 0;
                    while (j00 < l.ne00) : (j00 += 1) {
                        const soff = @as(usize, @intCast(j00)) * l.nb00 + @as(usize, @intCast(j01)) * l.nb01 +
                            @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03;
                        const doff = @as(usize, @intCast(j10)) * l.nb0 + @as(usize, @intCast(j11)) * l.nb1 +
                            @as(usize, @intCast(j12)) * l.nb2 + @as(usize, @intCast(j13)) * l.nb3;
                        @memcpy(ddata[doff..][0..@sizeOf(DstT)], sdata[soff..][0..@sizeOf(DstT)]);

                        // Note the bound here is `ne00`/`ne01`/… -- the
                        // *source* extents -- where the `else` arm below uses
                        // `ne0`/`ne1`/…. That asymmetry is the C's.
                        j10 += 1;
                        if (j10 == l.ne00) {
                            j10 = 0;
                            j11 += 1;
                            if (j11 == l.ne01) {
                                j11 = 0;
                                j12 += 1;
                                if (j12 == l.ne02) {
                                    j12 = 0;
                                    j13 += 1;
                                    if (j13 == l.ne03) j13 = 0;
                                }
                            }
                        }
                    }
                }
                j10 += l.ne00 * (l.ne01 - ir1);
                while (j10 >= l.ne0) {
                    j10 -= l.ne0;
                    j11 += 1;
                    if (j11 == l.ne1) {
                        j11 = 0;
                        j12 += 1;
                        if (j12 == l.ne2) {
                            j12 = 0;
                            j13 += 1;
                            if (j13 == l.ne3) j13 = 0;
                        }
                    }
                }
            }
        }
    } else {
        var j03: i64 = 0;
        while (j03 < l.ne03) : (j03 += 1) {
            var j02: i64 = 0;
            while (j02 < l.ne02) : (j02 += 1) {
                j10 += l.ne00 * ir0;
                while (j10 >= l.ne0) {
                    j10 -= l.ne0;
                    j11 += 1;
                    if (j11 == l.ne1) {
                        j11 = 0;
                        j12 += 1;
                        if (j12 == l.ne2) {
                            j12 = 0;
                            j13 += 1;
                            if (j13 == l.ne3) j13 = 0;
                        }
                    }
                }
                var j01: i64 = ir0;
                while (j01 < ir1) : (j01 += 1) {
                    var j00: i64 = 0;
                    while (j00 < l.ne00) : (j00 += 1) {
                        const soff = @as(usize, @intCast(j00)) * l.nb00 + @as(usize, @intCast(j01)) * l.nb01 +
                            @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03;
                        const doff = @as(usize, @intCast(j10)) * l.nb0 + @as(usize, @intCast(j11)) * l.nb1 +
                            @as(usize, @intCast(j12)) * l.nb2 + @as(usize, @intCast(j13)) * l.nb3;
                        const src0_ptr: *const SrcT = @ptrCast(@alignCast(sdata + soff));
                        const dst_ptr: *DstT = @ptrCast(@alignCast(ddata + doff));
                        const tmp = common.toF32(SrcT, src0_ptr.*);
                        dst_ptr.* = common.fromF32(DstT, tmp);

                        j10 += 1;
                        if (j10 == l.ne0) {
                            j10 = 0;
                            j11 += 1;
                            if (j11 == l.ne1) {
                                j11 = 0;
                                j12 += 1;
                                if (j12 == l.ne2) {
                                    j12 = 0;
                                    j13 += 1;
                                    if (j13 == l.ne3) j13 = 0;
                                }
                            }
                        }
                    }
                }
                j10 += l.ne00 * (l.ne01 - ir1);
                while (j10 >= l.ne0) {
                    j10 -= l.ne0;
                    j11 += 1;
                    if (j11 == l.ne1) {
                        j11 = 0;
                        j12 += 1;
                        if (j12 == l.ne2) {
                            j12 = 0;
                            j13 += 1;
                            if (j13 == l.ne3) j13 = 0;
                        }
                    }
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_dup_to_q` (ops.cpp:270 @c1d0e7a00), the
/// `template<typename src_t>` form.
fn dupToQ(comptime SrcT: type, params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(c.ggml_nelements(dst) == c.ggml_nelements(src0), "ggml_nelements(dst) == ggml_nelements(src0)");
    impl.assert(!c.ggml_is_quantized(src0.type), "!ggml_is_quantized(src0->type)");

    const l = common.UnaryLocals.of(src0, dst);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    // parallelize by rows
    const nr = l.ne01;
    const dr = @divTrunc(nr + nth - 1, nth);
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const traits = c.ggml_get_type_traits_cpu(dst.type);
    if (c.ggml_is_contiguous(dst) and l.nb00 == @sizeOf(SrcT) and traits.*.from_float != null) {
        // casting non-quantized types --> intermediate f32 --> quantized
        const quantize_row_q = traits.*.from_float.?;
        const wdata: [*]f32 = @ptrCast(@alignCast(params.wdata.?));
        const src0_f32 = wdata + (@as(usize, @intCast(l.ne00)) + common.cache_line_size_f32) * @as(usize, @intCast(ith));

        var id: usize = 0;
        const rs = l.nb0 * @as(usize, @intCast(@divTrunc(l.ne00, c.ggml_blck_size(dst.type))));
        const dst_ptr: [*]u8 = @ptrCast(dst.data.?);
        const sdata: [*]const u8 = @ptrCast(src0.data.?);

        var j03: i64 = 0;
        while (j03 < l.ne03) : (j03 += 1) {
            var j02: i64 = 0;
            while (j02 < l.ne02) : (j02 += 1) {
                id += rs * @as(usize, @intCast(ir0));
                var j01: i64 = ir0;
                while (j01 < ir1) : (j01 += 1) {
                    const soff = @as(usize, @intCast(j01)) * l.nb01 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03;
                    const src0_ptr: [*]const SrcT = @ptrCast(@alignCast(sdata + soff));

                    var j00: i64 = 0;
                    while (j00 < l.ne00) : (j00 += 1) {
                        src0_f32[@intCast(j00)] = common.toF32(SrcT, src0_ptr[@intCast(j00)]);
                    }

                    quantize_row_q(src0_f32, dst_ptr + id, l.ne00);
                    id += rs;
                }
                id += rs * @as(usize, @intCast(l.ne01 - ir1));
            }
        }
    } else {
        impl.abort("not implemented");
    }
}

/// Ports `ggml_compute_forward_dup_bytes` (ops.cpp:326 @c1d0e7a00).
///
/// "A simplified version of ggml_compute_forward_dup that doesn't do float
/// upcasting, and just plain old memcpy", in the C's words.
fn dupBytes(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    impl.assert(c.ggml_nelements(dst) == c.ggml_nelements(src0), "ggml_nelements(dst) == ggml_nelements(src0)");
    impl.assert(src0.type == dst.type, "src0->type == dst->type");

    const l = common.UnaryLocals.of(src0, dst);

    if (c.ggml_is_contiguous(src0) and c.ggml_is_contiguous(dst)) {
        dupSameCont(params, dst);
        return;
    }

    const type_size = c.ggml_type_size(src0.type);

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    // parallelize by rows
    const nr = l.ne01;
    const dr = @divTrunc(nr + nth - 1, nth);
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const sdata: [*]const u8 = @ptrCast(src0.data.?);
    const ddata: [*]u8 = @ptrCast(dst.data.?);

    if (src0.type == dst.type and c.ggml_are_same_shape(src0, dst) and
        l.nb00 == type_size and l.nb0 == type_size)
    {
        // copy by rows
        const rs = c.ggml_row_size(src0.type, l.ne00);
        var j03: i64 = 0;
        while (j03 < l.ne03) : (j03 += 1) {
            var j02: i64 = 0;
            while (j02 < l.ne02) : (j02 += 1) {
                var j01: i64 = ir0;
                while (j01 < ir1) : (j01 += 1) {
                    const doff = @as(usize, @intCast(j01)) * l.nb1 + @as(usize, @intCast(j02)) * l.nb2 + @as(usize, @intCast(j03)) * l.nb3;
                    const soff = @as(usize, @intCast(j01)) * l.nb01 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03;
                    @memcpy(ddata[doff..][0..rs], sdata[soff..][0..rs]);
                }
            }
        }
        return;
    }

    if (c.ggml_is_contiguous(dst)) {
        var id: usize = 0;
        const rs = @as(usize, @intCast(l.ne00)) * type_size;

        if (l.nb00 == type_size) {
            // src0 is contiguous on first dimension, copy by rows
            var j03: i64 = 0;
            while (j03 < l.ne03) : (j03 += 1) {
                var j02: i64 = 0;
                while (j02 < l.ne02) : (j02 += 1) {
                    id += rs * @as(usize, @intCast(ir0));
                    var j01: i64 = ir0;
                    while (j01 < ir1) : (j01 += 1) {
                        const soff = @as(usize, @intCast(j01)) * l.nb01 + @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03;
                        @memcpy(ddata[id..][0..rs], sdata[soff..][0..rs]);
                        id += rs;
                    }
                    id += rs * @as(usize, @intCast(l.ne01 - ir1));
                }
            }
        } else {
            var j03: i64 = 0;
            while (j03 < l.ne03) : (j03 += 1) {
                var j02: i64 = 0;
                while (j02 < l.ne02) : (j02 += 1) {
                    id += rs * @as(usize, @intCast(ir0));
                    var j01: i64 = ir0;
                    while (j01 < ir1) : (j01 += 1) {
                        var j00: i64 = 0;
                        while (j00 < l.ne00) : (j00 += 1) {
                            const soff = @as(usize, @intCast(j00)) * l.nb00 + @as(usize, @intCast(j01)) * l.nb01 +
                                @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03;
                            @memcpy(ddata[id..][0..type_size], sdata[soff..][0..type_size]);
                            id += type_size;
                        }
                    }
                    id += rs * @as(usize, @intCast(l.ne01 - ir1));
                }
            }
        }
        return;
    }

    // dst counters
    var k10: i64 = 0;
    var j11: i64 = 0;
    var j12: i64 = 0;
    var j13: i64 = 0;

    // number of blocks in a row
    const nk00 = @divTrunc(l.ne00, c.ggml_blck_size(src0.type));
    const nk0 = @divTrunc(l.ne0, c.ggml_blck_size(dst.type));

    var j03: i64 = 0;
    while (j03 < l.ne03) : (j03 += 1) {
        var j02: i64 = 0;
        while (j02 < l.ne02) : (j02 += 1) {
            k10 += nk00 * ir0;
            while (k10 >= nk0) {
                k10 -= nk0;
                j11 += 1;
                if (j11 == l.ne1) {
                    j11 = 0;
                    j12 += 1;
                    if (j12 == l.ne2) {
                        j12 = 0;
                        j13 += 1;
                        if (j13 == l.ne3) j13 = 0;
                    }
                }
            }
            var j01: i64 = ir0;
            while (j01 < ir1) : (j01 += 1) {
                var k00: i64 = 0;
                while (k00 < nk00) : (k00 += 1) {
                    const soff = @as(usize, @intCast(k00)) * l.nb00 + @as(usize, @intCast(j01)) * l.nb01 +
                        @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03;
                    const doff = @as(usize, @intCast(k10)) * l.nb0 + @as(usize, @intCast(j11)) * l.nb1 +
                        @as(usize, @intCast(j12)) * l.nb2 + @as(usize, @intCast(j13)) * l.nb3;
                    @memcpy(ddata[doff..][0..type_size], sdata[soff..][0..type_size]);

                    k10 += 1;
                    if (k10 == nk0) {
                        k10 = 0;
                        j11 += 1;
                        if (j11 == l.ne1) {
                            j11 = 0;
                            j12 += 1;
                            if (j12 == l.ne2) {
                                j12 = 0;
                                j13 += 1;
                                if (j13 == l.ne3) j13 = 0;
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (l.ne01 - ir1);
            while (k10 >= nk0) {
                k10 -= nk0;
                j11 += 1;
                if (j11 == l.ne1) {
                    j11 = 0;
                    j12 += 1;
                    if (j12 == l.ne2) {
                        j12 = 0;
                        j13 += 1;
                        if (j13 == l.ne3) j13 = 0;
                    }
                }
            }
        }
    }
}

/// Ports `ggml_compute_forward_dup_from_q` (ops.cpp:475 @c1d0e7a00).
fn dupFromQ(params: *const ComputeParams, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]);
    const src1 = impl.one(Tensor, dst.src[1]);

    const l = common.BinaryLocals.of(src0, src1, dst);

    const @"type" = src0.type;
    const dequantize_row_q = c.ggml_get_type_traits(@"type").*.to_float.?;

    const qk = c.ggml_blck_size(@"type");
    const nr = @divTrunc(c.ggml_nelements(src1), qk);

    // destination must be contiguous in the first dimension
    impl.assert(l.nb10 == c.ggml_type_size(dst.type), "nb10 == ggml_type_size(dst->type)");
    // must either have first dimension large enough to hold a row, or fully contiguous
    impl.assert(@rem(l.ne10, qk) == 0 or c.ggml_is_contiguous(dst), "(ne10 % qk) == 0 || ggml_is_contiguous(dst)");

    const ith: i64 = params.ith;
    const nth: i64 = params.nth;

    const dr = @divTrunc(nr + nth - 1, nth);
    const ir0 = dr * ith;
    const ir1 = @min(ir0 + dr, nr);

    const sdata: [*]const u8 = @ptrCast(src0.data.?);
    const ddata: [*]u8 = @ptrCast(dst.data.?);

    var ir: i64 = ir0;
    while (ir < ir1) : (ir += 1) {
        // The C narrows to `uint32_t` here, and the index arithmetic below is
        // done on the narrowed value.
        const i: i64 = @as(u32, @truncate(@as(u64, @bitCast(ir * qk))));

        const j03 = @divTrunc(i, l.ne00 * l.ne01 * l.ne02);
        const j02 = @divTrunc(i - j03 * l.ne00 * l.ne01 * l.ne02, l.ne00 * l.ne01);
        const j01 = @divTrunc(i - j03 * l.ne00 * l.ne01 * l.ne02 - j02 * l.ne01 * l.ne00, l.ne00);
        const j00 = i - j03 * l.ne00 * l.ne01 * l.ne02 - j02 * l.ne01 * l.ne00 - j01 * l.ne00;
        const x_offset = @as(usize, @intCast(@divTrunc(j00, qk))) * l.nb00 + @as(usize, @intCast(j01)) * l.nb01 +
            @as(usize, @intCast(j02)) * l.nb02 + @as(usize, @intCast(j03)) * l.nb03;

        const j13 = @divTrunc(i, l.ne10 * l.ne11 * l.ne12);
        const j12 = @divTrunc(i - j13 * l.ne10 * l.ne11 * l.ne12, l.ne10 * l.ne11);
        const j11 = @divTrunc(i - j13 * l.ne10 * l.ne11 * l.ne12 - j12 * l.ne10 * l.ne11, l.ne10);
        const j10 = i - j13 * l.ne10 * l.ne11 * l.ne12 - j12 * l.ne10 * l.ne11 - j11 * l.ne10;
        const dst_offset = @as(usize, @intCast(j10)) * l.nb10 + @as(usize, @intCast(j11)) * l.nb11 +
            @as(usize, @intCast(j12)) * l.nb12 + @as(usize, @intCast(j13)) * l.nb13;

        dequantize_row_q(sdata + x_offset, @ptrCast(@alignCast(ddata + dst_offset)), qk);
    }
}

/// Ports `ggml_compute_forward_dup` (ops.cpp:526 @c1d0e7a00).
pub export fn ggml_compute_forward_dup(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    const src0 = impl.one(Tensor, dst.src[0]);

    if (src0.type == dst.type) {
        dupBytes(params, dst);
        return;
    }

    const F16 = c.ggml_fp16_t;
    const BF16 = c.ggml_bf16_t;

    switch (src0.type) {
        c.GGML_TYPE_F16 => {
            if (dst.type == c.GGML_TYPE_F16) dupFlt(F16, F16, params, dst) else if (dst.type == c.GGML_TYPE_BF16) dupFlt(F16, BF16, params, dst) else if (dst.type == c.GGML_TYPE_F32) dupFlt(F16, f32, params, dst) else dupToQ(F16, params, dst);
        },
        c.GGML_TYPE_BF16 => {
            if (dst.type == c.GGML_TYPE_F16) dupFlt(BF16, F16, params, dst) else if (dst.type == c.GGML_TYPE_BF16) dupFlt(BF16, BF16, params, dst) else if (dst.type == c.GGML_TYPE_F32) dupFlt(BF16, f32, params, dst) else dupToQ(BF16, params, dst);
        },
        c.GGML_TYPE_F32 => {
            if (dst.type == c.GGML_TYPE_F16) dupFlt(f32, F16, params, dst) else if (dst.type == c.GGML_TYPE_BF16) dupFlt(f32, BF16, params, dst) else if (dst.type == c.GGML_TYPE_F32) dupFlt(f32, f32, params, dst) else if (dst.type == c.GGML_TYPE_I32) dupFlt(f32, i32, params, dst) else dupToQ(f32, params, dst);
        },
        c.GGML_TYPE_I32 => {
            if (dst.type == c.GGML_TYPE_F32) dupFlt(i32, f32, params, dst) else impl.abort("not implemented");
        },
        else => {
            if (c.ggml_is_quantized(src0.type) and dst.type == c.GGML_TYPE_F32) {
                dupFromQ(params, dst);
                return;
            }
            impl.abort("fatal error");
        },
    }
}

/// Ports `ggml_compute_forward_cpy` (ops.cpp:4830 @c1d0e7a00).
pub export fn ggml_compute_forward_cpy(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    ggml_compute_forward_dup(params, dst);
}

/// Ports `ggml_compute_forward_cont` (ops.cpp:4838 @c1d0e7a00).
pub export fn ggml_compute_forward_cont(params: *const ComputeParams, dst: *Tensor) callconv(.c) void {
    ggml_compute_forward_dup(params, dst);
}
