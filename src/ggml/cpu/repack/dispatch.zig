//! `ggml::cpu::repack`'s dispatch: the `tensor_traits` template, the
//! `extra_buffer_type`, and the `CPU_REPACK` buffer type that registers
//! them.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/repack.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration names the C++ it replaces and the line
//! it began at.
//!
//! # This layer is invisible to the symbol count
//!
//! Everything here has **C++ linkage** in the reference: a class template
//! over `<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>` with virtual
//! methods, a derived `extra_buffer_type`, and file-local statics. None of
//! it appears in `nm -gU | grep -v _Z`, which is why `port-coverage` read
//! `36 / 36 symbols (100%)` for a `repack.cpp` port that could not
//! function. `repack.cpp` is one of the four translation units `CLAUDE.md`
//! names as exceptions to the unmangled-exports contract, and this file is
//! what that exception meant. `scripts/cluster-check` asserts
//! `module.dispatch_implemented` at comptime so the count cannot read as
//! complete again on its own.
//!
//! # The template becomes a comptime-parameterised struct
//!
//! C++ instantiates sixteen `tensor_traits<…>` on this target and takes the
//! address of a `static const` of each. Here `TensorTraits(...)` returns a
//! type whose `instance` is that static, fronted by the explicit vtable
//! `cpu/extra.zig` defines. The four parameters select which exported
//! kernel the instantiation calls, through the three comptime tables below;
//! the C writes those as explicit template specialisations.
//!
//! The five `<…, 1, 16>` instantiations next to them, the first being
//! `q4_0_16x1_q8_0` (ggml-cpu/repack.cpp:4566 @c1d0e7a00), are inside
//! `#if defined __riscv_zvfh` and are not ported — see `convert.zig` for
//! the converters they would have reached.

const std = @import("std");
const impl = @import("../../impl.zig");
const defs = @import("../defs.zig");
const extra = @import("../extra.zig");
const features = @import("../features.zig");
const threading = @import("../threading.zig");
const cpu_traits = @import("../traits.zig");
const convert = @import("convert.zig");

const c = impl.c;
const Tensor = defs.Tensor;
const ComputeParams = defs.ComputeParams;

/// The interleaved kernels, by their C names. They are exported by the
/// sibling files in this directory and by `arm/`; declared rather than
/// imported so this file reads as the C's dispatch table does, and so a
/// missing one is a link error naming the C symbol.
const kernels = struct {
    const Gemm = fn (c_int, [*c]f32, usize, ?*const anyopaque, ?*const anyopaque, c_int, c_int) callconv(.c) void;

    extern fn ggml_gemv_q4_0_4x4_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemv_q4_0_4x8_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemv_q4_0_8x8_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemv_q2_K_8x8_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemv_q4_K_8x4_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemv_q4_K_8x8_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemv_q5_K_8x4_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemv_q5_K_8x8_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemv_q6_K_8x4_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemv_q6_K_8x8_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemv_iq4_nl_4x4_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemv_iq4_nl_8x8_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemv_mxfp4_4x4_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemv_mxfp4_8x8_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemv_q8_0_4x4_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemv_q8_0_4x8_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;

    extern fn ggml_gemm_q4_0_4x4_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemm_q4_0_4x8_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemm_q4_0_8x8_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemm_q2_K_8x8_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemm_q4_K_8x4_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemm_q4_K_8x8_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemm_q5_K_8x4_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemm_q5_K_8x8_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemm_q6_K_8x4_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemm_q6_K_8x8_q8_K(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemm_iq4_nl_4x4_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemm_iq4_nl_8x8_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemm_mxfp4_4x4_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemm_mxfp4_8x8_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemm_q8_0_4x4_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;
    extern fn ggml_gemm_q8_0_4x8_q8_0(n: c_int, s: [*c]f32, bs: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, nr: c_int, nc: c_int) void;

    extern fn ggml_quantize_mat_q8_0_4x4(x: [*c]const f32, vy: ?*anyopaque, k: i64) void;
    extern fn ggml_quantize_mat_q8_0_4x8(x: [*c]const f32, vy: ?*anyopaque, k: i64) void;
    extern fn ggml_quantize_mat_q8_K_4x4(x: [*c]const f32, vy: ?*anyopaque, k: i64) void;
    extern fn ggml_quantize_mat_q8_K_4x8(x: [*c]const f32, vy: ?*anyopaque, k: i64) void;
};

/// The block type a `tensor_traits` instantiation is over. The C uses the
/// C++ type itself as the template parameter; Zig needs a tag, because the
/// three tables below select on identity and several of the block structs
/// are structurally identical.
pub const Bloc = enum { q4_0, q4_K, q2_K, q5_K, q6_K, iq4_nl, mxfp4, q8_0 };

/// Ports the sixteen live specialisations of `gemv`
/// (ggml-cpu/repack.cpp:3963 @c1d0e7a00) as one comptime table.
fn gemvFor(comptime b: Bloc, comptime inter: i64, comptime cols: i64) *const kernels.Gemm {
    return switch (b) {
        .q4_0 => switch (inter * 100 + cols) {
            404 => &kernels.ggml_gemv_q4_0_4x4_q8_0,
            804 => &kernels.ggml_gemv_q4_0_4x8_q8_0,
            808 => &kernels.ggml_gemv_q4_0_8x8_q8_0,
            else => @compileError("no q4_0 gemv for this shape"),
        },
        .q2_K => &kernels.ggml_gemv_q2_K_8x8_q8_K,
        .q4_K => if (inter == 4) &kernels.ggml_gemv_q4_K_8x4_q8_K else &kernels.ggml_gemv_q4_K_8x8_q8_K,
        .q5_K => if (inter == 4) &kernels.ggml_gemv_q5_K_8x4_q8_K else &kernels.ggml_gemv_q5_K_8x8_q8_K,
        .q6_K => if (inter == 4) &kernels.ggml_gemv_q6_K_8x4_q8_K else &kernels.ggml_gemv_q6_K_8x8_q8_K,
        .iq4_nl => if (cols == 4) &kernels.ggml_gemv_iq4_nl_4x4_q8_0 else &kernels.ggml_gemv_iq4_nl_8x8_q8_0,
        .mxfp4 => if (cols == 4) &kernels.ggml_gemv_mxfp4_4x4_q8_0 else &kernels.ggml_gemv_mxfp4_8x8_q8_0,
        .q8_0 => if (inter == 4) &kernels.ggml_gemv_q8_0_4x4_q8_0 else &kernels.ggml_gemv_q8_0_4x8_q8_0,
    };
}

/// Ports the sixteen live specialisations of `gemm`
/// (ggml-cpu/repack.cpp:4060 @c1d0e7a00).
fn gemmFor(comptime b: Bloc, comptime inter: i64, comptime cols: i64) *const kernels.Gemm {
    return switch (b) {
        .q4_0 => switch (inter * 100 + cols) {
            404 => &kernels.ggml_gemm_q4_0_4x4_q8_0,
            804 => &kernels.ggml_gemm_q4_0_4x8_q8_0,
            808 => &kernels.ggml_gemm_q4_0_8x8_q8_0,
            else => @compileError("no q4_0 gemm for this shape"),
        },
        .q2_K => &kernels.ggml_gemm_q2_K_8x8_q8_K,
        .q4_K => if (inter == 4) &kernels.ggml_gemm_q4_K_8x4_q8_K else &kernels.ggml_gemm_q4_K_8x8_q8_K,
        .q5_K => if (inter == 4) &kernels.ggml_gemm_q5_K_8x4_q8_K else &kernels.ggml_gemm_q5_K_8x8_q8_K,
        .q6_K => if (inter == 4) &kernels.ggml_gemm_q6_K_8x4_q8_K else &kernels.ggml_gemm_q6_K_8x8_q8_K,
        .iq4_nl => if (cols == 4) &kernels.ggml_gemm_iq4_nl_4x4_q8_0 else &kernels.ggml_gemm_iq4_nl_8x8_q8_0,
        .mxfp4 => if (cols == 4) &kernels.ggml_gemm_mxfp4_4x4_q8_0 else &kernels.ggml_gemm_mxfp4_8x8_q8_0,
        .q8_0 => if (inter == 4) &kernels.ggml_gemm_q8_0_4x4_q8_0 else &kernels.ggml_gemm_q8_0_4x8_q8_0,
    };
}

/// Ports the four specialisations of `ggml_quantize_mat_t`, `ggml_quantize_mat_t`,
/// `ggml_quantize_mat_t` and `ggml_quantize_mat_t`
/// (ggml-cpu/repack.cpp:318, 324, 330, 336 @c1d0e7a00): four rows of
/// activations at a time.
fn quantizeMatFor(comptime inter: i64, comptime param: c_uint) *const fn ([*c]const f32, ?*anyopaque, i64) callconv(.c) void {
    return if (param == c.GGML_TYPE_Q8_0)
        (if (inter == 4) &kernels.ggml_quantize_mat_q8_0_4x4 else &kernels.ggml_quantize_mat_q8_0_4x8)
    else
        (if (inter == 4) &kernels.ggml_quantize_mat_q8_K_4x4 else &kernels.ggml_quantize_mat_q8_K_4x8);
}

/// Ports the specialisations of `repack` (ggml-cpu/repack.cpp:3868
/// @c1d0e7a00), which pair a block type and shape with one of
/// `convert.zig`'s drivers.
fn repackFor(comptime b: Bloc, comptime inter: i64, comptime cols: i64) *const ConvertFn {
    return switch (b) {
        .q4_0 => switch (inter * 100 + cols) {
            404, 804 => &convert.repackQ4_0toQ4_0_4,
            808 => &convert.repackQ4_0toQ4_0_8,
            else => @compileError("no q4_0 repack for this shape"),
        },
        .q4_K => &convert.repackQ4_KtoQ4_K_8,
        .q2_K => &convert.repackQ2_KtoQ2_K_8,
        .q5_K => &convert.repackQ5_KtoQ5_K_8,
        .q6_K => &convert.repackQ6_KtoQ6_K_8,
        .iq4_nl => if (cols == 4) &convert.repackIq4NltoIq4Nl_4 else &convert.repackIq4NltoIq4Nl_8,
        .mxfp4 => if (cols == 4) &convert.repackMxfp4toMxfp4_4 else &convert.repackMxfp4toMxfp4_8,
        .q8_0 => &convert.repackQ8_0toQ8_0_4,
    };
}

// -----------------------------------------------------------------------------
// tensor_traits

/// Ports the class template `tensor_traits` (ggml-cpu/repack.cpp:4158
/// @c1d0e7a00), together with the `tensor_traits_base`
/// (ggml-cpu/repack.cpp:4153 @c1d0e7a00) above it, whose only job is to add
/// `repack` to the interface.
///
/// Parameters (all comptime, the C's template parameters):
/// - `bloc`: the stored block type.
/// - `inter_size`: `INTER_SIZE`, the activation interleave.
/// - `nb_cols`: `NB_COLS`, how many weight rows one interleaved block holds.
/// - `param_type`: `PARAM_TYPE`, the type activations are quantized to.
///
/// Return: a type whose `instance` is the one `static const` the C holds
/// per instantiation, and whose `base` is the `extra.TensorTraits` the
/// vtable dispatch reaches it through.
fn TensorTraits(
    comptime bloc: Bloc,
    comptime inter_size: i64,
    comptime nb_cols: i64,
    comptime param_type: c_uint,
) type {
    return struct {
        const gemv = gemvFor(bloc, inter_size, nb_cols);
        const gemm = gemmFor(bloc, inter_size, nb_cols);
        const quantize_mat = quantizeMatFor(inter_size, param_type);
        const repackTensor = repackFor(bloc, inter_size, nb_cols);

        const vtable: extra.TensorTraits.VTable = .{
            .work_size = workSize,
            .compute_forward = computeForward,
        };

        /// The C's `static const tensor_traits<…>`, one per instantiation.
        const instance: Extended = .{
            .base = .{ .vtable = &vtable },
            .repack = repackWithLog,
        };

        /// Ports `tensor_traits`'s own `repack` (ggml-cpu/repack.cpp:4519
        /// @c1d0e7a00): the log line, then the converter the template
        /// parameters select.
        ///
        /// `INTER_SIZE` is a template argument in the C, so it is baked in
        /// here too rather than carried as a field — the converter's
        /// second parameter is the only place it is still a value.
        fn repackWithLog(t: *c.ggml_tensor, data: *const anyopaque, data_size: usize) callconv(.c) c_int {
            impl.logDebug("%s: repack tensor %s with %s_%dx%d\n", .{
                "repack",
                @as([*:0]const u8, @ptrCast(&t.name)),
                c.ggml_type_name(t.type),
                @as(c_int, nb_cols),
                @as(c_int, inter_size),
            });
            return repackTensor(t, @intCast(inter_size), data, data_size);
        }

        /// Ports `work_size` (ggml-cpu/repack.cpp:4160 @c1d0e7a00).
        fn workSize(self: *const extra.TensorTraits, n_threads: c_int, op: *const Tensor, size: *usize) callconv(.c) bool {
            _ = self;
            _ = n_threads;
            switch (op.op) {
                // Not really a Q8_0, but the same size.
                c.GGML_OP_MUL_MAT => {
                    size.* = c.ggml_row_size(param_type, c.ggml_nelements(op.src[1]));
                    return true;
                },
                c.GGML_OP_MUL_MAT_ID => {
                    var sz = c.ggml_row_size(param_type, c.ggml_nelements(op.src[1]));
                    sz = impl.pad(sz, @sizeOf(i64)); // + padding for the next block

                    // `p.*.ne[2]` on a `[*c]` types as the whole `[4]i64`
                    // in Zig 0.16, so both are narrowed first.
                    const ne02 = impl.one(Tensor, op.src[0]).ne[2]; // n_as, n_expert
                    const ne12 = impl.one(Tensor, op.src[1]).ne[2]; // n_tokens

                    const sizeof_mmid_row_mapping = @sizeOf(i64);

                    sz += sizeof_mmid_row_mapping * @as(usize, @intCast(ne02)) * @as(usize, @intCast(ne12 + 1));

                    size.* = sz;
                    return true;
                },
                // The C's `GGML_ABORT("fatal error")` here is commented out.
                else => {},
            }
            return false;
        }

        /// Ports `compute_forward` (ggml-cpu/repack.cpp:4189 @c1d0e7a00).
        fn computeForward(self: *const extra.TensorTraits, params: *ComputeParams, op: *Tensor) callconv(.c) bool {
            _ = self;
            switch (op.op) {
                c.GGML_OP_MUL_MAT => {
                    forwardMulMat(params, op);
                    return true;
                },
                c.GGML_OP_MUL_MAT_ID => {
                    forwardMulMatId(params, op);
                    return true;
                },
                else => {},
            }
            return false;
        }

        /// Ports `forward_mul_mat_one_chunk` (ggml-cpu/repack.cpp:4204 @c1d0e7a00).
        fn forwardMulMatOneChunk(
            params: *ComputeParams,
            op: *Tensor,
            src0_start: i64,
            src0_end: i64,
            src1_start: i64,
            src1_end: i64,
        ) void {
            const src0 = impl.one(Tensor, op.src[0]);
            const src1 = impl.one(Tensor, op.src[1]);
            const dst = op;

            const l = defs.BinaryLocals.of(src0, src1, dst);

            const src1_col_stride = c.ggml_row_size(param_type, l.ne10);

            impl.assert(l.ne03 == 1 and l.ne13 == 1, "ne03 == 1 && ne13 == 1");
            impl.assert(@rem(l.ne12, l.ne02) == 0, "ne12 % ne02 == 0");
            const r2 = @divTrunc(l.ne12, l.ne02);

            const j12 = @divTrunc(src1_start, l.ne1);
            const j11 = src1_start - j12 * l.ne1;

            // Determine batch index
            const j02 = @divTrunc(j12, r2);

            const j1 = j11;
            const j2 = j12;

            const src0_ptr: [*]const u8 = @as([*]const u8, @ptrCast(src0.data.?)) + @as(usize, @intCast(j02)) * l.nb02;
            const src1_ptr: [*]const u8 = @as([*]const u8, @ptrCast(params.wdata.?)) +
                @as(usize, @intCast(j11 + j12 * l.ne11)) * src1_col_stride;
            const dst_ptr: [*]u8 = @as([*]u8, @ptrCast(dst.data.?)) +
                (@as(usize, @intCast(j1)) * l.nb1 + @as(usize, @intCast(j2)) * l.nb2);

            const nrows = src1_end - src1_start;
            const ncols = src0_end - src0_start;

            impl.assert(
                @intFromPtr(src1_ptr) + src1_col_stride * @as(usize, @intCast(nrows)) <=
                    @intFromPtr(params.wdata.?) + params.wsize,
                "the quantized activations fit in wdata",
            );

            // More than three rows of src1 means gemm pays; otherwise gemv.
            if (nrows > 3) {
                gemm(
                    @intCast(l.ne00),
                    @ptrCast(@alignCast(dst_ptr + @as(usize, @intCast(src0_start)) * @sizeOf(f32))),
                    l.nb1 / l.nb0,
                    src0_ptr + @as(usize, @intCast(src0_start)) * l.nb01,
                    src1_ptr,
                    @intCast(nrows - @rem(nrows, 4)),
                    @intCast(ncols),
                );
            }
            var iter = nrows - @rem(nrows, 4);
            while (iter < nrows) : (iter += 1) {
                gemv(
                    @intCast(l.ne00),
                    @ptrCast(@alignCast(dst_ptr + @as(usize, @intCast(iter)) * l.nb1 +
                        @as(usize, @intCast(src0_start)) * @sizeOf(f32))),
                    @intCast(l.ne01),
                    src0_ptr + @as(usize, @intCast(src0_start)) * l.nb01,
                    src1_ptr + src1_col_stride * @as(usize, @intCast(iter)),
                    1, // nrows
                    @intCast(ncols),
                );
            }
        }

        /// Ports `forward_mul_mat` (ggml-cpu/repack.cpp:4253 @c1d0e7a00).
        fn forwardMulMat(params: *ComputeParams, op: *Tensor) void {
            const src0 = impl.one(Tensor, op.src[0]);
            const src1 = impl.one(Tensor, op.src[1]);
            const dst = op;

            const l = defs.BinaryLocals.of(src0, src1, dst);

            const ith = params.ith;
            const nth = params.nth;

            impl.assert(l.ne0 == l.ne01, "ne0 == ne01");
            impl.assert(l.ne1 == l.ne11, "ne1 == ne11");
            impl.assert(l.ne2 == l.ne12, "ne2 == ne12");
            impl.assert(l.ne3 == l.ne13, "ne3 == ne13");

            // dst cannot be transposed or permuted
            impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");
            impl.assert(l.nb0 <= l.nb1, "nb0 <= nb1");
            impl.assert(l.nb1 <= l.nb2, "nb1 <= nb2");
            impl.assert(l.nb2 <= l.nb3, "nb2 <= nb3");

            // Only 3D tensors; the C has a TODO for the general 4D case.
            impl.assert(l.ne03 == 1, "ne03 == 1");
            impl.assert(l.ne13 == 1, "ne13 == 1");
            impl.assert(l.ne3 == 1, "ne3 == 1");

            impl.assert(src1.type == c.GGML_TYPE_F32, "src1 is f32");
            impl.assert(c.ggml_n_dims(src0) == 2, "src0 is 2D");

            const wdata: [*]u8 = @ptrCast(params.wdata.?);
            const nbw1 = c.ggml_row_size(param_type, l.ne10);
            const nbw2 = nbw1 * @as(usize, @intCast(l.ne11));

            impl.assert(params.wsize >= nbw2 * @as(usize, @intCast(l.ne12)), "wsize is large enough");

            const from_float = c.ggml_get_type_traits_cpu(param_type).*.from_float.?;

            // Quantization is done in planes to avoid extra complexity in
            // chunking: flattening dimensions not a multiple of INTER_SIZE
            // would need extra handling per broadcast shape.
            for (0..@intCast(l.ne12)) |plane| {
                const j12: i64 = @intCast(plane);
                const data_ptr: [*]u8 = @as([*]u8, @ptrCast(src1.data.?)) + @as(usize, @intCast(j12)) * l.nb12;
                const wdata_ptr: [*]u8 = wdata + @as(usize, @intCast(j12)) * nbw2;

                var j11: i64 = @as(i64, ith) * 4;
                while (j11 < l.ne11 - @rem(l.ne11, 4)) : (j11 += @as(i64, nth) * 4) {
                    quantize_mat(
                        @ptrCast(@alignCast(data_ptr + @as(usize, @intCast(j11)) * l.nb11)),
                        wdata_ptr + @as(usize, @intCast(j11)) * nbw1,
                        l.ne10,
                    );
                }

                const j11_processed = l.ne11 - @rem(l.ne11, 4);
                j11 = j11_processed + ith;
                while (j11 < l.ne11) : (j11 += nth) {
                    from_float(
                        @ptrCast(@alignCast(data_ptr + @as(usize, @intCast(j11)) * l.nb11)),
                        wdata_ptr + @as(usize, @intCast(j11)) * nbw1,
                        l.ne10,
                    );
                }
            }

            // disable for NUMA
            const disable_chunking = threading.ggml_is_numa();

            // 4x chunks per thread
            const nr0 = c.ggml_nrows(src0);

            const nth_scaled = nth * 4;
            const chunk_size0 = @divTrunc(nr0 + nth_scaled - 1, nth_scaled);
            var nchunk0 = @divTrunc(nr0 + chunk_size0 - 1, chunk_size0);

            // src1 is chunked only by full planes: when we flatten we have
            // to route dimensions not a multiple of the q8 INTER_SIZE
            // through GEMV. `nchunk1 = ne12` also leaves models with no 3D
            // tensors chunked exactly as they were.
            const nchunk1 = l.ne12;

            // A chunk below NB_COLS could overlap its neighbour once both
            // are rounded up to an NB_COLS boundary below.
            const min_chunk_size: i64 = nb_cols;
            if (nchunk0 > 0 and @divTrunc(nr0, nchunk0) < min_chunk_size and nr0 >= min_chunk_size) {
                nchunk0 = @divTrunc(nr0 + min_chunk_size - 1, min_chunk_size);
            }

            var dr0 = @divTrunc(nr0 + nchunk0 - 1, nchunk0);
            // Only raise nchunk0 to nth if that will not make chunks too small.
            if (nth == 1 or ((nchunk0 < nth or disable_chunking) and @divTrunc(nr0 + nth - 1, nth) >= min_chunk_size)) {
                nchunk0 = nth;
                dr0 = @divTrunc(nr0 + nchunk0 - 1, nchunk0);
            }

            const max_nchunk = @divTrunc(nr0 + min_chunk_size - 1, min_chunk_size);
            nchunk0 = @min(nchunk0, max_nchunk);

            if (ith == 0) {
                // Every thread starts at ith, so the first unclaimed chunk
                // is nth. That saves a round of coordination at the start.
                threading.ggml_threadpool_chunk_set(@ptrCast(@alignCast(params.threadpool.?)), nth);
            }

            threading.ggml_barrier(@ptrCast(@alignCast(params.threadpool.?)));

            // The first chunk comes from our thread id; the rest are claimed.
            var current_chunk: i64 = ith;

            while (current_chunk < nchunk0 * nchunk1) {
                const ith0 = @rem(current_chunk, nchunk0);
                const ith1 = @divTrunc(current_chunk, nchunk0);

                var src0_start = dr0 * ith0;
                var src0_end = @min(src0_start + dr0, nr0);

                // full-plane range for src1
                const src1_start = ith1 * l.ne11;
                const src1_end = (ith1 + 1) * l.ne11;

                // Round both boundaries up to NB_COLS so no row is dropped;
                // the chunk-size floor above is what keeps that from
                // overlapping the next chunk.
                src0_start = if (@rem(src0_start, nb_cols) != 0) src0_start + nb_cols - @rem(src0_start, nb_cols) else src0_start;
                src0_end = if (@rem(src0_end, nb_cols) != 0) src0_end + nb_cols - @rem(src0_end, nb_cols) else src0_end;
                src0_end = @min(src0_end, l.ne01);

                // Make sure the current plane is the last one before exiting.
                if (src0_start >= src0_end) {
                    current_chunk = threading.ggml_threadpool_chunk_add(@ptrCast(@alignCast(params.threadpool.?)), 1);
                    continue;
                }

                forwardMulMatOneChunk(params, dst, src0_start, src0_end, src1_start, src1_end);

                current_chunk = threading.ggml_threadpool_chunk_add(@ptrCast(@alignCast(params.threadpool.?)), 1);
            }
        }

        /// Ports `forward_mul_mat_id` (ggml-cpu/repack.cpp:4386 @c1d0e7a00).
        fn forwardMulMatId(params: *ComputeParams, op: *Tensor) void {
            const src0 = impl.one(Tensor, op.src[0]);
            const src1 = impl.one(Tensor, op.src[1]);
            const ids = impl.one(Tensor, op.src[2]);
            const dst = op;

            const l = defs.BinaryLocals.of(src0, src1, dst);

            const ith = params.ith;
            const nth = params.nth;

            const from_float = c.ggml_get_type_traits_cpu(param_type).*.from_float.?;

            // we don't support permuted src0 or src1
            impl.assert(l.nb00 == c.ggml_type_size(src0.type), "src0 is contiguous in its rows");
            impl.assert(l.nb10 == c.ggml_type_size(src1.type), "src1 is contiguous in its rows");

            // dst cannot be transposed or permuted
            impl.assert(l.nb0 == @sizeOf(f32), "nb0 == sizeof(float)");
            impl.assert(l.nb0 <= l.nb1, "nb0 <= nb1");
            impl.assert(l.nb1 <= l.nb2, "nb1 <= nb2");
            impl.assert(l.nb2 <= l.nb3, "nb2 <= nb3");

            impl.assert(l.ne03 == 1, "ne03 == 1");
            impl.assert(l.ne13 == 1, "ne13 == 1");
            impl.assert(l.ne3 == 1, "ne3 == 1");

            impl.assert(src1.type == c.GGML_TYPE_F32, "src1 is f32");

            // row groups
            const n_ids = ids.ne[0]; // n_expert_used
            const n_as = l.ne02; // n_expert

            const nbw1 = c.ggml_row_size(param_type, l.ne10);
            const nbw2 = nbw1 * @as(usize, @intCast(l.ne11));
            const nbw3 = nbw2 * @as(usize, @intCast(l.ne12));

            impl.assert(
                params.wsize >= impl.pad(nbw3, @sizeOf(i64)) +
                    @as(usize, @intCast(n_as)) * @as(usize, @intCast(l.ne12 + 1)) * @sizeOf(MmidRowMapping),
                "wsize covers the activations and the row map",
            );

            const wdata: [*]u8 = @ptrCast(params.wdata.?);
            const wdata_src1_end: [*]u8 = wdata + impl.pad(nbw3, @sizeOf(i64));

            // [n_as][ne12 + 1] of mmid_row_mapping, which is two int32_t,
            // so the counts and the map share one int64-aligned run.
            const matrix_row_counts: [*]i64 = @ptrCast(@alignCast(wdata_src1_end)); // [n_as]
            const matrix_rows: [*]MmidRowMapping = @ptrCast(@alignCast(matrix_row_counts + @as(usize, @intCast(n_as)))); // [n_as][ne12]

            // src1: float32 => param type
            for (0..@intCast(l.ne12)) |plane| {
                const j12: usize = plane;
                var j11: i64 = ith;
                while (j11 < l.ne11) : (j11 += nth) {
                    from_float(
                        @ptrCast(@alignCast(@as([*]u8, @ptrCast(src1.data.?)) + j12 * l.nb12 + @as(usize, @intCast(j11)) * l.nb11)),
                        wdata + j12 * nbw2 + @as(usize, @intCast(j11)) * nbw1,
                        l.ne10,
                    );
                }
            }

            // The C's `MMID_MATRIX_ROW(row_id, i1)` macro.
            const Row = struct {
                inline fn at(rows: [*]MmidRowMapping, row_id: i64, j1: i64, ne12: i64) *MmidRowMapping {
                    return &rows[@intCast(row_id * ne12 + j1)];
                }
            };

            if (ith == 0) {
                // initialize matrix_row_counts
                @memset(matrix_row_counts[0..@intCast(n_as)], 0);

                // group rows by src0 matrix
                var iid1: i64 = 0;
                while (iid1 < ids.ne[1]) : (iid1 += 1) {
                    var id: i64 = 0;
                    while (id < n_ids) : (id += 1) {
                        const j02: i32 = @as(*const i32, @ptrCast(@alignCast(
                            @as([*]const u8, @ptrCast(ids.data.?)) +
                                @as(usize, @intCast(iid1)) * ids.nb[1] +
                                @as(usize, @intCast(id)) * ids.nb[0],
                        ))).*;

                        impl.assert(j02 >= 0 and j02 < n_as, "the expert index is in range");

                        Row.at(matrix_rows, j02, matrix_row_counts[@intCast(j02)], l.ne12).* =
                            .{ .i1 = @intCast(id), .i2 = @intCast(iid1) };
                        matrix_row_counts[@intCast(j02)] += 1;
                    }
                }
            }

            threading.ggml_barrier(@ptrCast(@alignCast(params.threadpool.?)));

            // compute each matrix multiplication in sequence
            var cur_a: i64 = 0;
            while (cur_a < n_as) : (cur_a += 1) {
                const cne1 = matrix_row_counts[@intCast(cur_a)];

                if (cne1 == 0) continue;

                const src0_cur: [*]const u8 = @as([*]const u8, @ptrCast(src0.data.?)) + @as(usize, @intCast(cur_a)) * l.nb02;

                const nr1 = cne1; // src1 rows

                var src0_cur_start = @divTrunc(@as(i64, ith) * l.ne01, nth);
                var src0_cur_end = @divTrunc(@as(i64, ith + 1) * l.ne01, nth);

                // Round up to NB_COLS so no row is dropped.
                src0_cur_start = if (@rem(src0_cur_start, nb_cols) != 0) src0_cur_start + nb_cols - @rem(src0_cur_start, nb_cols) else src0_cur_start;
                src0_cur_end = if (@rem(src0_cur_end, nb_cols) != 0) src0_cur_end + nb_cols - @rem(src0_cur_end, nb_cols) else src0_cur_end;
                if (src0_cur_end > l.ne01) src0_cur_end = l.ne01;

                // The C returns rather than continuing: with the split made
                // by thread index alone, a thread with nothing to do in one
                // expert has nothing to do in any of them.
                if (src0_cur_start >= src0_cur_end) return;

                var ir1: i64 = 0;
                while (ir1 < nr1) : (ir1 += 1) {
                    const row_mapping = Row.at(matrix_rows, cur_a, ir1, l.ne12).*;

                    const id: i64 = row_mapping.i1; // selected expert index

                    const j11 = @rem(id, l.ne11);
                    const j12: i64 = row_mapping.i2; // row index in src1

                    const j1 = id; // selected expert index
                    const j2 = j12; // row

                    const src1_col: [*]const u8 = wdata +
                        @as(usize, @intCast(j11)) * nbw1 + @as(usize, @intCast(j12)) * nbw2;

                    gemv(
                        @intCast(l.ne00),
                        @ptrCast(@alignCast(@as([*]u8, @ptrCast(dst.data.?)) +
                            (@as(usize, @intCast(j1)) * l.nb1 + @as(usize, @intCast(j2)) * l.nb2) +
                            @as(usize, @intCast(src0_cur_start)) * @sizeOf(f32))),
                        @intCast(l.ne01),
                        src0_cur + @as(usize, @intCast(src0_cur_start)) * l.nb01,
                        src1_col,
                        1,
                        @intCast(src0_cur_end - src0_cur_start),
                    );
                }
            }
        }
    };
}

/// Ports `mmid_row_mapping` (ggml-cpu/repack.cpp:4423 @c1d0e7a00),
/// declared inside `forward_mul_mat_id` in the C.
const MmidRowMapping = extern struct {
    i1: i32,
    i2: i32,
};

// -----------------------------------------------------------------------------
// The instances, and the choice between them

/// Ports the `static const` instances inside
/// `ggml_repack_get_optimal_repack_type` (ggml-cpu/repack.cpp:4528
/// @c1d0e7a00). The five RISC-V ones beside them are behind `#if defined
/// __riscv_zvfh` and are not here.
const inst = struct {
    const q4_0_4x4_q8_0 = TensorTraits(.q4_0, 4, 4, c.GGML_TYPE_Q8_0);
    const q4_0_4x8_q8_0 = TensorTraits(.q4_0, 8, 4, c.GGML_TYPE_Q8_0);
    const q4_0_8x8_q8_0 = TensorTraits(.q4_0, 8, 8, c.GGML_TYPE_Q8_0);
    const q4_K_8x4_q8_K = TensorTraits(.q4_K, 4, 8, c.GGML_TYPE_Q8_K);
    const q4_K_8x8_q8_K = TensorTraits(.q4_K, 8, 8, c.GGML_TYPE_Q8_K);
    const q5_K_8x4_q8_K = TensorTraits(.q5_K, 4, 8, c.GGML_TYPE_Q8_K);
    const q5_K_8x8_q8_K = TensorTraits(.q5_K, 8, 8, c.GGML_TYPE_Q8_K);
    const q6_K_8x4_q8_K = TensorTraits(.q6_K, 4, 8, c.GGML_TYPE_Q8_K);
    const q6_K_8x8_q8_K = TensorTraits(.q6_K, 8, 8, c.GGML_TYPE_Q8_K);
    const q2_K_8x8_q8_K = TensorTraits(.q2_K, 8, 8, c.GGML_TYPE_Q8_K);
    const iq4_nl_4x4_q8_0 = TensorTraits(.iq4_nl, 4, 4, c.GGML_TYPE_Q8_0);
    const iq4_nl_8x8_q8_0 = TensorTraits(.iq4_nl, 8, 8, c.GGML_TYPE_Q8_0);
    const mxfp4_4x4_q8_0 = TensorTraits(.mxfp4, 4, 4, c.GGML_TYPE_Q8_0);
    const mxfp4_8x8_q8_0 = TensorTraits(.mxfp4, 8, 8, c.GGML_TYPE_Q8_0);
    const q8_0_4x4_q8_0 = TensorTraits(.q8_0, 4, 4, c.GGML_TYPE_Q8_0);
    const q8_0_4x8_q8_0 = TensorTraits(.q8_0, 8, 4, c.GGML_TYPE_Q8_0);
};

/// The one method `tensor_traits_base` (ggml-cpu/repack.cpp:4153
/// @c1d0e7a00) adds to the interface. Three parameters, like the C's:
/// `INTER_SIZE` reaches the converter as a template argument, not an
/// argument.
const RepackFn = fn (*c.ggml_tensor, *const anyopaque, usize) callconv(.c) c_int;

/// `convert.zig`'s drivers, which *do* take the interleave as a value —
/// the C's `repack_q4_0_to_q4_0_4_bl(t, 4, data, size)`.
const ConvertFn = fn (*c.ggml_tensor, c_int, *const anyopaque, usize) callconv(.c) c_int;

/// The `tensor_traits_base *` a tensor's `extra` holds: the vtable front,
/// plus the `repack` entry only `set_tensor` reaches.
///
/// Spelled out as its own name because the two directions — storing the
/// base pointer, recovering the extended one — are what the C's
/// `const_cast` and downcast do.
const Extended = extern struct {
    base: extra.TensorTraits,
    repack: *const RepackFn,
};

/// The base pointer the C hands back from
/// `ggml_repack_get_optimal_repack_type`, which is what lands in
/// `tensor->extra`. `set_tensor` casts it back to `*const Extended` to
/// reach `repack`, exactly as the C++ downcasts — sound because the base
/// is the first field.
inline fn traitsOf(comptime T: type) *const extra.TensorTraits {
    return &T.instance.base;
}

/// Ports `ggml_repack_get_optimal_repack_type` (ggml-cpu/repack.cpp:4528
/// @c1d0e7a00).
///
/// Parameters:
/// - `cur`: the weight tensor being considered.
///
/// Return: the traits to hang off `cur->extra`, or null when no
/// interleaved kernel fits this type and shape. Borrowed; the instances
/// are statics that live for the process.
pub fn getOptimalRepackType(cur: *const Tensor) ?*const extra.TensorTraits {
    const ne1 = cur.ne[1];

    if (cur.type == c.GGML_TYPE_Q4_0) {
        if (features.ggml_cpu_has_avx2() != 0 or
            (features.ggml_cpu_has_sve() != 0 and features.ggml_cpu_has_matmul_int8() != 0 and features.ggml_cpu_get_sve_cnt() == 32))
        {
            if (@rem(ne1, 8) == 0) return traitsOf(inst.q4_0_8x8_q8_0);
        }
        if (features.ggml_cpu_has_neon() != 0 and features.ggml_cpu_has_matmul_int8() != 0) {
            if (@rem(ne1, 4) == 0) return traitsOf(inst.q4_0_4x8_q8_0);
        }
        if (features.ggml_cpu_has_neon() != 0 and features.ggml_cpu_has_dotprod() != 0) {
            if (@rem(ne1, 4) == 0) return traitsOf(inst.q4_0_4x4_q8_0);
        }
        // The C's `ggml_cpu_has_riscv_v()` arm is empty without __riscv_zvfh.
    } else if (cur.type == c.GGML_TYPE_Q4_K) {
        if (features.ggml_cpu_has_avx2() != 0) {
            if (@rem(ne1, 8) == 0) return traitsOf(inst.q4_K_8x8_q8_K);
        }
        if (features.ggml_cpu_has_neon() != 0 and features.ggml_cpu_has_matmul_int8() != 0) {
            if (@rem(ne1, 8) == 0) return traitsOf(inst.q4_K_8x8_q8_K);
        }
        if (features.ggml_cpu_has_neon() != 0 and features.ggml_cpu_has_dotprod() != 0) {
            if (@rem(ne1, 8) == 0) return traitsOf(inst.q4_K_8x4_q8_K);
        }
    } else if (cur.type == c.GGML_TYPE_Q2_K) {
        if (features.ggml_cpu_has_avx512() != 0) {
            if (@rem(ne1, 8) == 0) return traitsOf(inst.q2_K_8x8_q8_K);
        }
    } else if (cur.type == c.GGML_TYPE_Q5_K) {
        if (features.ggml_cpu_has_neon() != 0 and features.ggml_cpu_has_matmul_int8() != 0) {
            if (@rem(ne1, 8) == 0) return traitsOf(inst.q5_K_8x8_q8_K);
        }
        if (features.ggml_cpu_has_neon() != 0 and features.ggml_cpu_has_dotprod() != 0) {
            if (@rem(ne1, 8) == 0) return traitsOf(inst.q5_K_8x4_q8_K);
        }
    } else if (cur.type == c.GGML_TYPE_Q6_K) {
        if (features.ggml_cpu_has_neon() != 0 and features.ggml_cpu_has_matmul_int8() != 0) {
            if (@rem(ne1, 8) == 0) return traitsOf(inst.q6_K_8x8_q8_K);
        }
        if (features.ggml_cpu_has_neon() != 0 and features.ggml_cpu_has_dotprod() != 0) {
            if (@rem(ne1, 8) == 0) return traitsOf(inst.q6_K_8x4_q8_K);
        }
    } else if (cur.type == c.GGML_TYPE_IQ4_NL) {
        if (features.ggml_cpu_has_avx2() != 0) {
            if (@rem(ne1, 8) == 0) return traitsOf(inst.iq4_nl_8x8_q8_0);
        }
        if (features.ggml_cpu_has_neon() != 0 and features.ggml_cpu_has_dotprod() != 0) {
            if (@rem(ne1, 4) == 0) return traitsOf(inst.iq4_nl_4x4_q8_0);
        }
    } else if (cur.type == c.GGML_TYPE_MXFP4) {
        if (features.ggml_cpu_has_avx2() != 0) {
            if (@rem(ne1, 8) == 0) return traitsOf(inst.mxfp4_8x8_q8_0);
        }
        if (features.ggml_cpu_has_neon() != 0 and features.ggml_cpu_has_dotprod() != 0) {
            if (@rem(ne1, 4) == 0) return traitsOf(inst.mxfp4_4x4_q8_0);
        }
    } else if (cur.type == c.GGML_TYPE_Q8_0) {
        if (features.ggml_cpu_has_neon() != 0 and features.ggml_cpu_has_matmul_int8() != 0) {
            if (@rem(ne1, 4) == 0) return traitsOf(inst.q8_0_4x8_q8_0);
        }
        if (features.ggml_cpu_has_neon() != 0 and features.ggml_cpu_has_dotprod() != 0) {
            if (@rem(ne1, 4) == 0) return traitsOf(inst.q8_0_4x4_q8_0);
        }
    }

    return null;
}

// -----------------------------------------------------------------------------
// The buffer

/// Ports `ggml_backend_cpu_repack_buffer_init_tensor` (ggml-cpu/repack.cpp:4726
/// @c1d0e7a00).
fn bufferInitTensor(buffer: c.ggml_backend_buffer_t, tensor: [*c]c.ggml_tensor) callconv(.c) c.ggml_status {
    _ = buffer;
    const t = impl.one(c.ggml_tensor, tensor);
    t.extra = @ptrCast(@constCast(getOptimalRepackType(t)));
    return c.GGML_STATUS_SUCCESS;
}

/// Ports `ggml_backend_cpu_repack_buffer_set_tensor` (ggml-cpu/repack.cpp:4733
/// @c1d0e7a00).
///
/// This is where the weights are interleaved: the loader writes the stored
/// blocks here once, and they are repacked in place of being copied.
fn bufferSetTensor(
    buffer: c.ggml_backend_buffer_t,
    tensor: [*c]c.ggml_tensor,
    data: ?*const anyopaque,
    offset: usize,
    size: usize,
) callconv(.c) void {
    _ = buffer;
    const t = impl.one(c.ggml_tensor, tensor);
    impl.assert(offset == 0, "offset == 0");
    impl.assert(size == c.ggml_nbytes(t), "size == ggml_nbytes(tensor)");

    // The C's downcast from `tensor_traits *` to `tensor_traits_base *`.
    const traits: *const Extended = @ptrCast(@alignCast(t.extra.?));
    const ok = traits.repack(t, data.?, size);

    impl.assert(ok == 0, "the repack succeeded");
}

/// Ports `ggml_backend_cpu_repack_buffer_type_get_name` (ggml-cpu/repack.cpp:4745
/// @c1d0e7a00).
fn buftGetName(buft: c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    _ = buft;
    return "CPU_REPACK";
}

/// Ports `ggml_backend_cpu_repack_buffer_type_alloc_buffer`
/// (ggml-cpu/repack.cpp:4751 @c1d0e7a00).
///
/// The allocation itself is the plain CPU buffer type's; only three of the
/// iface entries are swapped, which is what makes this a *view* of host
/// memory with a different write path rather than a second allocator.
fn buftAllocBuffer(buft: c.ggml_backend_buffer_type_t, size: usize) callconv(.c) c.ggml_backend_buffer_t {
    const buffer = c.ggml_backend_buft_alloc_buffer(c.ggml_backend_cpu_buffer_type(), size);

    if (buffer == null) return null;

    const b = impl.one(c.struct_ggml_backend_buffer, buffer);
    b.buft = buft;
    b.iface.init_tensor = bufferInitTensor;
    b.iface.set_tensor = bufferSetTensor;
    b.iface.get_tensor = null;
    b.iface.cpy_tensor = null;
    return buffer;
}

/// Ports `ggml_backend_cpu_repack_buffer_type_get_alignment`
/// (ggml-cpu/repack.cpp:4766 @c1d0e7a00).
fn buftGetAlignment(buft: c.ggml_backend_buffer_type_t) callconv(.c) usize {
    _ = buft;
    return impl.tensor_alignment;
}

// -----------------------------------------------------------------------------
// extra_buffer_type

/// Ports `extra_buffer_type` (ggml-cpu/repack.cpp:4773 @c1d0e7a00), in the
/// C's `ggml::cpu::repack` namespace.
const ExtraBuffer = struct {
    const vtable: extra.ExtraBufferType.VTable = .{
        .supports_op = supportsOp,
        .get_tensor_traits = getTensorTraits,
    };

    /// The C's `new ggml::cpu::repack::extra_buffer_type()`, which is never
    /// deleted. A static serves, since it holds no state.
    const instance: extra.ExtraBufferType = .{ .vtable = &vtable };

    /// Ports `supports_op` (ggml-cpu/repack.cpp:4774 @c1d0e7a00).
    ///
    /// The C writes two branches, `MUL_MAT` requiring a 2D `src[0]` and
    /// `MUL_MAT_ID` a 3D one, with identical bodies otherwise. They are
    /// folded here into one test on `dims`.
    fn supportsOp(self: *const extra.ExtraBufferType, dev: c.ggml_backend_dev_t, op: *const Tensor) callconv(.c) bool {
        _ = self;
        _ = dev;
        const src0 = op.src[0] orelse return false;

        const dims: c_int = if (op.op == c.GGML_OP_MUL_MAT) 2 else 3;
        if ((op.op == c.GGML_OP_MUL_MAT or op.op == c.GGML_OP_MUL_MAT_ID) and
            src0.*.buffer != null and
            c.ggml_n_dims(src0) == dims and
            src0.*.buffer.*.buft == bufferType() and
            getOptimalRepackType(src0) != null)
        {
            const src1 = op.src[1] orelse return false;
            if (src1.*.buffer != null and !c.ggml_backend_buft_is_host(src1.*.buffer.*.buft)) {
                return false;
            }
            if (src1.*.type == c.GGML_TYPE_F32) return true;
            // The C has a commented-out GGML_TYPE_Q8_0 arm here: "may be
            // possible if Q8_0 packed".
        }
        return false;
    }

    /// Ports `get_tensor_traits` (ggml-cpu/repack.cpp:4810 @c1d0e7a00).
    fn getTensorTraits(self: *const extra.ExtraBufferType, op: *const Tensor) callconv(.c) ?*const extra.TensorTraits {
        _ = self;
        if (op.op == c.GGML_OP_MUL_MAT or op.op == c.GGML_OP_MUL_MAT_ID) {
            const src0 = op.src[0] orelse return null;
            if (src0.*.buffer != null and src0.*.buffer.*.buft == bufferType()) {
                return @ptrCast(@alignCast(src0.*.extra));
            }
        }
        return null;
    }
};

/// Ports `ggml_backend_cpu_repack_buffer_type` (ggml-cpu/repack.cpp:4821
/// @c1d0e7a00).
///
/// The C's is a function-local `static` whose `.device` member is a call —
/// so it is dynamically initialised under a guard, which is the shape
/// written out here. `cpu_backend.zig` uses the same pattern for its own
/// magic statics, for the reason recorded there: Zig 0.16 has no
/// `std.once`.
var buft_storage: c.struct_ggml_backend_buffer_type = undefined;
var buft_ready = std.atomic.Value(bool).init(false);
var buft_mutex: std.c.pthread_mutex_t = .{};

fn bufferType() c.ggml_backend_buffer_type_t {
    if (!buft_ready.load(.acquire)) {
        _ = std.c.pthread_mutex_lock(&buft_mutex);
        defer _ = std.c.pthread_mutex_unlock(&buft_mutex);
        if (!buft_ready.load(.acquire)) {
            buft_storage = .{
                .iface = .{
                    .get_name = buftGetName,
                    .alloc_buffer = buftAllocBuffer,
                    .get_alignment = buftGetAlignment,
                    .get_max_size = null, // defaults to SIZE_MAX
                    .get_alloc_size = null, // defaults to ggml_nbytes
                    .is_host = null,
                },
                .device = c.ggml_backend_reg_dev_get(c.ggml_backend_cpu_reg(), 0),
                .context = @ptrCast(@constCast(&ExtraBuffer.instance)),
            };
            buft_ready.store(true, .release);
        }
    }
    return &buft_storage;
}

/// The barrel's entry point. See `bufferType`.
///
/// Return: the `CPU_REPACK` buffer type, borrowed for the life of the
/// process.
pub fn ggml_backend_cpu_repack_buffer_type() c.ggml_backend_buffer_type_t {
    return bufferType();
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "mmid_row_mapping is two int32 in an int64 slot" {
    // The C relies on this: the counts array and the row map share one
    // `int64_t`-aligned run, sized `n_as*(ne12 + 1)*sizeof(mmid_row_mapping)`.
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(MmidRowMapping));
    try std.testing.expectEqual(@as(usize, @sizeOf(i64)), @sizeOf(MmidRowMapping));
}

test "the extended traits layout starts with the vtable front" {
    // `set_tensor` recovers the `repack` entry by casting a
    // `tensor_traits *` back to the derived type, which is only sound if
    // the base is at offset zero -- the same guarantee C++ gives for a
    // non-virtual single-inheritance base.
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Extended, "base"));

    // Two instantiations over the same block type and column count, so the
    // only thing separating them is `INTER_SIZE`: distinct `repack`
    // thunks is what says the template argument really was baked in.
    const a: *const Extended = @ptrCast(&inst.q4_0_4x4_q8_0.instance);
    const b: *const Extended = @ptrCast(&inst.q4_0_4x8_q8_0.instance);
    try std.testing.expect(a.repack != b.repack);
    try std.testing.expect(a.base.vtable != b.base.vtable);
}
