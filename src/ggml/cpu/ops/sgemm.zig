//! `llamafile_sgemm`, the tinyBLAS matrix-multiply fast path.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration below names the C++ it replaces and the
//! line it began at.
//!
//! # Only a tenth of the file compiles here
//!
//! 391 live lines of 4,164: the templated x86 bulk is behind `__AVX__`,
//! `__AVX512F__` and `__AVX2__`. `PLAN.md` once called this "the awkward
//! one" on the raw count, which the measurement contradicted.
//!
//! **Four instantiations are live**, and nothing else:
//!
//! | A × B | class | entry condition |
//! |---|---|---|
//! | f32 × f32 | `tinyBLAS<4, f32x4>` | `n >= 4` |
//! | f16 × f16 | `tinyBLAS<8, f16x8>` | `n >= 8` |
//! | q8_0 × q8_0 | `tinyBLAS_Q0_ARM<block_q8_0>` | `n >= 2` |
//! | q4_0 × q8_0 | `tinyBLAS_Q0_ARM<block_q4_0>` | `n >= 2` |
//!
//! `BF16`, `Q5_0` and `IQ4_NL` reach a `return false` on this target. The
//! third `load` specialisation (`f32x4` from `ggml_fp16_t`) serves an arm
//! that is dead here, so it is not ported.
//!
//! # The one symbol
//!
//! `llamafile_sgemm` returns **false** when it has no kernel for the shape
//! or type pair, and the caller — `cpu/mulmat.zig` — then computes the
//! product itself. Returning false is a normal outcome, not an error.
//!
//! # Fused multiplies
//!
//! Every accumulation here is an explicit FMA intrinsic in the C:
//! `vfmaq_f32`, `vfmaq_f16`, `vmlaq_n_f32`. They are named with `@mulAdd`
//! rather than written as `a*b + c`, which would round twice. The generic
//! `madd` (sgemm.cpp:134 @c1d0e7a00) *is* two roundings, but both live element types
//! have an FMA specialisation, so it is never instantiated here.
//!
//! `hsum` is `vaddvq_f32`, which is **pairwise** — see `CLAUDE.md`,
//! "Porting notes".

const std = @import("std");
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const defs = @import("../defs.zig");
const threading = @import("../threading.zig");
const convert = @import("../convert.zig");
const neon = @import("../quants/arm/neon.zig");

const c = impl.c;
const ComputeParams = common.ComputeParams;
const f32x4 = neon.f32x4;
const f16x8 = neon.f16x8;
const i8x16 = neon.i8x16;

/// Ports `unhalf` (sgemm.cpp:80 @c1d0e7a00).
///
/// `GGML_CPU_FP16_TO_FP32`, which on NEON is the hardware conversion the
/// other ported ARM kernels reach through `convert.cpuFp16ToFp32`.
inline fn unhalf(d: c.ggml_fp16_t) f32 {
    return convert.cpuFp16ToFp32(d);
}

/// Ports `hsum` (sgemm.cpp:232 @c1d0e7a00): `vaddvq_f32`, a
/// **pairwise** reduce, not the ordered one `@reduce(.Add, ...)` gives.
inline fn hsum4(x: f32x4) f32 {
    return neon.addvq_f32(x);
}

/// Ports `hsum` (sgemm.cpp:238 @c1d0e7a00): widen both halves
/// to `f32x4`, add, then the pairwise reduce.
inline fn hsum8(x: f16x8) f32 {
    return neon.addvq_f32(neon.cvt_f32_f16_half(x, false) + neon.cvt_f32_f16_half(x, true));
}

/// Ports `BLOCK_SIZE` (sgemm.cpp:475 @c1d0e7a00).
inline fn blockSize(comptime M: i64, m: i64) i64 {
    const nb_bloc_m = @divTrunc(m + M - 1, M);
    return if (@rem(m, nb_bloc_m) == 0) @divTrunc(m, nb_bloc_m) else @divTrunc(m, nb_bloc_m) + 1;
}

/// Ports `BLOC_POS` (sgemm.cpp:480 @c1d0e7a00).
inline fn blocPos(ib: i64, ibN: i64, bloc_size: i64) i64 {
    return if (ib < ibN) ib * bloc_size else ibN * bloc_size + (ib - ibN) * (bloc_size - 1);
}

/// Ports `tinyBLAS` (sgemm.cpp:485 @c1d0e7a00), the float path.
///
/// `KN` is the element count of one vector load, `V` the vector type and
/// `T` the scalar it is loaded from. The C's six template parameters
/// collapse to three because `D == V` and `TA == TB == T` at both live
/// instantiations, and `TC` is always `f32`.
fn TinyBLAS(comptime KN: i64, comptime V: type, comptime T: type) type {
    return struct {
        const Self = @This();

        params: *const ComputeParams,
        A: [*]const T,
        B: [*]const T,
        C: [*]f32,
        k: i64,
        lda: i64,
        ldb: i64,
        ldc: i64,

        /// Ports `load`, `load` (sgemm.cpp:305, 310 @c1d0e7a00): `vld1q_f32` and
        /// `vld1q_f16`. The f16 one reinterprets rather than converts --
        /// `ggml_fp16_t` is a `u16` after the import.
        inline fn load(p: [*]const T) V {
            if (T == f32) return p[0..4].*;
            return @bitCast(@as(@Vector(8, u16), p[0..8].*));
        }

        /// Ports `madd`, `madd` (sgemm.cpp:165, 170 @c1d0e7a00): `vfmaq_f32(c, b, a)`
        /// and `vfmaq_f16(c, b, a)`, both **fused**.
        inline fn madd(a: V, b: V, acc: V) V {
            return @mulAdd(V, b, a, acc);
        }

        inline fn hsum(x: V) f32 {
            return if (V == f32x4) hsum4(x) else hsum8(x);
        }

        /// Ports `matmul` (sgemm.cpp:494 @c1d0e7a00).
        fn matmul(self: *const Self, m: i64, n: i64) bool {
            if (@rem(self.k, KN) != 0) return false;
            if (@rem(m, 16) == 0 and @divTrunc(m, 16) >= self.params.nth) {
                self.mnpack(4, 6, 4, m, n, blockSize(6, n), 12);
                return true;
            }
            if (@rem(m, 8) == 0) {
                self.mnpack(4, 6, 2, m, n, blockSize(6, n), 12);
                return true;
            }
            if (@rem(m, 4) == 0) {
                self.mnpack(4, 6, 1, m, n, blockSize(6, n), 12);
                return true;
            }
            return false;
        }

        /// Ports `mnpack` (sgemm.cpp:536 @c1d0e7a00).
        ///
        /// The C recurses on the template parameter `RN` down to 1, picking
        /// the arm where `RN == SIZE_N`. Here the recursion is an `inline
        /// for` over the same descending range, which Zig unrolls the same
        /// way; `RM` and `BM` never change across it.
        inline fn mnpack(
            self: *const Self,
            comptime RM: i64,
            comptime RN_max: i64,
            comptime BM: i64,
            m: i64,
            n: i64,
            size_n: i64,
            bn: i64,
        ) void {
            comptime var RN: i64 = RN_max;
            inline while (RN >= 1) : (RN -= 1) {
                if (size_n == RN) return self.gemm(RM, RN, BM, m, n, bn);
            }
            impl.logError("mnpack block size not supported\n", .{});
            impl.abort("false");
        }

        /// Ports `gemm_bloc` (sgemm.cpp:549 @c1d0e7a00).
        ///
        /// The `if constexpr (RM <= RN)` arms differ in which operand is
        /// held in registers; both are reproduced because the multiply
        /// order they feed `madd` differs, and these are FMAs.
        inline fn gemmBloc(self: *const Self, comptime RM: i64, comptime RN: i64, ii: i64, jj: i64) void {
            var Cv: [RN][RM]V = .{.{@as(V, @splat(0))} ** RM} ** RN;

            var l: i64 = 0;
            while (l < self.k) : (l += KN) {
                if (RM <= RN) {
                    var Av: [RM]V = undefined;
                    inline for (0..RM) |i| {
                        Av[i] = load(self.A + @as(usize, @intCast(self.lda * (ii + @as(i64, i)) + l)));
                    }
                    inline for (0..RN) |j| {
                        const Bv = load(self.B + @as(usize, @intCast(self.ldb * (jj + @as(i64, j)) + l)));
                        inline for (0..RM) |i| {
                            Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
                        }
                    }
                } else {
                    var Bv: [RN]V = undefined;
                    inline for (0..RN) |j| {
                        Bv[j] = load(self.B + @as(usize, @intCast(self.ldb * (jj + @as(i64, j)) + l)));
                    }
                    inline for (0..RM) |i| {
                        const Av = load(self.A + @as(usize, @intCast(self.lda * (ii + @as(i64, i)) + l)));
                        inline for (0..RN) |j| {
                            Cv[j][i] = madd(Av, Bv[j], Cv[j][i]);
                        }
                    }
                }
            }
            inline for (0..RN) |j| {
                inline for (0..RM) |i| {
                    self.C[@intCast(self.ldc * (jj + @as(i64, j)) + (ii + @as(i64, i)))] = hsum(Cv[j][i]);
                }
            }
        }

        /// Ports `gemm` (sgemm.cpp:583 @c1d0e7a00).
        ///
        /// The work is handed out through the threadpool's chunk counter,
        /// not a static split, so the row range a thread gets depends on how
        /// fast the others run. The two barriers are the C's.
        fn gemm(
            self: *const Self,
            comptime RM: i64,
            comptime RN: i64,
            comptime BM: i64,
            m: i64,
            n: i64,
            bn: i64,
        ) void {
            impl.assert(@rem(m, RM * BM) == 0, "m % (RM * BM) == 0");
            const ytiles = @divTrunc(m, RM * BM);
            const xtiles = @divTrunc(n + RN - 1, RN);
            const jj_RN = xtiles - (xtiles * RN - n);
            const NB_BN = if (xtiles < bn) 1 else @divTrunc(xtiles + @divTrunc(bn, 2), bn);
            const SIZE_BN = if (@rem(xtiles, NB_BN) == 0) @divTrunc(xtiles, NB_BN) else @divTrunc(xtiles, NB_BN) + 1;
            const jj_BN = NB_BN - (NB_BN * SIZE_BN - xtiles);
            const nb_job = ytiles * NB_BN;

            const tp: *defs.Threadpool = @ptrCast(@alignCast(self.params.threadpool.?));
            if (self.params.ith == 0) {
                impl.assert(jj_BN * SIZE_BN + (NB_BN - jj_BN) * (SIZE_BN - 1) == xtiles, "jj_BN * SIZE_BN + (NB_BN - jj_BN) * (SIZE_BN - 1) == xtiles");
                threading.ggml_threadpool_chunk_set(tp, self.params.nth);
            }
            threading.ggml_barrier(tp);

            var job: i64 = self.params.ith;
            while (job < nb_job) {
                const ii = @rem(job, ytiles) * RM * BM;
                const jb = @divTrunc(job, ytiles);
                const jr0 = blocPos(jb, jj_BN, SIZE_BN);
                const jrN = blocPos(jb + 1, jj_BN, SIZE_BN);
                const jj0 = blocPos(jr0, jj_RN, RN);
                const jj2 = blocPos(jrN, jj_RN, RN);
                const jj1 = if (jj2 < jj_RN * RN) jj2 else jj_RN * RN;

                var bi: i64 = 0;
                while (bi < BM * RM) : (bi += RM) {
                    var jj = jj0;
                    while (jj < jj1) : (jj += RN) {
                        self.gemmBloc(RM, RN, ii + bi, jj);
                    }
                    if (RN > 1) {
                        while (jj < jj2) : (jj += RN - 1) {
                            self.gemmBloc(RM, RN - 1, ii + bi, jj);
                        }
                    }
                    impl.assert(jj == jj2, "jj == jj2");
                }
                job = threading.ggml_threadpool_chunk_add(tp, 1);
            }
            threading.ggml_barrier(tp);
        }
    };
}

/// Ports `tinyBLAS_Q0_ARM` (sgemm.cpp:1216 @c1d0e7a00), the quantized
/// path. `TA` is `block_q8_0` or `block_q4_0`; `B` is always `block_q8_0`.
///
/// Unlike `tinyBLAS` this one splits its work statically and takes no
/// barriers: each thread computes `duty` tiles and writes disjoint output.
fn TinyBlasQ0Arm(comptime TA: type) type {
    return struct {
        const Self = @This();

        A: [*]const TA,
        B: [*]const c.block_q8_0,
        C: [*]f32,
        k: i64,
        lda: i64,
        ldb: i64,
        ldc: i64,
        ith: i64,
        nth: i64,

        /// For `block_q8_0`, ports `load_lo` and `load_hi`
        /// (sgemm.cpp:1320, 1324 @c1d0e7a00): the two halves of a 32-byte
        /// `qs`, loaded as-is.
        inline fn loadLoQ8(b: *const c.block_q8_0) i8x16 {
            return @bitCast(@as(@Vector(16, i8), b.qs[0..16].*));
        }
        inline fn loadHiQ8(b: *const c.block_q8_0) i8x16 {
            return @bitCast(@as(@Vector(16, i8), b.qs[16..32].*));
        }

        /// For `block_q4_0`, ports `load_lo` and `load_hi`
        /// (sgemm.cpp:1328, 1334 @c1d0e7a00): one 16-byte load, masked low
        /// or shifted high, then the `-8` bias both nibbles carry.
        ///
        /// **NEON integer arithmetic wraps**, so the subtraction is `-%`.
        inline fn loadLoQ4(b: *const c.block_q4_0) i8x16 {
            const raw: @Vector(16, u8) = b.qs[0..16].*;
            const lo: @Vector(16, i8) = @bitCast(raw & @as(@Vector(16, u8), @splat(0x0f)));
            return lo -% @as(@Vector(16, i8), @splat(8));
        }
        inline fn loadHiQ4(b: *const c.block_q4_0) i8x16 {
            const raw: @Vector(16, u8) = b.qs[0..16].*;
            const hi: @Vector(16, i8) = @bitCast(raw >> @as(@Vector(16, u3), @splat(4)));
            return hi -% @as(@Vector(16, i8), @splat(8));
        }

        inline fn loadLo(b: *const TA) i8x16 {
            return if (TA == c.block_q8_0) loadLoQ8(b) else loadLoQ4(b);
        }
        inline fn loadHi(b: *const TA) i8x16 {
            return if (TA == c.block_q8_0) loadHiQ8(b) else loadHiQ4(b);
        }

        /// Ports `matmul` (sgemm.cpp:1226 @c1d0e7a00).
        fn matmul(self: *const Self, m: i64, n: i64) void {
            self.mnpack(0, m, 0, n);
        }

        /// Ports `mnpack` (sgemm.cpp:1231 @c1d0e7a00).
        ///
        /// The C switches on `min(m-m0,3) << 4 | min(n-n0,3)` to pick a tile
        /// shape, then recurses over the two leftover strips. The switch is
        /// written out the same way so the tile order — which decides which
        /// thread computes what — is identical.
        fn mnpack(self: *const Self, m0: i64, m: i64, n0: i64, n: i64) void {
            const mc: i64, const nc: i64 = blk: {
                const sel = (@min(m - m0, 3) << 4) | @min(n - n0, 3);
                switch (sel) {
                    0x33 => {
                        self.gemm(3, 3, m0, m, n0, n);
                        break :blk .{ 3, 3 };
                    },
                    0x32 => {
                        self.gemm(3, 2, m0, m, n0, n);
                        break :blk .{ 3, 2 };
                    },
                    0x23 => {
                        self.gemm(2, 3, m0, m, n0, n);
                        break :blk .{ 2, 3 };
                    },
                    0x22 => {
                        self.gemm(2, 2, m0, m, n0, n);
                        break :blk .{ 2, 2 };
                    },
                    0x31 => {
                        self.gemm(3, 1, m0, m, n0, n);
                        break :blk .{ 3, 1 };
                    },
                    0x13 => {
                        self.gemm(1, 3, m0, m, n0, n);
                        break :blk .{ 1, 3 };
                    },
                    0x21 => {
                        self.gemm(2, 1, m0, m, n0, n);
                        break :blk .{ 2, 1 };
                    },
                    0x12 => {
                        self.gemm(1, 2, m0, m, n0, n);
                        break :blk .{ 1, 2 };
                    },
                    0x11 => {
                        self.gemm(1, 1, m0, m, n0, n);
                        break :blk .{ 1, 1 };
                    },
                    else => return,
                }
            };
            const mp = m0 + @divTrunc(m - m0, mc) * mc;
            const np = n0 + @divTrunc(n - n0, nc) * nc;
            self.mnpack(mp, m, n0, np);
            self.mnpack(m0, m, np, n);
        }

        /// Ports `gemm` (sgemm.cpp:1289 @c1d0e7a00).
        ///
        /// The accumulation is `vmlaq_n_f32`, which is **fused**: one
        /// rounding of `Cv + scale * dot`, not a separate multiply and add.
        /// `vdotq_s32` is integer, so the widening-product form `neon.dotq_s32`
        /// uses is bit-identical rather than merely close.
        fn gemm(self: *const Self, comptime RM: i64, comptime RN: i64, m0: i64, m: i64, n0: i64, n: i64) void {
            const ytiles = @divTrunc(m - m0, RM);
            const xtiles = @divTrunc(n - n0, RN);
            const tiles = xtiles * ytiles;
            const duty = @divTrunc(tiles + self.nth - 1, self.nth);
            const start = duty * self.ith;
            var end = start + duty;
            if (end > tiles) end = tiles;

            var job: i64 = start;
            while (job < end) : (job += 1) {
                const ii = m0 + @divTrunc(job, xtiles) * RM;
                const jj = n0 + @rem(job, xtiles) * RN;
                var Cv: [RN][RM]f32x4 = .{.{@as(f32x4, @splat(0))} ** RM} ** RN;

                var l: i64 = 0;
                while (l < self.k) : (l += 1) {
                    inline for (0..RN) |j| {
                        inline for (0..RM) |i| {
                            const a = &self.A[@intCast(self.lda * (ii + @as(i64, i)) + l)];
                            const b = &self.B[@intCast(self.ldb * (jj + @as(i64, j)) + l)];
                            const dot = neon.dotq_s32(
                                neon.dotq_s32(@splat(0), loadLo(a), loadLoQ8(b)),
                                loadHi(a),
                                loadHiQ8(b),
                            );
                            Cv[j][i] = neon.mla_n_f32(
                                Cv[j][i],
                                neon.cvt_f32_s32(dot),
                                unhalf(a.d) * unhalf(b.d),
                            );
                        }
                    }
                }
                inline for (0..RN) |j| {
                    inline for (0..RM) |i| {
                        self.C[@intCast(self.ldc * (jj + @as(i64, j)) + (ii + @as(i64, i)))] = hsum4(Cv[j][i]);
                    }
                }
            }
        }
    };
}

/// Ports `llamafile_sgemm` (sgemm.cpp:3805 @c1d0e7a00).
///
/// Returns **false** when there is no kernel for this shape or type pair,
/// and `cpu/mulmat.zig` then computes the product itself. On this target
/// `BF16`, `Q5_0` and `IQ4_NL` always take that path.
///
/// Parameters:
/// - `params`: thread index, count and threadpool.
/// - `m`, `n`, `k`: the product's dimensions.
/// - `A`, `lda`, `B`, `ldb`, `C`, `ldc`: operands and their leading strides.
/// - `Atype`, `Btype`, `Ctype`: `ggml_type` of each operand.
///
/// Return: true if the product was computed here.
pub export fn llamafile_sgemm(
    params: *const ComputeParams,
    m: i64,
    n: i64,
    k: i64,
    A: ?*const anyopaque,
    lda: i64,
    B: ?*const anyopaque,
    ldb: i64,
    C: ?*anyopaque,
    ldc: i64,
    Atype: c_int,
    Btype: c_int,
    Ctype: c_int,
) callconv(.c) bool {
    if (n < 2) return false;
    if (Ctype != c.GGML_TYPE_F32) return false;

    switch (Atype) {
        c.GGML_TYPE_F32 => {
            if (Btype != c.GGML_TYPE_F32) return false;
            if (n < 4) return false;
            const tb = TinyBLAS(4, f32x4, f32){
                .params = params,
                .k = k,
                .A = @ptrCast(@alignCast(A.?)),
                .lda = lda,
                .B = @ptrCast(@alignCast(B.?)),
                .ldb = ldb,
                .C = @ptrCast(@alignCast(C.?)),
                .ldc = ldc,
            };
            return tb.matmul(m, n);
        },
        c.GGML_TYPE_F16 => {
            if (n < 8) return false;
            if (Btype != c.GGML_TYPE_F16) return false;
            const tb = TinyBLAS(8, f16x8, c.ggml_fp16_t){
                .params = params,
                .k = k,
                .A = @ptrCast(@alignCast(A.?)),
                .lda = lda,
                .B = @ptrCast(@alignCast(B.?)),
                .ldb = ldb,
                .C = @ptrCast(@alignCast(C.?)),
                .ldc = ldc,
            };
            return tb.matmul(m, n);
        },
        c.GGML_TYPE_Q8_0 => {
            if (Btype != c.GGML_TYPE_Q8_0) return false;
            const tb = TinyBlasQ0Arm(c.block_q8_0){
                .k = k,
                .A = @ptrCast(@alignCast(A.?)),
                .lda = lda,
                .B = @ptrCast(@alignCast(B.?)),
                .ldb = ldb,
                .C = @ptrCast(@alignCast(C.?)),
                .ldc = ldc,
                .ith = params.ith,
                .nth = params.nth,
            };
            tb.matmul(m, n);
            return true;
        },
        c.GGML_TYPE_Q4_0 => {
            if (Btype != c.GGML_TYPE_Q8_0) return false;
            const tb = TinyBlasQ0Arm(c.block_q4_0){
                .k = k,
                .A = @ptrCast(@alignCast(A.?)),
                .lda = lda,
                .B = @ptrCast(@alignCast(B.?)),
                .ldb = ldb,
                .C = @ptrCast(@alignCast(C.?)),
                .ldc = ldc,
                .ith = params.ith,
                .nth = params.nth,
            };
            tb.matmul(m, n);
            return true;
        },
        // The `GGML_TYPE_BF16`, `GGML_TYPE_Q5_0` and `GGML_TYPE_IQ4_NL` arms
        // (sgemm.cpp:3893, 4115, 4131 @c1d0e7a00) all reach a
        // `return false` on this target: their arms are behind x86 or SVE
        // feature macros this build does not define.
        else => return false,
    }
}
