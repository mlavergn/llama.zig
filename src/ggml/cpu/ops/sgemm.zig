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
//! **Five instantiations can be live**, chosen by the same feature macros
//! the C's `#if`s test:
//!
//! | A × B | class | entry condition | needs |
//! |---|---|---|---|
//! | f32 × f32 | `tinyBLAS<4, f32x4>` | `n >= 4` | NEON |
//! | f16 × f16 | `tinyBLAS<8, f16x8>` | `n >= 8` | `FP16_VECTOR_ARITHMETIC` |
//! | f16 × f32 | `tinyBLAS<4, f32x4>` | `n >= 2` | NEON without it |
//! | q8_0 × q8_0 | `tinyBLAS_Q0_ARM<block_q8_0>` | `n >= 2` | `DOTPROD` |
//! | q4_0 × q8_0 | `tinyBLAS_Q0_ARM<block_q4_0>` | `n >= 2` | `DOTPROD` |
//!
//! Every Apple core takes the f16 × f16 and the two quantized rows; Zig's
//! `aarch64-linux` baseline, `generic`, has neither feature and takes the
//! f16 × f32 row instead, through the third `load` specialisation, `f32x4`
//! from `ggml_fp16_t`. `BF16`, `Q5_0` and `IQ4_NL` reach a `return false`
//! on ARM.
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
//!
//! # x86
//!
//! At x86-64-v3 the float rows take `tinyBLAS<8, __m256>` -- f32 × f32,
//! bf16 × bf16 and, with F16C, f16 × f16, none with a minimum `n` -- through
//! the same generic `TinyBLAS` with the AVX `load`, `madd` and `hsum`, and
//! the tile constants of `VECTOR_REGISTERS == 16`. The four quantized rows
//! -- q8_0, q4_0, q5_0 and iq4_nl against q8_0 -- take `tinyBLAS_Q0_AVX`,
//! with its `__AVX2__ && __F16C__` tile kernels.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("../../impl.zig");
const common = @import("common.zig");
const defs = @import("../defs.zig");
const threading = @import("../threading.zig");
const convert = @import("../convert.zig");
const neon = @import("../simd.zig").neon;
const qblocks = @import("../../quants/blocks.zig");
const avx = @import("../quants/x86/avx2.zig");

const c = impl.c;
const ComputeParams = common.ComputeParams;
const f32x4 = neon.f32x4;
const f16x8 = neon.f16x8;
const i8x16 = neon.i8x16;

/// `__m256`.
const f32x8 = @Vector(8, f32);

/// Whether this build takes the C's x86 arms. `simd.zig` has already refused
/// an x86 level other than v3.
const is_x86 = builtin.cpu.arch == .x86_64;

/// Ports `VECTOR_REGISTERS`, `VECTOR_REGISTERS` (sgemm.cpp:67, 69 @c1d0e7a00): 32 with NEON or
/// AVX-512, 16 otherwise. It sets `tinyBLAS::matmul`'s tile constants.
const vector_registers = if (is_x86) 16 else 32;

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

/// Ports `hsum`, `hsum` (sgemm.cpp:268, 252 @c1d0e7a00), the `__m256` one and the
/// `__m128` one it calls: fold the upper 128 bits onto the lower, then
/// `_mm_movehl_ps` and `_mm_movehdup_ps`, `(h0+h2)+(h1+h3)`.
inline fn hsumAvx(x: f32x8) f32 {
    const lo: @Vector(4, f32) = .{ x[0], x[1], x[2], x[3] };
    const hi: @Vector(4, f32) = .{ x[4], x[5], x[6], x[7] };
    const h = hi + lo;
    return (h[0] + h[2]) + (h[1] + h[3]);
}

/// Ports `load`, `load`, `load` (sgemm.cpp:341, 347, 354 @c1d0e7a00), the
/// `__m256` ones: `_mm256_loadu_ps`; bf16 widened by a 16-bit shift, exactly;
/// and with F16C `_mm256_cvtph_ps`, also exact.
inline fn loadAvx(comptime T: type, p: [*]const T) f32x8 {
    if (T == f32) return p[0..8].*;
    if (T == c.ggml_bf16_t) {
        const p16: [*]const u16 = @ptrCast(p);
        const h: @Vector(8, u16) = p16[0..8].*;
        return @bitCast(@as(@Vector(8, u32), h) << @splat(16));
    }
    const h: @Vector(8, f16) = @bitCast(@as(@Vector(8, u16), p[0..8].*));
    return @floatCast(h);
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
/// `KN` is the element count of one vector load, `V` the vector type, and
/// `TA` and `TB` the scalars `A` and `B` are loaded from. The C's six
/// template parameters collapse to four because `D == V` at every live
/// instantiation and `TC` is always `f32`.
fn TinyBLAS(comptime KN: i64, comptime V: type, comptime TA: type, comptime TB: type) type {
    return struct {
        const Self = @This();

        params: *const ComputeParams,
        A: [*]const TA,
        B: [*]const TB,
        C: [*]f32,
        k: i64,
        lda: i64,
        ldb: i64,
        ldc: i64,

        /// Ports `load`, `load`, `load` (sgemm.cpp:305, 310, 313 @c1d0e7a00):
        /// `vld1q_f32`, `vld1q_f16`, and `vcvt_f32_f16(vld1_f16(p))`. The
        /// second reinterprets rather than converts -- `ggml_fp16_t` is a
        /// `u16` after the import; the third widens four halves, exactly.
        inline fn load(comptime T: type, p: [*]const T) V {
            if (V == f32x8) return loadAvx(T, p);
            if (T == f32) return p[0..4].*;
            if (V == f32x4) return neon.cvt_f32_f16(neon.load_f16x4(p));
            return @bitCast(@as(@Vector(8, u16), p[0..8].*));
        }

        /// Ports `madd`, `madd`, `madd` (sgemm.cpp:165, 170, 141 @c1d0e7a00):
        /// `vfmaq_f32(c, b, a)`, `vfmaq_f16(c, b, a)` and `_mm256_fmadd_ps(a,
        /// b, c)`, all **fused**, and all the same product.
        inline fn madd(a: V, b: V, acc: V) V {
            return @mulAdd(V, b, a, acc);
        }

        inline fn hsum(x: V) f32 {
            if (V == f32x8) return hsumAvx(x);
            return if (V == f32x4) hsum4(x) else hsum8(x);
        }

        /// Ports `matmul` (sgemm.cpp:494 @c1d0e7a00): `RN` 6 and `bn` 12 with
        /// 32 vector registers, `RN` 3 and `bn` 24 with 16.
        fn matmul(self: *const Self, m: i64, n: i64) bool {
            const RN = if (vector_registers == 32) 6 else 3;
            const bn = if (vector_registers == 32) 12 else 24;
            if (@rem(self.k, KN) != 0) return false;
            if (@rem(m, 16) == 0 and @divTrunc(m, 16) >= self.params.nth) {
                self.mnpack(4, RN, 4, m, n, blockSize(RN, n), bn);
                return true;
            }
            if (@rem(m, 8) == 0) {
                self.mnpack(4, RN, 2, m, n, blockSize(RN, n), bn);
                return true;
            }
            if (@rem(m, 4) == 0) {
                self.mnpack(4, RN, 1, m, n, blockSize(RN, n), bn);
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
            var Cv: [RN][RM]V = @splat(@splat(@as(V, @splat(0))));

            var l: i64 = 0;
            while (l < self.k) : (l += KN) {
                if (RM <= RN) {
                    var Av: [RM]V = undefined;
                    inline for (0..RM) |i| {
                        Av[i] = load(TA, self.A + @as(usize, @intCast(self.lda * (ii + @as(i64, i)) + l)));
                    }
                    inline for (0..RN) |j| {
                        const Bv = load(TB, self.B + @as(usize, @intCast(self.ldb * (jj + @as(i64, j)) + l)));
                        inline for (0..RM) |i| {
                            Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
                        }
                    }
                } else {
                    var Bv: [RN]V = undefined;
                    inline for (0..RN) |j| {
                        Bv[j] = load(TB, self.B + @as(usize, @intCast(self.ldb * (jj + @as(i64, j)) + l)));
                    }
                    inline for (0..RM) |i| {
                        const Av = load(TA, self.A + @as(usize, @intCast(self.lda * (ii + @as(i64, i)) + l)));
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
                var Cv: [RN][RM]f32x4 = @splat(@splat(@as(f32x4, @splat(0))));

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

/// Ports `tinyBLAS_Q0_AVX` (sgemm.cpp:1353 @c1d0e7a00), the x86 quantized
/// path at x86-64-v3: `__AVX2__` and `__F16C__`, `VECTOR_REGISTERS == 16`.
/// `TA` is `block_q8_0`, `block_q4_0`, `block_q5_0` or `block_iq4_nl`; `B`
/// is always `block_q8_0`.
///
/// Like `tinyBLAS_Q0_ARM` it splits its work statically, each thread
/// computing `duty` tiles of disjoint output. Its three tile kernels --
/// `gemm4xN`, `gemmMx4` and the generic `gemm` -- differ only in what they
/// hoist out of the inner loop: every output element is `hsum` of one fused
/// `scale * updot + Cv` per block, with `scale` the two blocks' deltas'
/// product. The tiling decides which thread computes an element, never its
/// value.
fn TinyBlasQ0Avx(comptime TA: type) type {
    return struct {
        const Self = @This();

        A: [*]const TA,
        B: [*]const qblocks.Q8_0,
        C: [*]f32,
        k: i64,
        lda: i64,
        ldb: i64,
        ldc: i64,
        ith: i64,
        nth: i64,

        /// Ports `denibble` (sgemm.cpp:1766 @c1d0e7a00): sixteen bytes to
        /// 32 nibbles, low first.
        inline fn denibble(p: *const [16]u8) avx.m256i {
            const x = avx.mm_loadu_si128(p);
            return avx.mm256_and_si256(avx.mm256_set1_epi8(15), avx.mm256_set_m128i(avx.mm_srli_epi16(x, 4), x));
        }

        /// Ports `bittobyte` (sgemm.cpp:1773 @c1d0e7a00): `q5_0`'s fifth
        /// bits as `0xF0` where the bit is clear, so OR-ing them in makes the
        /// nibble minus 16, signed.
        inline fn bittobyte(p: *const [4]u8) avx.m256i {
            const x32 = std.mem.readInt(u32, p, .little);
            const bytes = avx.mm256_cmpeq_epi8(
                avx.mm256_set1_epi64x(-1),
                avx.mm256_or_si256(
                    avx.mm256_set1_epi64x(0x7fbfdfeff7fbfdfe),
                    avx.mm256_shuffle_epi8(avx.mm256_set1_epi32(@bitCast(x32)), avx.mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000)),
                ),
            );
            return avx.mm256_andnot_si256(bytes, avx.mm256_set1_epi8(@bitCast(@as(u8, 0xF0))));
        }

        /// Ports `load`, `load`, `load`, `load` (sgemm.cpp:1684, 1696, 1710, 1740 @c1d0e7a00):
        /// a block's 32 weights as signed bytes. `iq4_nl`'s is `load1` over
        /// `load0`, each a `pshufb` through the codebook, which the
        /// constructor copies into `iq4nlt`.
        inline fn load(comptime T: type, b: *const T) avx.m256i {
            if (T == qblocks.Q8_0) return avx.mm256_loadu_si256(&b.qs);
            if (T == qblocks.Q4_0) return avx.mm256_sub_epi8(denibble(&b.qs), avx.mm256_set1_epi8(8));
            if (T == qblocks.Q5_0) return avx.mm256_or_si256(denibble(&b.qs), bittobyte(&b.qh));
            comptime std.debug.assert(T == qblocks.IQ4_NL);
            const iq4nlt = avx.mm_loadu_si128(&c.kvalues_iq4nl);
            const x = avx.mm_loadu_si128(&b.qs);
            const lo = avx.mm_shuffle_epi8(iq4nlt, avx.mm_and_si128(avx.mm_set1_epi8(15), x));
            const hi = avx.mm_shuffle_epi8(iq4nlt, avx.mm_and_si128(avx.mm_set1_epi8(15), avx.mm_srli_epi16(x, 4)));
            return avx.mm256_set_m128i(hi, lo);
        }

        /// Ports `updot` (sgemm.cpp:1754 @c1d0e7a00), the arm without VNNI:
        /// unsigned by signed bytes, summed in pairs twice, as floats.
        inline fn updot(u: avx.m256i, s: avx.m256i) avx.m256 {
            const res = avx.mm256_madd_epi16(avx.mm256_set1_epi16(1), avx.mm256_maddubs_epi16(u, s));
            return avx.mm256_cvtepi32_ps(res);
        }

        /// The C's `updot(_mm256_sign_epi8(a, a), _mm256_sign_epi8(b, a))`:
        /// a signed dot product through the unsigned instruction.
        inline fn dot(av: avx.m256i, bv: avx.m256i) avx.m256 {
            return updot(avx.mm256_sign_epi8(av, av), avx.mm256_sign_epi8(bv, av));
        }

        inline fn aAt(self: *const Self, row: i64, l: i64) *const TA {
            return &self.A[@intCast(self.lda * row + l)];
        }

        inline fn bAt(self: *const Self, col: i64, l: i64) *const qblocks.Q8_0 {
            return &self.B[@intCast(self.ldb * col + l)];
        }

        /// Ports `matmul` (sgemm.cpp:1371 @c1d0e7a00).
        fn matmul(self: *const Self, m: i64, n: i64) void {
            self.mnpack(0, m, 0, n);
        }

        /// Ports `mnpack` (sgemm.cpp:1376 @c1d0e7a00), the
        /// `VECTOR_REGISTERS == 16` cases with their `__AVX2__ && __F16C__`
        /// tile kernels: 4x2 for `0x44`, `0x43` and `0x42`, 2x4 for `0x34`
        /// and `0x24`, and `0x33` falling through to 3x2.
        fn mnpack(self: *const Self, m0: i64, m: i64, n0: i64, n: i64) void {
            const mc: i64, const nc: i64 = blk: {
                const sel = (@min(m - m0, 4) << 4) | @min(n - n0, 4);
                switch (sel) {
                    0x44, 0x43, 0x42 => {
                        self.gemm4xN(2, m0, m, n0, n);
                        break :blk .{ 4, 2 };
                    },
                    0x34, 0x24 => {
                        self.gemmMx4(2, m0, m, n0, n);
                        break :blk .{ 2, 4 };
                    },
                    0x33, 0x32 => {
                        self.gemm(3, 2, m0, m, n0, n);
                        break :blk .{ 3, 2 };
                    },
                    0x23 => {
                        self.gemm(2, 3, m0, m, n0, n);
                        break :blk .{ 2, 3 };
                    },
                    0x41 => {
                        self.gemm4xN(1, m0, m, n0, n);
                        break :blk .{ 4, 1 };
                    },
                    0x22 => {
                        self.gemm(2, 2, m0, m, n0, n);
                        break :blk .{ 2, 2 };
                    },
                    0x14 => {
                        self.gemmMx4(1, m0, m, n0, n);
                        break :blk .{ 1, 4 };
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

        /// The static split every tile kernel opens with.
        inline fn duty(self: *const Self, tiles: i64) struct { i64, i64 } {
            const d = @divTrunc(tiles + self.nth - 1, self.nth);
            const start = d * self.ith;
            var end = start + d;
            if (end > tiles) end = tiles;
            return .{ start, end };
        }

        /// Four halves to floats with F16C, as `_mm_cvtph_ps` over the
        /// packed `uint64_t` the C builds.
        inline fn cvtph4(h0: u16, h1: u16, h2: u16, h3: u16) @Vector(4, f32) {
            const h: @Vector(4, f16) = @bitCast(@Vector(4, u16){ h0, h1, h2, h3 });
            return @floatCast(h);
        }

        /// Ports `gemm4xN` (sgemm.cpp:1524 @c1d0e7a00): four rows of `A`,
        /// their deltas converted together, against `RN` columns.
        fn gemm4xN(self: *const Self, comptime RN: i64, m0: i64, m: i64, n0: i64, n: i64) void {
            const ytiles = @divTrunc(m - m0, 4);
            const xtiles = @divTrunc(n - n0, RN);
            const start, const end = self.duty(xtiles * ytiles);
            var job = start;
            while (job < end) : (job += 1) {
                const ii = m0 + @divTrunc(job, xtiles) * 4;
                const jj = n0 + @rem(job, xtiles) * RN;
                var Cv: [RN][4]avx.m256 = @splat(@splat(avx.mm256_setzero_ps()));
                var l: i64 = 0;
                while (l < self.k) : (l += 1) {
                    // Convert delta values for four blocks to float values
                    const da = cvtph4(self.aAt(ii + 0, l).d, self.aAt(ii + 1, l).d, self.aAt(ii + 2, l).d, self.aAt(ii + 3, l).d);
                    var avec: [4]avx.m256i = undefined;
                    inline for (0..4) |i| avec[i] = load(TA, self.aAt(ii + i, l));
                    inline for (0..RN) |j| {
                        const b = self.bAt(jj + j, l);
                        const db: @Vector(4, f32) = @splat(unhalf(b.d));
                        // The four delta products, each broadcast across
                        // the vector by `_mm256_shuffle_ps`
                        const dvec = da * db;
                        inline for (0..4) |i| {
                            Cv[j][i] = avx.mm256_fmadd_ps(@splat(dvec[i]), dot(avec[i], load(qblocks.Q8_0, b)), Cv[j][i]);
                        }
                    }
                }
                inline for (0..RN) |j| {
                    inline for (0..4) |i| {
                        self.C[@intCast(self.ldc * (jj + j) + (ii + i))] = hsumAvx(Cv[j][i]);
                    }
                }
            }
        }

        /// Ports `gemmMx4` (sgemm.cpp:1578 @c1d0e7a00): `RM` rows against
        /// four columns of `B`, the columns' deltas converted together.
        fn gemmMx4(self: *const Self, comptime RM: i64, m0: i64, m: i64, n0: i64, n: i64) void {
            const ytiles = @divTrunc(m - m0, RM);
            const xtiles = @divTrunc(n - n0, 4);
            const start, const end = self.duty(xtiles * ytiles);
            var job = start;
            while (job < end) : (job += 1) {
                const ii = m0 + @divTrunc(job, xtiles) * RM;
                const jj = n0 + @rem(job, xtiles) * 4;
                var Cv: [4][RM]avx.m256 = @splat(@splat(avx.mm256_setzero_ps()));
                var l: i64 = 0;
                while (l < self.k) : (l += 1) {
                    // Convert delta values for four blocks to float values
                    const db = cvtph4(self.bAt(jj + 0, l).d, self.bAt(jj + 1, l).d, self.bAt(jj + 2, l).d, self.bAt(jj + 3, l).d);
                    var bvec: [4]avx.m256i = undefined;
                    inline for (0..4) |j| bvec[j] = load(qblocks.Q8_0, self.bAt(jj + j, l));
                    inline for (0..RM) |i| {
                        const a = self.aAt(ii + i, l);
                        const da: @Vector(4, f32) = @splat(unhalf(a.d));
                        const dvec = da * db;
                        inline for (0..4) |j| {
                            Cv[j][i] = avx.mm256_fmadd_ps(@splat(dvec[j]), dot(load(TA, a), bvec[j]), Cv[j][i]);
                        }
                    }
                }
                inline for (0..4) |j| {
                    inline for (0..RM) |i| {
                        self.C[@intCast(self.ldc * (jj + j) + (ii + i))] = hsumAvx(Cv[j][i]);
                    }
                }
            }
        }

        /// Ports `gemm` (sgemm.cpp:1635 @c1d0e7a00), its `__AVX2__` arm.
        fn gemm(self: *const Self, comptime RM: i64, comptime RN: i64, m0: i64, m: i64, n0: i64, n: i64) void {
            const ytiles = @divTrunc(m - m0, RM);
            const xtiles = @divTrunc(n - n0, RN);
            const start, const end = self.duty(xtiles * ytiles);
            var job = start;
            while (job < end) : (job += 1) {
                const ii = m0 + @divTrunc(job, xtiles) * RM;
                const jj = n0 + @rem(job, xtiles) * RN;
                var Cv: [RN][RM]avx.m256 = @splat(@splat(avx.mm256_setzero_ps()));
                var l: i64 = 0;
                while (l < self.k) : (l += 1) {
                    inline for (0..RN) |j| {
                        inline for (0..RM) |i| {
                            const a = self.aAt(ii + i, l);
                            const b = self.bAt(jj + j, l);
                            const udTmp = dot(load(TA, a), load(qblocks.Q8_0, b));
                            Cv[j][i] = avx.mm256_fmadd_ps(@splat(unhalf(a.d) * unhalf(b.d)), udTmp, Cv[j][i]);
                        }
                    }
                }
                inline for (0..RN) |j| {
                    inline for (0..RM) |i| {
                        self.C[@intCast(self.ldc * (jj + j) + (ii + i))] = hsumAvx(Cv[j][i]);
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

    if (comptime is_x86) return sgemmAvx(params, m, n, k, A, lda, B, ldb, C, ldc, Atype, Btype);

    switch (Atype) {
        c.GGML_TYPE_F32 => {
            if (Btype != c.GGML_TYPE_F32) return false;
            if (n < 4) return false;
            const tb = TinyBLAS(4, f32x4, f32, f32){
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
        // The `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` arm, then the plain
        // `__ARM_NEON` one (sgemm.cpp:3974, 3984 @c1d0e7a00).
        c.GGML_TYPE_F16 => if (comptime neon.has_fp16_va) {
            if (n < 8) return false;
            if (Btype != c.GGML_TYPE_F16) return false;
            const tb = TinyBLAS(8, f16x8, c.ggml_fp16_t, c.ggml_fp16_t){
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
        } else {
            if (Btype != c.GGML_TYPE_F32) return false;
            const tb = TinyBLAS(4, f32x4, c.ggml_fp16_t, f32){
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
        // Both quantized arms run `tinyBLAS_Q0_ARM` only behind
        // `__ARM_FEATURE_DOTPROD` (sgemm.cpp:4052 @c1d0e7a00); without it
        // each is the `#else`, which returns false.
        c.GGML_TYPE_Q8_0 => {
            if (comptime !neon.has_dotprod) return false;
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
            if (comptime !neon.has_dotprod) return false;
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

/// One `tinyBLAS_Q0_AVX` row of the switch: `B` must be `q8_0`.
fn q0(comptime TA: type, params: *const ComputeParams, m: i64, n: i64, k: i64, A: ?*const anyopaque, lda: i64, B: ?*const anyopaque, ldb: i64, C: ?*anyopaque, ldc: i64, Btype: c_int) bool {
    if (Btype != c.GGML_TYPE_Q8_0) return false;
    const tb = TinyBlasQ0Avx(TA){
        .A = @ptrCast(@alignCast(A.?)),
        .B = @ptrCast(@alignCast(B.?)),
        .C = @ptrCast(@alignCast(C.?)),
        .k = k,
        .lda = lda,
        .ldb = ldb,
        .ldc = ldc,
        .ith = params.ith,
        .nth = params.nth,
    };
    tb.matmul(m, n);
    return true;
}

/// `llamafile_sgemm`'s switch at x86-64-v3: the `__AVX__ || __AVX2__` f32
/// arm, the `__AVX2__` bf16 arm and the F16C f16 arm -- the `#elif`s at
/// lines 3838, 3910 and 3966 of `sgemm.cpp` -- then the four
/// `tinyBLAS_Q0_AVX` rows.
fn sgemmAvx(
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
) bool {
    const float = struct {
        fn run(comptime TA: type, comptime TB: type, p: *const ComputeParams, mm: i64, nn: i64, kk: i64, a: ?*const anyopaque, la: i64, b: ?*const anyopaque, lb: i64, cc: ?*anyopaque, lc: i64) bool {
            const tb = TinyBLAS(8, f32x8, TA, TB){
                .params = p,
                .k = kk,
                .A = @ptrCast(@alignCast(a.?)),
                .lda = la,
                .B = @ptrCast(@alignCast(b.?)),
                .ldb = lb,
                .C = @ptrCast(@alignCast(cc.?)),
                .ldc = lc,
            };
            return tb.matmul(mm, nn);
        }
    }.run;
    switch (Atype) {
        c.GGML_TYPE_F32 => {
            if (Btype != c.GGML_TYPE_F32) return false;
            return float(f32, f32, params, m, n, k, A, lda, B, ldb, C, ldc);
        },
        c.GGML_TYPE_BF16 => {
            if (Btype != c.GGML_TYPE_BF16) return false;
            return float(c.ggml_bf16_t, c.ggml_bf16_t, params, m, n, k, A, lda, B, ldb, C, ldc);
        },
        c.GGML_TYPE_F16 => {
            if (Btype != c.GGML_TYPE_F16) return false;
            return float(c.ggml_fp16_t, c.ggml_fp16_t, params, m, n, k, A, lda, B, ldb, C, ldc);
        },
        // sgemm.cpp:4041, 4078, 4115, 4131 @c1d0e7a00: each `__AVX2__` arm
        // takes `tinyBLAS_Q0_AVX` against `q8_0`, with no minimum `n`.
        c.GGML_TYPE_Q8_0 => return q0(qblocks.Q8_0, params, m, n, k, A, lda, B, ldb, C, ldc, Btype),
        c.GGML_TYPE_Q4_0 => return q0(qblocks.Q4_0, params, m, n, k, A, lda, B, ldb, C, ldc, Btype),
        c.GGML_TYPE_Q5_0 => return q0(qblocks.Q5_0, params, m, n, k, A, lda, B, ldb, C, ldc, Btype),
        c.GGML_TYPE_IQ4_NL => return q0(qblocks.IQ4_NL, params, m, n, k, A, lda, B, ldb, C, ldc, Btype),
        else => return false,
    }
}
