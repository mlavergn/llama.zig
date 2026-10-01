// Capture the reference behaviour of the three float dot products in
// ggml-cpu/vec.cpp: `ggml_vec_dot_f32`, `ggml_vec_dot_f16` and
// `ggml_vec_dot_bf16`.
//
// Emits a Zig source file of golden values: the exact bits of the float each
// kernel writes, for each of several input patterns.
//
// Why these need goldens
// ----------------------
// They are accumulating reductions, so the answer depends on the accumulator
// structure and the summation order, not just the arithmetic. Measured, the
// three differ from each other:
//
//   f32   four NEON accumulators, FMA per step, tree reduction (offset 2 then
//         1), finishing with `vaddvq_f32` -- which is a *pairwise* reduce, not
//         the ordered sum `@reduce(.Add, ...)` gives -- then a scalar f32 tail.
//   f16   eight-wide *half precision* accumulators (`vaddq_f16`), reduced
//         through f32, with a `ggml_float` (double) scalar tail.
//   bf16  no SIMD arm on this target at all: a plain double accumulation.
//
// Nothing else in this project would catch getting one of those wrong.
// `make backend-ops` compares CPU against Metal with a *tolerance*, and on a
// Metal machine these never run under `make port` at all -- measured, by
// putting an abort in a CPU kernel and watching inference finish without
// hitting it. `scripts/vecdot-prefix` and `scripts/vecdot-golden` cover only
// the quantized kernels.
//
// Bits, not floats
// ----------------
// Recorded as the raw u32 so the comparison is exact. A dot product right to
// six decimals and wrong in the last bit is a porting bug, and that is what
// this is for.
//
// Contraction is captured **on**, matching the ARM quant kernels and for the
// same reason: the fusion sites here are explicit intrinsics
// (`GGML_F32_VEC_FMA`), not expressions a compiler chose, so the port can name
// them with `@mulAdd` and match exactly. See `CLAUDE.md`, "Float contraction".
//
// Inputs are a fixed LCG so this is reproducible, captured once at the pin.
//
// Build and run: see scripts/vec-golden

#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Declared rather than included: `ggml-cpu/vec.h` pulls in the SIMD mapping
// headers, and only these three signatures are needed.
void ggml_vec_dot_f32 (int n, float * s, size_t bs, const float      * x, size_t bx, const float      * y, size_t by, int nrc);
void ggml_vec_dot_f16 (int n, float * s, size_t bs,       ggml_fp16_t * x, size_t bx,       ggml_fp16_t * y, size_t by, int nrc);
void ggml_vec_dot_bf16(int n, float * s, size_t bs,       ggml_bf16_t * x, size_t bx,       ggml_bf16_t * y, size_t by, int nrc);

// 512 elements: a multiple of the f32 step (16) and the f16 step (32), with
// several whole iterations of each so a bug in the accumulator loop shows up
// rather than being masked by a single pass.
#define NELEM 512

// A deliberately awkward length: 512 + 7 leaves a tail for both the f32 step
// (16) and the f16 step (32), so the scalar remainder loop is exercised. The
// tails use different accumulator types from the vector bodies -- f32 for the
// f32 kernel, double for the other two -- so a port that gets the body right
// and the tail wrong passes every NELEM-only check.
#define NELEM_TAIL 519

enum pattern { P_RANDOM, P_ZEROS, P_SIGNS, P_OPPOSED, P_LOPSIDED, P_TIES, P_SKEWED, NPATTERNS };

static const char * pattern_names[NPATTERNS] = {
    "random", "zeros", "signs", "opposed", "lopsided", "ties", "skewed",
};

static uint32_t lcg_state = 1;

static float lcg_next(void) {
    lcg_state = 1103515245u * lcg_state + 12345u;
    return ((float) (lcg_state >> 16) / 32768.0f) - 1.0f;
}

// The same six patterns `harness/vecdot_golden.c` uses, for the same reasons:
// only `zeros` may produce a zero result, and `ties` exists because no
// pseudorandom input lands on an exact rounding boundary.
static void fill(enum pattern p, float * x, float * y, int n) {
    lcg_state = 1;
    for (int i = 0; i < n; i++) {
        switch (p) {
            case P_RANDOM:
                x[i] = lcg_next();
                y[i] = lcg_next();
                break;
            case P_ZEROS:
                x[i] = 0.0f;
                y[i] = 0.0f;
                break;
            case P_SIGNS:
                x[i] = (i % 2) ? 1.0f : -0.5f;
                y[i] = (i % 3) ? 0.75f : -1.0f;
                break;
            case P_OPPOSED:
                // Near-total cancellation: the running sum passes through ~0
                // while the terms stay large.
                //
                // Note this does **not** discriminate summation order --
                // measured. With y == -x every term is -x*x, all the same
                // sign, and a plain left-to-right sum reproduces the C's
                // four-accumulator tree exactly. It is here for the
                // cancellation, not the ordering.
                x[i] = lcg_next();
                y[i] = -x[i];
                break;
            case P_LOPSIDED:
                // Four orders of magnitude apart. **This and `random` are the
                // two patterns that discriminate summation order** -- measured
                // against a naive left-to-right sum, which differs from the C
                // in the last bit on exactly these two and agrees on the other
                // four. Mixed signs and mixed magnitudes are what make
                // reassociation visible.
                x[i] = lcg_next() * 1e-4f;
                y[i] = lcg_next() * 4.0f;
                break;
            case P_TIES:
                if (i == 0) {
                    x[i] = 127.0f;
                    y[i] = 127.0f;
                } else {
                    const float half = (float) ((i % 9) - 4) + 0.5f;
                    x[i] = half;
                    y[i] = -half;
                }
                break;
            case P_SKEWED:
                // The only pattern that separates the **pairwise** final
                // reduce from an ordered one. Measured: without it, all six
                // others pass with `vaddvq_f32` replaced by an ordered sum.
                //
                // `vaddvq_f32` reduces as `(l0+l1) + (l2+l3)`; an ordered sum
                // is `((l0+l1)+l2) + l3`. Those agree unless the two *pairs*
                // differ enough in magnitude for the grouping to change a
                // rounding -- which is why `lopsided`, whose magnitudes vary
                // per element but not per lane, does not reach it.
                //
                // Lane k of the surviving accumulator holds every element
                // with `i % 4 == k`, so keying the scale on `i % 4` is the
                // only way to shape the lanes from the input. Three small
                // lanes against one dominant one does it. The scale is a
                // power of two so the scaling is exact and this tests the
                // reduction rather than the multiply.
                x[i] = lcg_next() * ((i % 4) == 3 ? 1.0f : 0x1p-8f);
                y[i] = lcg_next();
                break;
            default:
                x[i] = 0.0f;
                y[i] = 0.0f;
                break;
        }
    }
}

static uint32_t bits_of(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

int main(void) {
    static float xf[NELEM_TAIL];
    static float yf[NELEM_TAIL];
    static ggml_fp16_t xh[NELEM_TAIL], yh[NELEM_TAIL];
    static ggml_bf16_t xb[NELEM_TAIL], yb[NELEM_TAIL];

    // The f16 and bf16 tables have to be live before any conversion.
    ggml_cpu_init();

    printf("//! Golden values for the float dot products of"
           " `llama.cpp/ggml/src/ggml-cpu/vec.cpp`.\n");
    printf("//!\n");
    printf("//! **Generated by `scripts/vec-golden`. Do not edit.**\n");
    printf("//!\n");
    printf("//! Captured from the reference C at v0.3.0 (`c1d0e7a00`) with\n");
    printf("//! `-ffp-contract=on`, matching the ARM quant goldens: the fusion\n");
    printf("//! sites in `vec.cpp` are explicit `GGML_F32_VEC_FMA` intrinsics, so\n");
    printf("//! the port can name them with `@mulAdd` and match exactly.\n");
    printf("//!\n");
    printf("//! Compared on **bits**, never a tolerance. These kernels are\n");
    printf("//! accumulating reductions whose whole risk is the last bit, and no\n");
    printf("//! other gate in this project can see them: `backend-ops` compares\n");
    printf("//! against Metal with a tolerance, and on a Metal machine they never\n");
    printf("//! run under `make port` at all.\n");
    printf("//!\n");
    printf("//! Each kernel is captured at two lengths: `nelem`, a whole number of\n");
    printf("//! vector steps, and `nelem_tail`, which leaves a scalar remainder.\n");
    printf("//! The tails accumulate in a different type from the bodies, so a\n");
    printf("//! port that gets the body right and the tail wrong passes the first\n");
    printf("//! and fails the second.\n");
    printf("\n");
    printf("/// One kernel's result for each input pattern, as raw `f32` bits.\n");
    printf("pub const Dot = struct {\n");
    for (int p = 0; p < NPATTERNS; p++) {
        printf("    %s: u32,\n", pattern_names[p]);
    }
    printf("};\n\n");
    printf("/// Elements per dot product: a whole number of vector steps.\n");
    printf("pub const nelem = %d;\n\n", NELEM);
    printf("/// A length that leaves a scalar tail for both the f32 and f16 steps.\n");
    printf("pub const nelem_tail = %d;\n\n", NELEM_TAIL);

    const int lens[2]       = { NELEM, NELEM_TAIL };
    const char * sufs[2]    = { "", "_tail" };

    for (int li = 0; li < 2; li++) {
        const int n = lens[li];

        for (int k = 0; k < 3; k++) {
            const char * name = (k == 0) ? "f32" : (k == 1) ? "f16" : "bf16";
            // `f32` and `f16` are Zig primitive type names, so the
            // constants carry a `dot_` prefix rather than shadowing them.
            printf("pub const dot_%s%s = Dot{\n", name, sufs[li]);

            for (int p = 0; p < NPATTERNS; p++) {
                fill((enum pattern) p, xf, yf, n);

                float s = 0.0f;
                if (k == 0) {
                    ggml_vec_dot_f32(n, &s, 0, xf, 0, yf, 0, 1);
                } else if (k == 1) {
                    for (int i = 0; i < n; i++) {
                        xh[i] = ggml_fp32_to_fp16(xf[i]);
                        yh[i] = ggml_fp32_to_fp16(yf[i]);
                    }
                    ggml_vec_dot_f16(n, &s, 0, xh, 0, yh, 0, 1);
                } else {
                    for (int i = 0; i < n; i++) {
                        xb[i] = ggml_fp32_to_bf16(xf[i]);
                        yb[i] = ggml_fp32_to_bf16(yf[i]);
                    }
                    ggml_vec_dot_bf16(n, &s, 0, xb, 0, yb, 0, 1);
                }

                printf("    .%s = 0x%08X, // % .9g\n", pattern_names[p], bits_of(s), (double) s);
            }
            printf("};\n\n");
        }
    }

    return 0;
}
