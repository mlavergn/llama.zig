/* Compare the hand-declared Metal structs against the real headers.
 *
 * **Not a port.** This is a gate, driven by `scripts/struct-layout`.
 *
 * Why it exists: `src/ggml/metal/device_c.zig` hand-declares
 * `ggml_metal_device_props` rather than importing `ggml-metal-device.h`,
 * because that header would have to go into `impl.zig`'s single `cImport`
 * and widen the `c` namespace every ported file sees. The struct is 19
 * fields of which the port reads seven; the other twelve exist only to
 * place those seven at the right offsets.
 *
 * `CLAUDE.md` calls hand-transcription "the single largest source of
 * silent error", and this is the shape of error it means: a field typed
 * `int` where the C has `size_t` leaves every later field at the wrong
 * offset, and the port then reads `max_working_set_size` as
 * `max_buffer_size` and reports nothing.
 *
 * So the C compiler is asked. This includes the real header and compares
 * its own `sizeof`, `alignof`, `offsetof` and per-field width against the
 * values the Zig exports.
 */
#include "ggml-metal-device.h"
#include "ggml-metal-impl.h"

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>

extern "C" {
    size_t zz_props_sizeof(void);
    size_t zz_props_alignof(void);
    size_t zz_props_nfields(void);
    size_t zz_props_offset(size_t);
    size_t zz_props_field_size(size_t);

    size_t zz_pwp_sizeof(void);
    size_t zz_pwp_nfields(void);
    size_t zz_pwp_offset(size_t);
    size_t zz_pwp_field_size(size_t);
    struct ggml_metal_pipeline_with_params zz_pwp_roundtrip(void);

    size_t zz_bid_sizeof(void);
    size_t zz_bid_alignof(void);
    size_t zz_bid_nfields(void);
    size_t zz_bid_offset(size_t);
    size_t zz_bid_field_size(size_t);
    void   zz_bid_roundtrip(struct ggml_metal_buffer_id, void **, size_t *);

    long        zz_fc_value(size_t);
    const char *zz_fc_name(size_t);
    size_t      zz_fc_count(void);
}

namespace {

int fails = 0;
int checks = 0;

void cmp(const char *what, size_t got, size_t want) {
    checks++;
    if (got != want) {
        printf("  %-46s zig %-6zu  C %-6zu  WRONG\n", what, got, want);
        fails++;
    }
}

struct FieldSpec {
    const char *name;
    size_t      offset;
    size_t      size;
};

}  // namespace

#define F(member) { #member, offsetof(ggml_metal_device_props, member), sizeof(((ggml_metal_device_props *) nullptr)->member) }
#define F2(T, member) { #member, offsetof(T, member), sizeof(((T *) nullptr)->member) }

int main() {
    printf("=== ggml_metal_device_props: hand-declared vs the header ===\n");

    /* Declaration order, which is what zz_props_offset indexes. */
    const FieldSpec fields[] = {
        F(device),
        F(device_phys),
        F(device_virt),
        F(name),
        F(desc),
        F(max_buffer_size),
        F(max_working_set_size),
        F(max_theadgroup_memory_size),
        F(has_simdgroup_reduction),
        F(has_simdgroup_mm),
        F(has_unified_memory),
        F(has_bfloat),
        F(has_tensor),
        F(use_residency_sets),
        F(use_shared_buffers),
        F(supports_gpu_family_apple7),
        F(device_id),
        F(gpu_family),
        F(op_offload_min_batch_size),
    };
    const size_t n = sizeof(fields) / sizeof(*fields);

    cmp("sizeof", zz_props_sizeof(), sizeof(ggml_metal_device_props));
    cmp("alignof", zz_props_alignof(), alignof(ggml_metal_device_props));

    /* The field count is checked so that a field added upstream, or one
     * dropped on our side, cannot slip past as "every field I listed is
     * fine". The list above is this file's own and has to be extended
     * with the header; the count is what forces that. */
    cmp("field count", zz_props_nfields(), n);

    char lbl[128];
    for (size_t i = 0; i < n; i++) {
        snprintf(lbl, sizeof lbl, "offsetof(%s)", fields[i].name);
        cmp(lbl, zz_props_offset(i), fields[i].offset);
        snprintf(lbl, sizeof lbl, "sizeof(%s)", fields[i].name);
        cmp(lbl, zz_props_field_size(i), fields[i].size);
    }

    /* ggml_metal_pipeline_with_params: 40 bytes with interior padding,
     * and returned by value by all 68 get_pipeline_* functions. Both its
     * layout and the return path are checked -- a mismatch would corrupt
     * every pipeline lookup at once, and today's bf16 bug showed Zig can
     * get a by-value struct ABI wrong without saying so. */
    printf("--- ggml_metal_pipeline_with_params ---\n");
    {
        typedef ggml_metal_pipeline_with_params PWP;
        const FieldSpec pf[] = {
            F2(PWP, pipeline), F2(PWP, nsg), F2(PWP, nr0), F2(PWP, nr1),
            F2(PWP, smem),     F2(PWP, c4),  F2(PWP, cnt),
        };
        const size_t pn = sizeof(pf) / sizeof(*pf);

        cmp("pwp sizeof", zz_pwp_sizeof(), sizeof(PWP));
        cmp("pwp field count", zz_pwp_nfields(), pn);
        for (size_t i = 0; i < pn; i++) {
            snprintf(lbl, sizeof lbl, "pwp offsetof(%s)", pf[i].name);
            cmp(lbl, zz_pwp_offset(i), pf[i].offset);
            snprintf(lbl, sizeof lbl, "pwp sizeof(%s)", pf[i].name);
            cmp(lbl, zz_pwp_field_size(i), pf[i].size);
        }

        /* the return path, field by field */
        PWP r = zz_pwp_roundtrip();
        cmp("pwp return .pipeline", (size_t) (uintptr_t) r.pipeline, (size_t) 0xdead0000u);
        cmp("pwp return .nsg",  (size_t) r.nsg,  (size_t) 11);
        cmp("pwp return .nr0",  (size_t) r.nr0,  (size_t) 22);
        cmp("pwp return .nr1",  (size_t) r.nr1,  (size_t) 33);
        cmp("pwp return .smem", r.smem,          (size_t) 44444);
        cmp("pwp return .c4",   (size_t) r.c4,   (size_t) 1);
        cmp("pwp return .cnt",  (size_t) r.cnt,  (size_t) 0);
    }

    /* ggml_metal_buffer_id: 16 bytes, both returned by value from the
     * .m and passed by value back into it. The round-trip is the half
     * Zig 0.16 gets wrong for some shapes, so it is exercised rather
     * than assumed from the size. */
    printf("--- ggml_metal_buffer_id ---\n");
    cmp("buffer_id: sizeof",  zz_bid_sizeof(),  sizeof(struct ggml_metal_buffer_id));
    cmp("buffer_id: alignof", zz_bid_alignof(), alignof(struct ggml_metal_buffer_id));
    cmp("buffer_id: field count", zz_bid_nfields(), 2);
    cmp("buffer_id.metal offset", zz_bid_offset(0), offsetof(struct ggml_metal_buffer_id, metal));
    cmp("buffer_id.offs offset",  zz_bid_offset(1), offsetof(struct ggml_metal_buffer_id, offs));
    cmp("buffer_id.metal width",  zz_bid_field_size(0), sizeof(((struct ggml_metal_buffer_id *) 0)->metal));
    cmp("buffer_id.offs width",   zz_bid_field_size(1), sizeof(((struct ggml_metal_buffer_id *) 0)->offs));
    {
        /* Non-zero and distinct in both fields: a receive path that
         * zeroes the struct, or one that swaps the fields, both fail. */
        struct ggml_metal_buffer_id in;
        in.metal = (void *) 0xfeedfacecafe0001ull;
        in.offs  = 0x2233445566778899ull;

        void  *got_metal = NULL;
        size_t got_offs  = 0;
        zz_bid_roundtrip(in, &got_metal, &got_offs);

        cmp("buffer_id round-trip: metal", (size_t) got_metal, (size_t) in.metal);
        cmp("buffer_id round-trip: offs",  got_offs, in.offs);
    }

    /* The FC_* and OP_* #defines the kernel names carry. A wrong one is
     * the quietest fault in the Metal port: `FC_UNARY + 1` names the slot
     * a Metal function constant is written to, so an off-by-one sets a
     * different constant and the kernel does something else. Nothing in
     * the ported Zig could notice, so the header is asked. */
    printf("--- ggml-metal-impl.h FC_* and OP_* ---\n");
    {
        struct NV { const char *name; long value; };
        const NV want[] = {
            { "SZ_SIMDGROUP", (long) SZ_SIMDGROUP },
            { "N_MM_NK", (long) N_MM_NK },
            { "N_MM_BLOCK_X", (long) N_MM_BLOCK_X },
            { "N_MM_BLOCK_Y", (long) N_MM_BLOCK_Y },
            { "N_MM_SIMD_GROUP_X", (long) N_MM_SIMD_GROUP_X },
            { "N_MM_SIMD_GROUP_Y", (long) N_MM_SIMD_GROUP_Y },
            { "N_R0_Q1_0", (long) N_R0_Q1_0 },
            { "N_SG_Q1_0", (long) N_SG_Q1_0 },
            { "N_R0_Q2_0", (long) N_R0_Q2_0 },
            { "N_SG_Q2_0", (long) N_SG_Q2_0 },
            { "N_R0_Q4_0", (long) N_R0_Q4_0 },
            { "N_SG_Q4_0", (long) N_SG_Q4_0 },
            { "N_R0_Q4_1", (long) N_R0_Q4_1 },
            { "N_SG_Q4_1", (long) N_SG_Q4_1 },
            { "N_R0_Q5_0", (long) N_R0_Q5_0 },
            { "N_SG_Q5_0", (long) N_SG_Q5_0 },
            { "N_R0_Q5_1", (long) N_R0_Q5_1 },
            { "N_SG_Q5_1", (long) N_SG_Q5_1 },
            { "N_R0_Q8_0", (long) N_R0_Q8_0 },
            { "N_SG_Q8_0", (long) N_SG_Q8_0 },
            { "N_R0_MXFP4", (long) N_R0_MXFP4 },
            { "N_SG_MXFP4", (long) N_SG_MXFP4 },
            { "N_R0_Q2_K", (long) N_R0_Q2_K },
            { "N_SG_Q2_K", (long) N_SG_Q2_K },
            { "N_R0_Q3_K", (long) N_R0_Q3_K },
            { "N_SG_Q3_K", (long) N_SG_Q3_K },
            { "N_R0_Q4_K", (long) N_R0_Q4_K },
            { "N_SG_Q4_K", (long) N_SG_Q4_K },
            { "N_R0_Q5_K", (long) N_R0_Q5_K },
            { "N_SG_Q5_K", (long) N_SG_Q5_K },
            { "N_R0_Q6_K", (long) N_R0_Q6_K },
            { "N_SG_Q6_K", (long) N_SG_Q6_K },
            { "N_R0_IQ1_S", (long) N_R0_IQ1_S },
            { "N_SG_IQ1_S", (long) N_SG_IQ1_S },
            { "N_R0_IQ1_M", (long) N_R0_IQ1_M },
            { "N_SG_IQ1_M", (long) N_SG_IQ1_M },
            { "N_R0_IQ2_XXS", (long) N_R0_IQ2_XXS },
            { "N_SG_IQ2_XXS", (long) N_SG_IQ2_XXS },
            { "N_R0_IQ2_XS", (long) N_R0_IQ2_XS },
            { "N_SG_IQ2_XS", (long) N_SG_IQ2_XS },
            { "N_R0_IQ2_S", (long) N_R0_IQ2_S },
            { "N_SG_IQ2_S", (long) N_SG_IQ2_S },
            { "N_R0_IQ3_XXS", (long) N_R0_IQ3_XXS },
            { "N_SG_IQ3_XXS", (long) N_SG_IQ3_XXS },
            { "N_R0_IQ3_S", (long) N_R0_IQ3_S },
            { "N_SG_IQ3_S", (long) N_SG_IQ3_S },
            { "N_R0_IQ4_NL", (long) N_R0_IQ4_NL },
            { "N_SG_IQ4_NL", (long) N_SG_IQ4_NL },
            { "N_R0_IQ4_XS", (long) N_R0_IQ4_XS },
            { "N_SG_IQ4_XS", (long) N_SG_IQ4_XS },
            { "N_R0_TQ2_0", (long) N_R0_TQ2_0 },
            { "N_SG_TQ2_0", (long) N_SG_TQ2_0 },
            { "FC_FLASH_ATTN_EXT_PAD", (long) FC_FLASH_ATTN_EXT_PAD },
            { "FC_FLASH_ATTN_EXT_BLK", (long) FC_FLASH_ATTN_EXT_BLK },
            { "FC_FLASH_ATTN_EXT", (long) FC_FLASH_ATTN_EXT },
            { "FC_FLASH_ATTN_EXT_VEC", (long) FC_FLASH_ATTN_EXT_VEC },
            { "FC_FLASH_ATTN_EXT_VEC_REDUCE", (long) FC_FLASH_ATTN_EXT_VEC_REDUCE },
            { "FC_MUL_MV", (long) FC_MUL_MV },
            { "FC_MUL_MM", (long) FC_MUL_MM },
            { "FC_ROPE", (long) FC_ROPE },
            { "FC_SSM_CONV", (long) FC_SSM_CONV },
            { "FC_SOLVE_TRI", (long) FC_SOLVE_TRI },
            { "FC_COUNT_EQUAL", (long) FC_COUNT_EQUAL },
            { "FC_UNARY", (long) FC_UNARY },
            { "FC_BIN", (long) FC_BIN },
            { "FC_SUM_ROWS", (long) FC_SUM_ROWS },
            { "FC_UPSCALE", (long) FC_UPSCALE },
            { "FC_GATED_DELTA_NET", (long) FC_GATED_DELTA_NET },
            { "OP_FLASH_ATTN_EXT_NQPSG", (long) OP_FLASH_ATTN_EXT_NQPSG },
            { "OP_FLASH_ATTN_EXT_NCPSG", (long) OP_FLASH_ATTN_EXT_NCPSG },
            { "OP_FLASH_ATTN_EXT_VEC_NQPSG", (long) OP_FLASH_ATTN_EXT_VEC_NQPSG },
            { "OP_FLASH_ATTN_EXT_VEC_NCPSG", (long) OP_FLASH_ATTN_EXT_VEC_NCPSG },
            { "OP_LIGHTNING_INDEXER_DK", (long) OP_LIGHTNING_INDEXER_DK },
            { "OP_LIGHTNING_INDEXER_NH", (long) OP_LIGHTNING_INDEXER_NH },
            { "OP_LIGHTNING_INDEXER_NHPTG", (long) OP_LIGHTNING_INDEXER_NHPTG },
            { "OP_LIGHTNING_INDEXER_NKPSG", (long) OP_LIGHTNING_INDEXER_NKPSG },
            { "OP_LIGHTNING_INDEXER_NSG", (long) OP_LIGHTNING_INDEXER_NSG },
            { "OP_LIGHTNING_INDEXER_NBPTG", (long) OP_LIGHTNING_INDEXER_NBPTG },
            { "OP_UNARY_NUM_SCALE", (long) OP_UNARY_NUM_SCALE },
            { "OP_UNARY_NUM_FILL", (long) OP_UNARY_NUM_FILL },
            { "OP_UNARY_NUM_CLAMP", (long) OP_UNARY_NUM_CLAMP },
            { "OP_UNARY_NUM_SQR", (long) OP_UNARY_NUM_SQR },
            { "OP_UNARY_NUM_SQRT", (long) OP_UNARY_NUM_SQRT },
            { "OP_UNARY_NUM_SIN", (long) OP_UNARY_NUM_SIN },
            { "OP_UNARY_NUM_COS", (long) OP_UNARY_NUM_COS },
            { "OP_UNARY_NUM_LOG", (long) OP_UNARY_NUM_LOG },
            { "OP_UNARY_NUM_LEAKY_RELU", (long) OP_UNARY_NUM_LEAKY_RELU },
            { "OP_UNARY_NUM_TANH", (long) OP_UNARY_NUM_TANH },
            { "OP_UNARY_NUM_RELU", (long) OP_UNARY_NUM_RELU },
            { "OP_UNARY_NUM_SIGMOID", (long) OP_UNARY_NUM_SIGMOID },
            { "OP_UNARY_NUM_GELU", (long) OP_UNARY_NUM_GELU },
            { "OP_UNARY_NUM_GELU_ERF", (long) OP_UNARY_NUM_GELU_ERF },
            { "OP_UNARY_NUM_GELU_QUICK", (long) OP_UNARY_NUM_GELU_QUICK },
            { "OP_UNARY_NUM_SILU", (long) OP_UNARY_NUM_SILU },
            { "OP_UNARY_NUM_ELU", (long) OP_UNARY_NUM_ELU },
            { "OP_UNARY_NUM_NEG", (long) OP_UNARY_NUM_NEG },
            { "OP_UNARY_NUM_ABS", (long) OP_UNARY_NUM_ABS },
            { "OP_UNARY_NUM_SGN", (long) OP_UNARY_NUM_SGN },
            { "OP_UNARY_NUM_STEP", (long) OP_UNARY_NUM_STEP },
            { "OP_UNARY_NUM_HARDSWISH", (long) OP_UNARY_NUM_HARDSWISH },
            { "OP_UNARY_NUM_HARDSIGMOID", (long) OP_UNARY_NUM_HARDSIGMOID },
            { "OP_UNARY_NUM_EXP", (long) OP_UNARY_NUM_EXP },
            { "OP_UNARY_NUM_SOFTPLUS", (long) OP_UNARY_NUM_SOFTPLUS },
            { "OP_UNARY_NUM_EXPM1", (long) OP_UNARY_NUM_EXPM1 },
            { "OP_UNARY_NUM_FLOOR", (long) OP_UNARY_NUM_FLOOR },
            { "OP_UNARY_NUM_CEIL", (long) OP_UNARY_NUM_CEIL },
            { "OP_UNARY_NUM_ROUND", (long) OP_UNARY_NUM_ROUND },
            { "OP_UNARY_NUM_TRUNC", (long) OP_UNARY_NUM_TRUNC },
            { "OP_UNARY_NUM_XIELU", (long) OP_UNARY_NUM_XIELU },
            { "OP_SUM_ROWS_NUM_SUM_ROWS", (long) OP_SUM_ROWS_NUM_SUM_ROWS },
            { "OP_SUM_ROWS_NUM_MEAN", (long) OP_SUM_ROWS_NUM_MEAN },
            { "N_MM_NK_TOTAL", (long) N_MM_NK_TOTAL }
        };
        const size_t wn = sizeof(want) / sizeof(*want);

        cmp("FC/OP constant count", zz_fc_count(), wn);

        for (size_t i = 0; i < wn; i++) {
            const char *got_name = zz_fc_name(i);
            checks++;
            if (!got_name || strcmp(got_name, want[i].name) != 0) {
                printf("  %-46s zig %-20s C %-20s WRONG\n", "constant order",
                       got_name ? got_name : "(null)", want[i].name);
                fails++;
                continue;
            }
            snprintf(lbl, sizeof lbl, "%s", want[i].name);
            cmp(lbl, (size_t) zz_fc_value(i), (size_t) want[i].value);
        }
    }

    printf("\n");
    if (fails == 0) {
        printf("PASS: %d layout facts agree, hand-declared vs header\n", checks);
        return 0;
    }
    printf("FAIL: %d of %d layout facts differ\n", fails, checks);
    return 1;
}
