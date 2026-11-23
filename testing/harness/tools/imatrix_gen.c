// Writes small GGUF imatrix files whose statistics reach the arms a
// collection over the fixtures cannot: NextN (MTP) layers with and without
// `imatrix.n_layer_nextn`, a layer gap, non-finite sums, zero counts, an
// all-zero and a constant tensor, a block index past `int`, a name of more
// than four parts, one wider than the statistics table's column, tensors
// with and without activation sums side by side,
// an `in_sum` whose size disagrees with `in_sum2`, and a second chunk size.
// The values come from a fixed LCG, so both sides read the same bytes.
//
// Usage: imatrix_gen <variant> <out.gguf>
//   variant: nextn nextn-nokey gap special mixed badact chunk64
#include "ggml.h"
#include "gguf.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng = 2463534242u;
static float frand(void) { // [0, 1)
    rng = rng * 1664525u + 1013904223u;
    return (rng >> 8) / 16777216.0f;
}

enum { F_ACT = 1, F_NAN = 2, F_ZERO = 4, F_CONST = 8, F_ZEROCOUNT = 16, F_BADACT = 32, F_NEG = 64 };

static struct ggml_context * ctx;
static struct gguf_context * g;

// one entry: nmat matrices of nrow sums each
static void entry(const char * name, int nrow, int nmat, int flags) {
    char n[256];
    struct ggml_tensor * s2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrow, nmat);
    struct ggml_tensor * cn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, nmat);
    snprintf(n, sizeof n, "%s.in_sum2", name); ggml_set_name(s2, n);
    snprintf(n, sizeof n, "%s.counts", name);  ggml_set_name(cn, n);
    float * d2 = (float *) s2->data;
    float * dc = (float *) cn->data;
    for (int m = 0; m < nmat; m++) {
        dc[m] = (flags & F_ZEROCOUNT) && m % 2 == 1 ? 0.0f : (float) (16 + (int) (frand() * 48));
    }
    for (int i = 0; i < nrow * nmat; i++) {
        const float cnt = dc[i / nrow];
        float v = cnt * (0.01f + frand() * frand() * 4.0f);
        if (flags & F_ZERO)  { v = 0.0f; }
        if (flags & F_CONST) { v = cnt * 0.5f; }
        if ((flags & F_NAN) && i % 7 == 3) { v = i % 2 ? INFINITY : NAN; }
        d2[i] = v;
    }
    gguf_add_tensor(g, s2);
    gguf_add_tensor(g, cn);
    if (flags & F_ACT) {
        const int nact = (flags & F_BADACT) ? nrow * nmat - 1 : nrow * nmat;
        struct ggml_tensor * s1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nact);
        snprintf(n, sizeof n, "%s.in_sum", name); ggml_set_name(s1, n);
        float * d1 = (float *) s1->data;
        for (int i = 0; i < nact; i++) {
            const float cnt = dc[(i % (nrow * nmat)) / nrow];
            float v = cnt * (frand() - ((flags & F_NEG) ? 0.8f : 0.3f));
            if (flags & F_ZERO)  { v = 0.0f; }
            if (flags & F_CONST) { v = cnt * 0.25f; }
            if ((flags & F_NAN) && i % 5 == 1) { v = NAN; }
            d1[i] = v;
        }
        gguf_add_tensor(g, s1);
    }
}

static void meta(int chunk_size, int nextn) {
    const char * ds[] = { "gen-a.txt", "gen-b.txt" };
    gguf_set_val_str(g, "general.type", "imatrix");
    gguf_set_arr_str(g, "imatrix.datasets", ds, 2);
    gguf_set_val_u32(g, "imatrix.chunk_count", 4);
    gguf_set_val_u32(g, "imatrix.chunk_size", (uint32_t) chunk_size);
    if (nextn > 0) { gguf_set_val_u32(g, "imatrix.n_layer_nextn", (uint32_t) nextn); }
}

static void model_layers(int n_trunk, int skip, int act_flags, int with_mtp) {
    char name[128];
    for (int il = 0; il < n_trunk; il++) {
        if (il == skip) { continue; }
        snprintf(name, sizeof name, "blk.%d.attn_q.weight", il);      entry(name, 24, 1, act_flags);
        snprintf(name, sizeof name, "blk.%d.ffn_up_exps.weight", il); entry(name, 16, 4, act_flags);
        snprintf(name, sizeof name, "blk.%d.ffn_down.weight", il);    entry(name, 40, 1, act_flags);
    }
    if (with_mtp) {
        snprintf(name, sizeof name, "blk.%d.attn_q.weight", n_trunk);        entry(name, 24, 1, act_flags);
        snprintf(name, sizeof name, "blk.%d.nextn.eh_proj.weight", n_trunk); entry(name, 48, 1, act_flags);
        snprintf(name, sizeof name, "blk.%d.ffn_down.weight", n_trunk);      entry(name, 40, 1, act_flags);
    }
    entry("output.weight", 24, 1, act_flags);
    entry("token_embd.weight", 24, 1, act_flags);
}

int main(int argc, char ** argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s variant out.gguf\n", argv[0]); return 2; }
    const char * v = argv[1];
    struct ggml_init_params gp = { /*.mem_size =*/ 16*1024*1024, /*.mem_buffer =*/ NULL, /*.no_alloc =*/ false };
    ctx = ggml_init(gp);
    g = gguf_init_empty();

    if (strcmp(v, "nextn") == 0) {
        meta(32, 1);
        model_layers(4, -1, F_ACT, 1);
    } else if (strcmp(v, "nextn-nokey") == 0) {
        meta(32, 0);
        model_layers(4, -1, F_ACT, 1);
    } else if (strcmp(v, "chunk64") == 0) {
        rng = 7;
        meta(64, 2);
        model_layers(4, -1, F_ACT, 1);
    } else if (strcmp(v, "gap") == 0) {
        meta(32, 0);
        model_layers(5, 2, F_ACT, 0);
    } else if (strcmp(v, "special") == 0) {
        meta(32, 0);
        entry("blk.0.attn_q.weight", 24, 1, F_ACT);
        entry("blk.1.attn_q.weight", 24, 1, F_ACT | F_NAN);
        entry("blk.2.attn_q.weight", 24, 1, F_ACT | F_ZERO);
        entry("blk.3.attn_q.weight", 24, 1, F_ACT | F_CONST);
        entry("blk.0.ffn_up_exps.weight", 16, 4, F_ACT | F_ZEROCOUNT);
        entry("blk.1.ffn_up_exps.weight", 16, 4, F_ACT | F_NEG);
        entry("blk.2.ffn_up_exps.weight", 16, 4, F_ACT | F_ZEROCOUNT | F_NAN);
        entry("blk.3.ffn_up_exps.weight", 16, 4, F_ACT);
        entry("blk.1.ffn.up.proj.weight", 8, 1, F_ACT);
        entry("blk.2.ffn.up.proj.weight", 8, 1, F_ACT | F_NEG);
        entry("blk.99999999999.attn_q.weight", 24, 1, F_ACT);
        entry("blk.x.attn_q.weight", 24, 1, F_ACT);
        entry("blk.3.attn_k", 8, 1, F_ACT);
        entry("blk.2.a_tensor_name_longer_than_the_table_column.weight", 8, 1, F_ACT);
        // a four-part and a five-part name in one block: process_tensor_name
        // keeps the whole name for the first and "cc.dd" for the second, so
        // the sort puts "blk.0.zz.weight" first only if the skip is right
        entry("blk.0.zz.weight", 8, 1, F_ACT);
        entry("blk.0.cc.dd.weight", 8, 1, F_ACT);
        entry("output.weight", 24, 1, F_ACT | F_NEG);
    } else if (strcmp(v, "mixed") == 0) {
        meta(32, 0);
        char name[128];
        for (int il = 0; il < 4; il++) {
            snprintf(name, sizeof name, "blk.%d.attn_q.weight", il);   entry(name, 24, 1, il % 2 ? 0 : F_ACT);
            snprintf(name, sizeof name, "blk.%d.ffn_down.weight", il); entry(name, 40, 1, F_ACT);
        }
        entry("output.weight", 24, 1, 0);
    } else if (strcmp(v, "badact") == 0) {
        meta(32, 0);
        entry("blk.0.attn_q.weight", 24, 1, F_ACT);
        entry("blk.1.attn_q.weight", 24, 1, F_ACT | F_BADACT);
    } else {
        fprintf(stderr, "unknown variant %s\n", v);
        return 2;
    }

    if (!gguf_write_to_file(g, argv[2], false)) { fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    gguf_free(g);
    ggml_free(ctx);
    return 0;
}
