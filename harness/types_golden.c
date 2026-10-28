// The checksums `src/ggml/types.zig` holds its name tables to, computed from
// the C. **Not a port**: our own harness, linked against the stock ggml that
// `llama.cpp.zmake` builds (scripts/types-golden), so the numbers never come
// from the code they check. Same mix as `types.zig`'s `mix`/`mixStr`.
#include <stdint.h>
#include <stdio.h>
#include "ggml.h"

static uint64_t mix(uint64_t h, uint64_t v) {
    return h ^ (v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
}

static uint64_t mix_str(uint64_t h, const char * s) {
    if (!s) return mix(h, 0xdeadbeef);
    for (; *s; s++) h = mix(h, (unsigned char) *s);
    return mix(h, 0);
}

int main(void) {
    uint64_t hn = 0, hs = 0, hu = 0, hg = 0;
    for (int i = 0; i < GGML_OP_COUNT; i++) {
        hn = mix_str(hn, ggml_op_name((enum ggml_op) i));
        hs = mix_str(hs, ggml_op_symbol((enum ggml_op) i));
    }
    for (int i = 0; i < GGML_UNARY_OP_COUNT; i++) hu = mix_str(hu, ggml_unary_op_name((enum ggml_unary_op) i));
    for (int i = 0; i < GGML_GLU_OP_COUNT; i++) hg = mix_str(hg, ggml_glu_op_name((enum ggml_glu_op) i));
    printf("const golden_op_name: u64 = 0x%016llx;\n", (unsigned long long) hn);
    printf("const golden_op_symbol: u64 = 0x%016llx;\n", (unsigned long long) hs);
    printf("const golden_unary_name: u64 = 0x%016llx;\n", (unsigned long long) hu);
    printf("const golden_glu_name: u64 = 0x%016llx;\n", (unsigned long long) hg);
    return 0;
}
