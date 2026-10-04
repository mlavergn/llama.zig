/* Call the port's small-struct and scalar entry points as a C caller does,
 * and check the values arrive.
 *
 * **Not a port.** This is a gate, driven by `scripts/abi-check`.
 *
 * Why it exists: Zig 0.16 on aarch64-macos miscompiles an `extern struct`
 * received by value across the C ABI unless its size is a multiple of 4
 * with no padding. `ggml_bf16_to_fp32` takes `ggml_bf16_t`, which is
 * `struct { uint16_t bits; }` -- two bytes -- and it returned 0.0 for
 * every input from the day `ggml.c` was ported. Nine correctness gates
 * missed it, because nothing on the inference path calls it: the row
 * variant takes a pointer, and Zig-internal callers never cross the C
 * ABI.
 *
 * The measured rule, over ten struct shapes:
 *
 *   1 byte                      zeros
 *   2 bytes                     zeros
 *   4 bytes, u16 + u8 + pad     zeros
 *   4 bytes, one u32 or f32     correct
 *   6 bytes, 3 x u16            zeros
 *   8 bytes, 8 x u8             correct
 *   8 bytes, 2 x f32            correct
 *   12 bytes, 3 x u32           correct
 *   16 bytes, bool + ptr        correct
 *   2 bytes declared align(4)   zeros
 *
 * One-directional: Zig as the *caller* passes these correctly; only Zig
 * as the *callee* is affected.
 */
#include "ggml.h"
#include "gguf.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;
static int checks = 0;

static void check(const char *what, double got, double want) {
    checks++;
    if (got == want) {
        printf("  %-40s %g\n", what, got);
    } else {
        printf("  %-40s got %g, want %g   WRONG\n", what, got, want);
        fails++;
    }
}

int main(void) {
    printf("=== small-struct and scalar entry points, as C calls them ===\n");

    /* ggml_bf16_t: struct { uint16_t bits; }, 2 bytes. The one that was
     * broken. Values are exact in bf16, so == is the right comparison.
     *
     * **The non-zero inputs are what make this a gate.** With the broken
     * signature the callee sees zeros, so 0x0000 -> 0.0 and 0x8000 ->
     * -0.0 both still compare equal and pass. Only the other four fail.
     * A gate whose inputs are all zero cannot see a bug that reads
     * zero. */
    {
        const uint16_t bits[] = { 0x3f80, 0x4000, 0xbfc0, 0x4120, 0x0000, 0x8000 };
        const float    want[] = {   1.0f,   2.0f,  -1.5f,  10.0f,   0.0f,  -0.0f };
        for (size_t i = 0; i < sizeof(want) / sizeof(*want); i++) {
            ggml_bf16_t b;
            b.bits = bits[i];
            char label[64];
            snprintf(label, sizeof label, "ggml_bf16_to_fp32(0x%04x)", bits[i]);
            check(label, ggml_bf16_to_fp32(b), want[i]);
        }
        /* and the round trip, which exercises the return direction */
        checks++;
        if (ggml_fp32_to_bf16(1.0f).bits == 0x3f80) {
            printf("  %-40s 0x3f80\n", "ggml_fp32_to_bf16(1.0)");
        } else {
            printf("  %-40s got 0x%04x, want 0x3f80   WRONG\n",
                   "ggml_fp32_to_bf16(1.0)", ggml_fp32_to_bf16(1.0f).bits);
            fails++;
        }
    }

    /* ggml_fp16_t is a typedef for uint16_t, not a struct -- included so a
     * future upstream change to a struct is caught here. */
    check("ggml_fp16_to_fp32(0x3c00)", ggml_fp16_to_fp32(0x3c00), 1.0);
    check("ggml_fp16_to_fp32(0xc000)", ggml_fp16_to_fp32(0xc000), -2.0);

    /* gguf_init_params: 16 bytes, bool + pointer with padding between.
     * Passed in two registers and measured correct, but it is the other
     * by-value struct in the exported surface, so it is pinned here. */
    {
        struct ggml_context *ctx = NULL;
        struct gguf_init_params p = { /* .no_alloc = */ true, /* .ctx = */ &ctx };
        /* A path that cannot exist, so this fails for file reasons and
         * never touches the parse -- what is under test is only that the
         * struct arrived intact enough to get there. */
        struct gguf_context *g = gguf_init_from_file("/nonexistent/abi-check.gguf", p);
        checks++;
        if (g == NULL) {
            printf("  %-40s returned NULL as expected\n", "gguf_init_from_file(missing)");
        } else {
            printf("  %-40s unexpectedly succeeded   WRONG\n", "gguf_init_from_file(missing)");
            fails++;
            gguf_free(g);
        }
    }

    /* ggml_init takes a 24-byte struct, passed in memory. */
    {
        struct ggml_init_params ip = { 1024 * 1024, NULL, true };
        struct ggml_context *c = ggml_init(ip);
        checks++;
        if (c != NULL) {
            printf("  %-40s non-NULL\n", "ggml_init(mem_size=1MiB, no_alloc)");
            ggml_free(c);
        } else {
            printf("  %-40s returned NULL   WRONG\n", "ggml_init");
            fails++;
        }
    }

    printf("\n");
    if (fails == 0) {
        printf("PASS: %d C-ABI entry points carry their arguments\n", checks);
        return 0;
    }
    printf("FAIL: %d of %d C-ABI entry points lose their arguments\n", fails, checks);
    return 1;
}
