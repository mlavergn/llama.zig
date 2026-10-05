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

#include <cstddef>
#include <cstdio>
#include <cstdint>

extern "C" {
    size_t zz_props_sizeof(void);
    size_t zz_props_alignof(void);
    size_t zz_props_nfields(void);
    size_t zz_props_offset(size_t);
    size_t zz_props_field_size(size_t);
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

    printf("\n");
    if (fails == 0) {
        printf("PASS: %d layout facts agree, hand-declared vs header\n", checks);
        return 0;
    }
    printf("FAIL: %d of %d layout facts differ\n", fails, checks);
    return 1;
}
