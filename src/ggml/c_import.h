/*
 * The importable half of ggml's headers, translated for `src/ggml/impl.zig`
 * as the `ggml_c` module (`b.addTranslateC` in `build/llamacpp.zig`).
 *
 * **Not a port.** `ggml-impl.h` is deliberately absent -- see `impl.zig`'s
 * module doc comment.
 */

/*
 * ggml-common.h holds the block layouts and, behind the IMPL define, the
 * i-quant codebooks -- iq2xxs_grid, ksigns_iq2xs, kvalues_iq4nl and the
 * rest, about 1,900 lines of tables.
 *
 * **Unlike ggml-impl.h, this header imports cleanly**, tables included, so
 * the ported quantizers index the same arrays the C does rather than a
 * hand-transcribed copy. Transcribing them would have been the single
 * largest source of silent error in the file.
 */
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
/*
 * Brings in ggml-common.h, and with it the block layouts, the QK block
 * sizes, and the reference quantize/dequantize declarations.
 */
#include "ggml-quants.h"
#include "ggml-threading.h"
#include "ggml-cpu.h"
/*
 * The GGUF container format: `enum gguf_type`, `struct gguf_init_params`
 * and `gguf_reader_callback_t`, for `gguf.zig`. It includes only `ggml.h`,
 * so unlike `ggml-impl.h` it imports cleanly.
 */
#include "gguf.h"
/*
 * The training API: `struct ggml_opt_params`, the optimizer and loss enums
 * and the opaque handle typedefs, for `opt.zig`. It includes only `ggml.h`
 * and `ggml-backend.h`, so it imports cleanly.
 */
#include "ggml-opt.h"
