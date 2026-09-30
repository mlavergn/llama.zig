//! Internal ggml helpers that `@cImport` cannot reach.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/ggml/src/ggml-impl.h`  — the bulk of this file
//! - `llama.cpp/ggml/src/ggml.c`       — `Object`, which `ggml_tensor_overhead`
//!                                       needs but which lives in a different
//!                                       ported file
//!
//! Both at v0.3.0 (`c1d0e7a00`). Each declaration below names the source file
//! and line it came from.
//!
//! This header is not importable: it pulls in `<arm_neon.h>`, whose `__mfp8`
//! type Zig's translate-c cannot parse. Everything here is therefore a
//! hand-written equivalent of a `static inline` or macro from that header.
//! Only what ported code actually needs is reproduced, so this file grows as
//! the port does.
//!
//! Nothing here defines a C symbol. These are inline equivalents; the real
//! symbols still come from the C side.

const std = @import("std");

/// The importable half of ggml's headers.
///
/// `ggml-impl.h` is deliberately absent -- see the module doc comment. Ported
/// code reaches every C type and symbol through here, so this list is the
/// single place that changes when a new header is needed.
pub const c = @cImport({
    // ggml-common.h holds the block layouts and, behind the IMPL define, the
    // i-quant codebooks -- iq2xxs_grid, ksigns_iq2xs, kvalues_iq4nl and the
    // rest, about 1,900 lines of tables.
    //
    // **Unlike ggml-impl.h, this header imports cleanly**, tables included, so
    // the ported quantizers index the same arrays the C does rather than a
    // hand-transcribed copy. Transcribing them would have been the single
    // largest source of silent error in the file.
    @cDefine("GGML_COMMON_DECL_C", "");
    @cDefine("GGML_COMMON_IMPL_C", "");
    @cInclude("ggml.h");
    @cInclude("ggml-alloc.h");
    @cInclude("ggml-backend.h");
    @cInclude("ggml-backend-impl.h");
    // Brings in ggml-common.h, and with it the block layouts, the QK block
    // sizes, and the reference quantize/dequantize declarations.
    @cInclude("ggml-quants.h");
    @cInclude("ggml-threading.h");
    @cInclude("ggml-cpu.h");
    // The GGUF container format: `enum gguf_type`, `struct gguf_init_params`
    // and `gguf_reader_callback_t`, for `gguf.zig`. It includes only `ggml.h`,
    // so unlike `ggml-impl.h` it imports cleanly.
    @cInclude("gguf.h");
});

// -----------------------------------------------------------------------------
// Logging and assertions
//
// `ggml_log_internal` and `ggml_abort` are variadic C functions declared in
// `ggml-impl.h` and `ggml.h`. They are redeclared here rather than reached
// through the import so the format-string call sites stay explicit.

pub extern fn ggml_log_internal(level: c_uint, format: [*:0]const u8, ...) void;

/// Ports the `GGML_LOG_ERROR` macro (ggml-impl.h:121 @c1d0e7a00).
pub const logError = struct {
    pub fn f(comptime fmt: [*:0]const u8, args: anytype) void {
        @call(.auto, ggml_log_internal, .{ c.GGML_LOG_LEVEL_ERROR, fmt } ++ args);
    }
}.f;

/// Ports the `GGML_LOG_INFO` macro (ggml-impl.h:119 @c1d0e7a00).
pub const logInfo = struct {
    pub fn f(comptime fmt: [*:0]const u8, args: anytype) void {
        @call(.auto, ggml_log_internal, .{ c.GGML_LOG_LEVEL_INFO, fmt } ++ args);
    }
}.f;

/// Ports the `GGML_LOG_WARN` macro (ggml-impl.h:120 @c1d0e7a00).
pub const logWarn = struct {
    pub fn f(comptime fmt: [*:0]const u8, args: anytype) void {
        @call(.auto, ggml_log_internal, .{ c.GGML_LOG_LEVEL_WARN, fmt } ++ args);
    }
}.f;

/// Ports `GGML_DEBUG` (ggml-impl.h:125 @c1d0e7a00).
///
/// Zero upstream, and upstream never changes it: it is a compile-time knob a
/// developer edits by hand. Kept as a named constant so the gate below reads
/// as the port of a real macro rather than as dead code someone might delete.
pub const debug_level = 0;

/// Ports the `GGML_PRINT_DEBUG` macro (ggml-impl.h:128 @c1d0e7a00).
///
/// **Emits nothing**, because `GGML_DEBUG` is 0 and the macro expands to an
/// empty statement. This matters: calling `logDebug` here instead would put a
/// line on stderr for every graph build and every context init, which the C
/// does not do. The arguments are still type-checked, so the call site cannot
/// rot.
pub inline fn printDebug(comptime fmt: [*:0]const u8, args: anytype) void {
    if (debug_level >= 1) logDebug(fmt, args);
}

/// Ports the `GGML_LOG_DEBUG` macro (ggml-impl.h:122 @c1d0e7a00).
pub const logDebug = struct {
    pub fn f(comptime fmt: [*:0]const u8, args: anytype) void {
        @call(.auto, ggml_log_internal, .{ c.GGML_LOG_LEVEL_DEBUG, fmt } ++ args);
    }
}.f;

/// Ports the `GGML_ABORT` macro (ggml.h:287 @c1d0e7a00).
///
/// Aborts through ggml's own handler rather than Zig's panic, so a failure in
/// ported code surfaces the same way a failure in C code does.
///
/// Parameters:
/// - `msg`: message passed to `ggml_abort` as a literal format string.
///
/// Return: never.
pub fn abort(comptime msg: []const u8) noreturn {
    c.ggml_abort("ggml-alloc.zig", 0, msg ++ "");
    unreachable;
}

/// Ports the `GGML_ASSERT` macro (ggml.h:288 @c1d0e7a00).
///
/// Unlike `std.debug.assert`, this stays live in release builds, matching the C
/// macro, which is not gated on `NDEBUG`.
///
/// Parameters:
/// - `ok`: the condition that must hold.
/// - `msg`: text identifying the assertion when it fails.
///
/// Return: nothing when `ok` holds; aborts otherwise.
pub fn assert(ok: bool, comptime msg: []const u8) void {
    if (!ok) abort("GGML_ASSERT(" ++ msg ++ ") failed");
}

// -----------------------------------------------------------------------------
// Heap allocation
//
// Ports `ggml_malloc` (ggml.c:405 @c1d0e7a00) and `ggml_calloc` (ggml.c:419 @c1d0e7a00), plus the
// `GGML_MALLOC` / `GGML_CALLOC` macros that wrap them. They are `inline
// static` in the C, so there is no symbol to call and they have to be
// reproduced -- and they are declared here rather than in the file that ports
// them because both the graph code and the context code allocate this way.
//
// Note the zero-size case returns null rather than aborting, which callers
// depend on: a graph built without gradients allocates no gradient arrays.

/// Ports `ggml_malloc` (ggml.c:405 @c1d0e7a00). Aborts on failure, as the C does.
pub fn ggmlMalloc(size: usize) ?*anyopaque {
    if (size == 0) {
        logWarn("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n", .{});
        return null;
    }
    return std.c.malloc(size) orelse {
        logError("%s: failed to allocate %6.2f MB\n", .{ "ggml_malloc", @as(f64, @floatFromInt(size)) / (1024.0 * 1024.0) });
        abort("fatal error");
    };
}

/// Ports `ggml_calloc` (ggml.c:419 @c1d0e7a00). Aborts on failure, as the C does.
pub fn ggmlCalloc(num: usize, size: usize) ?*anyopaque {
    if (num == 0 or size == 0) {
        logWarn("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n", .{});
        return null;
    }
    return std.c.calloc(num, size) orelse {
        logError("%s: failed to allocate %6.2f MB\n", .{ "ggml_calloc", @as(f64, @floatFromInt(size)) / (1024.0 * 1024.0) });
        abort("fatal error");
    };
}

// -----------------------------------------------------------------------------
// Arithmetic macros

/// Ports the `GGML_PAD` macro (ggml.h:267 @c1d0e7a00): rounds `x` up to a multiple of `n`.
///
/// Parameters:
/// - `x`: value to round up.
/// - `n`: alignment; must be a power of two, as in the C macro.
///
/// Return: `x` rounded up to the next multiple of `n`.
/// Ports `TENSOR_ALIGNMENT` (ggml-impl.h:44 @c1d0e7a00).
///
/// The alignment every backend buffer rounds its base up to. Lives here
/// rather than in `backend.zig` because `context.zig` wants it too, and
/// `ggml-impl.h` is not importable.
pub const tensor_alignment: usize = 32;

pub inline fn pad(x: usize, n: usize) usize {
    return (x + n - 1) & ~(n - 1);
}

// -----------------------------------------------------------------------------
// C pointer narrowing
//
// `@cImport` gives array fields the `[*c]T` type, which carries "may be null"
// and "may be zero-address". Indexing one yields `*allowzero T`, which will not
// coerce to a plain `*T`. Ported code knows these pointers are valid -- the C
// asserted as much before storing them -- so these helpers assert that once,
// here, instead of at every use.

/// Narrows a C many-pointer to a Zig many-pointer.
///
/// Parameters:
/// - `T`: pointee type.
/// - `p`: the C pointer; must be non-null.
///
/// Return: the same address, typed so indexing yields a normal pointer.
pub inline fn many(comptime T: type, p: [*c]T) [*]T {
    return @ptrCast(p);
}

/// Narrows a C pointer to a Zig single-item pointer.
///
/// Parameters:
/// - `T`: pointee type.
/// - `p`: the C pointer; must be non-null.
///
/// Return: the same address as a `*T`.
pub inline fn one(comptime T: type, p: [*c]T) *T {
    return @ptrCast(p);
}

// -----------------------------------------------------------------------------
// Tensor predicates

/// Ports `ggml_impl_is_view` (ggml-impl.h:103 @c1d0e7a00).
pub inline fn isView(t: *const c.ggml_tensor) bool {
    return t.view_src != null;
}

/// Ports `ggml_are_same_layout` (ggml-impl.h:75 @c1d0e7a00).
///
/// Parameters:
/// - `a`, `b`: tensors to compare; borrowed for the call only.
/// Ports `ggml_op_is_empty` (ggml-impl.h:90 @c1d0e7a00).
///
/// True for the five ops that produce a view rather than data. The graph
/// walks past them: there is nothing to compute.
///
/// Parameters:
/// - `op`: the op to test.
///
/// Return: true when the op writes nothing.
pub fn opIsEmpty(op: c.enum_ggml_op) bool {
    return switch (op) {
        c.GGML_OP_NONE,
        c.GGML_OP_RESHAPE,
        c.GGML_OP_TRANSPOSE,
        c.GGML_OP_VIEW,
        c.GGML_OP_PERMUTE,
        => true,
        else => false,
    };
}

///
/// Return: true when type, shape, and strides all match.
pub fn areSameLayout(a: *const c.ggml_tensor, b: *const c.ggml_tensor) bool {
    if (a.type != b.type) return false;
    for (0..c.GGML_MAX_DIMS) |i| {
        if (a.ne[i] != b.ne[i]) return false;
        if (a.nb[i] != b.nb[i]) return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Op parameters
//
// Every op stores its scalar configuration in the tensor's fixed `op_params`
// byte array, reinterpreted as i32 or f32. Ported from `ggml-impl.h` lines
// 147-171, where they are `static inline` and so have no symbol to link.

/// Ports `ggml_set_op_params` (ggml-impl.h:147 @c1d0e7a00).
///
/// Parameters:
/// - `tensor`: tensor whose parameters to set.
/// - `params`: source bytes, at most `GGML_MAX_OP_PARAMS`.
pub fn setOpParams(tensor: *c.ggml_tensor, params: []const u8) void {
    std.debug.assert(params.len <= c.GGML_MAX_OP_PARAMS);
    const dst: [*]u8 = @ptrCast(&tensor.op_params);
    @memcpy(dst[0..params.len], params);
}

/// Sets `op_params` from any value, which is how most callers use it.
///
/// The C writes a local struct and passes its address and size; this takes the
/// value directly so the size cannot disagree with the type.
pub fn setOpParamsValue(tensor: *c.ggml_tensor, value: anytype) void {
    const bytes = std.mem.asBytes(&value);
    setOpParams(tensor, bytes);
}

/// Ports `ggml_get_op_params_i32` (ggml-impl.h:153 @c1d0e7a00).
pub inline fn getOpParamsI32(tensor: *const c.ggml_tensor, i: usize) i32 {
    std.debug.assert(i < c.GGML_MAX_OP_PARAMS / @sizeOf(i32));
    return tensor.op_params[i];
}

/// Ports `ggml_get_op_params_f32` (ggml-impl.h:158 @c1d0e7a00).
pub inline fn getOpParamsF32(tensor: *const c.ggml_tensor, i: usize) f32 {
    std.debug.assert(i < c.GGML_MAX_OP_PARAMS / @sizeOf(f32));
    return @bitCast(tensor.op_params[i]);
}

/// Ports `ggml_set_op_params_i32` (ggml-impl.h:163 @c1d0e7a00).
pub inline fn setOpParamsI32(tensor: *c.ggml_tensor, i: usize, value: i32) void {
    std.debug.assert(i < c.GGML_MAX_OP_PARAMS / @sizeOf(i32));
    tensor.op_params[i] = value;
}

/// Ports `ggml_set_op_params_f32` (ggml-impl.h:168 @c1d0e7a00).
pub inline fn setOpParamsF32(tensor: *c.ggml_tensor, i: usize, value: f32) void {
    std.debug.assert(i < c.GGML_MAX_OP_PARAMS / @sizeOf(f32));
    tensor.op_params[i] = @bitCast(value);
}

/// Ports `ggml_compute_softplus_f32` (ggml-impl.h:107 @c1d0e7a00).
///
/// The branch at 20 is not an approximation shortcut: above it `expf` would
/// overflow while `log(1 + e^x)` is already indistinguishable from `x`.
pub inline fn softplus(input: f32) f32 {
    return if (input > 20.0) input else @log(1 + @exp(input));
}

// -----------------------------------------------------------------------------
// Hash set
//
// The set itself is allocated and freed on the C side; only the lookup path is
// `static inline` in the header and so has to be reproduced here.

/// Ports `struct ggml_map_custom1_op_params`,
/// `struct ggml_map_custom2_op_params`, `struct ggml_map_custom3_op_params`
/// and `struct ggml_custom_op_params` (ggml-impl.h:173, 179, 185, 191 @c1d0e7a00).
///
/// One struct covers all four: the C declares them separately because each
/// names a differently-typed function pointer, but the layout is identical and
/// only the layout crosses into `op_params`. The pointer is kept opaque here
/// for the same reason -- the callee casts it back before calling.
///
/// Layout must match the C: a backend reads these back out of `op_params`.
pub const CustomOpParams = extern struct {
    fun: ?*const anyopaque,
    n_tasks: c_int,
    userdata: ?*anyopaque,
};

/// Ports `ggml_bitset_t` (ggml-impl.h:199 @c1d0e7a00).
pub const Bitset = u32;

const bitset_shr = 5; // log2(@bitSizeOf(Bitset))
const bitset_mask = @bitSizeOf(Bitset) - 1;

/// Ports `struct ggml_hash_set` (ggml-impl.h:226 @c1d0e7a00).
///
/// Layout must match the C definition exactly: instances cross the ABI
/// boundary via `ggml_hash_set_new` and `ggml_hash_set_free`.
pub const HashSet = extern struct {
    size: usize,
    used: [*c]Bitset,
    keys: [*c]?*c.ggml_tensor,
};

/// Ports `struct ggml_cgraph` (ggml-impl.h:329 @c1d0e7a00).
///
/// Declared here rather than imported: the definition is in `ggml-impl.h`, so
/// `@cImport` only yields an opaque type. Layout must match the C exactly --
/// graphs are built by C and read here.
/// Ports `struct ggml_object` (ggml.c:957 @c1d0e7a00).
///
/// The header ggml puts in front of every allocation inside a context, forming
/// a linked list so the context can be walked and reset. Declared here rather
/// than in the file that ports it because `ggml_tensor_overhead` needs its
/// size and lives elsewhere.
///
/// Layout must match the C: contexts built by C code are read here. The
/// trailing padding is explicit in the C and kept explicit here, since the
/// struct size is asserted to be a multiple of `GGML_MEM_ALIGN`.
pub const Object = extern struct {
    offs: usize,
    size: usize,
    next: ?*Object,
    type: c.enum_ggml_object_type,
    padding: [4]u8,
};

/// Ports `enum ggml_cgraph_eval_order` (ggml-impl.h:323 @c1d0e7a00).
///
/// The values are spelled out rather than imported: the enum is declared in
/// `ggml-impl.h`, which `@cImport` cannot read.
pub const EvalOrder = c_uint;
pub const eval_order_left_to_right: EvalOrder = 0;
pub const eval_order_right_to_left: EvalOrder = 1;
pub const eval_order_count: EvalOrder = 2;

pub const CGraph = extern struct {
    size: c_int,
    n_nodes: c_int,
    n_leafs: c_int,

    nodes: [*c]?*c.ggml_tensor,
    grads: [*c]?*c.ggml_tensor,
    grad_accs: [*c]?*c.ggml_tensor,
    leafs: [*c]?*c.ggml_tensor,
    use_counts: [*c]i32,

    visited_hash_set: HashSet,

    order: EvalOrder,

    uid: u64,
};

/// Ports `GGML_HASHSET_FULL` (ggml-impl.h:223 @c1d0e7a00): the table was walked
/// end to end without finding the key or a free slot.
pub const hashset_full = std.math.maxInt(usize);

/// Ports `GGML_HASHSET_ALREADY_EXISTS` (ggml-impl.h:224 @c1d0e7a00).
pub const hashset_already_exists = std.math.maxInt(usize) - 1;

/// Ports `ggml_bitset_size` (ggml-impl.h:205 @c1d0e7a00): words needed for `n` bits.
pub inline fn bitsetSize(n: usize) usize {
    return (n + bitset_mask) >> bitset_shr;
}

/// Ports `ggml_bitset_get` (ggml-impl.h:209 @c1d0e7a00).
pub inline fn bitsetGet(bitset: [*c]const Bitset, i: usize) bool {
    return (bitset[i >> bitset_shr] & (@as(Bitset, 1) << @intCast(i & bitset_mask))) != 0;
}

/// Ports `ggml_bitset_set` (ggml-impl.h:213 @c1d0e7a00).
pub inline fn bitsetSet(bitset: [*c]Bitset, i: usize) void {
    bitset[i >> bitset_shr] |= @as(Bitset, 1) << @intCast(i & bitset_mask);
}

/// Ports `ggml_hash` (ggml-impl.h:254 @c1d0e7a00).
///
/// The low four bits are always zero because tensors are aligned, so the C
/// version shifts them out before taking the modulus.
pub inline fn hash(p: *const c.ggml_tensor) usize {
    return @intFromPtr(p) >> 4;
}

/// Ports `ggml_hash_find_or_insert` (ggml-impl.h:300 @c1d0e7a00).
///
/// Linear probing from the hash position. Aborts when the table is full, which
/// callers rely on: the returned index is always valid.
///
/// Parameters:
/// - `set`: the hash set to search and possibly insert into.
/// - `key`: tensor to look up.
///
/// Return: the index of `key`, inserting it if it was absent.
pub fn hashFindOrInsert(set: *HashSet, key: *c.ggml_tensor) usize {
    const h = hash(key) % set.size;
    var i = h;
    while (true) {
        if (!bitsetGet(set.used, i)) {
            bitsetSet(set.used, i);
            set.keys[i] = key;
            return i;
        }
        if (set.keys[i] == key) return i;
        i = (i + 1) % set.size;
        if (i == h) break;
    }
    abort("fatal error");
}

/// Ports `ggml_hash_find` (ggml-impl.h:259 @c1d0e7a00).
///
/// Return: the index where `key` is or would go, or `hashset_full` if the
/// table holds neither it nor a free slot. A returned index is only a *hit*
/// if the used bit is also set -- callers must check both, which is the
/// asymmetry that makes this easy to get wrong.
pub fn hashFind(set: *const HashSet, key: *const c.ggml_tensor) usize {
    const h = hash(key) % set.size;
    var i = h;
    while (bitsetGet(set.used, i) and set.keys[i] != key) {
        i = (i + 1) % set.size;
        if (i == h) return hashset_full; // visited every entry
    }
    return i;
}

/// Ports `ggml_hash_insert` (ggml-impl.h:279 @c1d0e7a00).
///
/// Return: the index it was inserted at, or `hashset_already_exists` when the
/// key was already present. Aborts when the table is full.
pub fn hashInsert(set: *HashSet, key: *c.ggml_tensor) usize {
    const h = hash(key) % set.size;
    var i = h;
    while (true) {
        if (!bitsetGet(set.used, i)) {
            bitsetSet(set.used, i);
            set.keys[i] = key;
            return i;
        }
        if (set.keys[i] == key) return hashset_already_exists;
        i = (i + 1) % set.size;
        if (i == h) break;
    }
    abort("fatal error");
}

/// Ports `ggml_hash_contains` (ggml-impl.h:274 @c1d0e7a00).
pub fn hashContains(set: *const HashSet, key: *const c.ggml_tensor) bool {
    const i = hashFind(set, key);
    return i != hashset_full and bitsetGet(set.used, i);
}

// -----------------------------------------------------------------------------
// Float conversions
//
// Ported from `ggml-impl.h`. These are the macros `ggml.c` reaches for on
// nearly every line that touches a quantized tensor, and they are the reason
// that file cannot simply call into C: the macros expand to `static inline`
// bodies, so there is no symbol to link against.
//
// ggml uses the generic bit-manipulation algorithms here even on ARM -- the
// NEON include at the top of `ggml-impl.h` does not shortcut these macros -- so
// what follows reproduces the arithmetic exactly rather than deferring to
// hardware conversion. The tests at the bottom of this file check every one of
// the 65,536 half-precision inputs against the C implementation.

/// Ports `ggml_e8m0_to_fp32_half` (ggml-impl.h:477 @c1d0e7a00), the expansion of
/// `GGML_E8M0_TO_FP32_HALF`.
///
/// MXFP4's shared exponent, halved: `kvalues_mxfp4` holds twice the E2M1
/// values, so the scale is halved to compensate. The C notes that NaNs are not
/// handled, and they are not handled here either.
pub fn e8m0ToFp32Half(x: u8) f32 {
    const bits: u32 = if (x < 2)
        // 0x00200000 = 2^-128, 0x00400000 = 2^-127: subnormal patterns.
        @as(u32, 0x00200000) << @intCast(x)
    else
        // 0.5 * 2^(x-127) = 2^(x-128), i.e. a normal with exponent x-1.
        @as(u32, x - 1) << 23;
    return @bitCast(bits);
}

/// Ports `ggml_ue4m3_to_fp32` (ggml-impl.h:502 @c1d0e7a00).
///
/// NVFP4's per-sub-block scale: unsigned, 4 exponent bits biased by 7, 3
/// mantissa bits. Halved for the same reason as `e8m0ToFp32Half`.
pub fn ue4m3ToFp32(x: u8) f32 {
    if (x == 0 or x == 0x7F) return 0.0;
    const exp: i32 = @intCast((x >> 3) & 0xF);
    const man: i32 = @intCast(x & 0x7);
    const raw = if (exp == 0)
        std.math.ldexp(@as(f32, @floatFromInt(man)), -9)
    else
        std.math.ldexp(1.0 + @as(f32, @floatFromInt(man)) / 8.0, exp - 7);
    return raw * 0.5;
}

/// Ports `ggml_fp32_to_ue4m3` (ggml-impl.h:517 @c1d0e7a00).
///
/// Round-to-nearest on the mantissa, with the carry out of the mantissa
/// handled by bumping the exponent -- and a second overflow check after that
/// bump, which is easy to drop and would silently produce a wrong scale for
/// values near the top of the range.
pub fn fp32ToUe4m3(x_in: f32) u8 {
    // Written as `!(x > 0)` in the C, which also rejects NaN.
    if (!(x_in > 0.0)) return 0;
    const x = if (x_in > 448.0) 448.0 else x_in;

    const bits: u32 = @bitCast(x);
    const fp32_exp: i32 = @as(i32, @intCast((bits >> 23) & 0xFF)) - 127;
    const fp32_man: i32 = @intCast((bits >> 20) & 0x7);
    var ue4m3_exp: i32 = fp32_exp + 7;

    if (ue4m3_exp <= 0) {
        // Subnormal: value = man * 2^-9.
        var man: i32 = @intFromFloat(x * 512.0 + 0.5);
        if (man > 7) man = 7;
        if (man < 1) return 0;
        return @intCast(man);
    }
    if (ue4m3_exp >= 15) return 0x7E;

    const round_bit: i32 = @intCast((bits >> 19) & 1);
    var ue4m3_man: i32 = fp32_man + round_bit;
    if (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp += 1;
        if (ue4m3_exp >= 15) return 0x7E;
    }
    return @intCast((ue4m3_exp << 3) | ue4m3_man);
}

/// A C float-to-integer cast: truncate toward zero.
///
/// Zig's `@intFromFloat` is checked and would trap where the C silently
/// produces an unspecified value. The quantizers rely on the cast for values
/// that are always in range by construction, so the clamp below never fires
/// for real input -- it exists so a NaN or an absurd weight cannot turn into a
/// crash. Out of range, the C is undefined, so clamping is as good an answer
/// as any and is the only one that is also safe.
pub inline fn truncTo(comptime T: type, f: anytype) T {
    const t = @trunc(f);
    if (std.math.isNan(t)) return 0;
    const lo: @TypeOf(t) = @floatFromInt(std.math.minInt(T));
    const hi: @TypeOf(t) = @floatFromInt(std.math.maxInt(T));
    if (t <= lo) return std.math.minInt(T);
    if (t >= hi) return std.math.maxInt(T);
    return @intFromFloat(t);
}

/// Ports `fp32_from_bits` (ggml-impl.h:366 @c1d0e7a00).
inline fn fp32FromBits(w: u32) f32 {
    return @bitCast(w);
}

/// Ports `fp32_to_bits` (ggml-impl.h:375 @c1d0e7a00).
inline fn fp32ToBits(f: f32) u32 {
    return @bitCast(f);
}

/// Ports `ggml_compute_fp16_to_fp32` (ggml-impl.h:384 @c1d0e7a00), the expansion of
/// `GGML_FP16_TO_FP32`.
///
/// Widens a half to a float by rebuilding the exponent rather than by a
/// hardware convert, which is what the C does and therefore what the ported
/// code must do.
///
/// Parameters:
/// - `h`: the half-precision bit pattern.
///
/// Return: the widened value. Exact for every input, including subnormals,
/// infinities, and NaNs.
pub fn fp16ToFp32(h: u16) f32 {
    const w = @as(u32, h) << 16;
    const sign = w & 0x80000000;
    const two_w = w +% w;

    const exp_offset: u32 = 0xE0 << 23;
    const exp_scale: f32 = @bitCast(@as(u32, 0x7800000)); // 0x1.0p-112
    const normalized_value = fp32FromBits((two_w >> 4) +% exp_offset) * exp_scale;

    const magic_mask: u32 = 126 << 23;
    const magic_bias: f32 = 0.5;
    const denormalized_value = fp32FromBits((two_w >> 17) | magic_mask) - magic_bias;

    const denormalized_cutoff: u32 = 1 << 27;
    const result = sign |
        (if (two_w < denormalized_cutoff)
            fp32ToBits(denormalized_value)
        else
            fp32ToBits(normalized_value));
    return fp32FromBits(result);
}

/// Ports `ggml_compute_fp32_to_fp16` (ggml-impl.h:407 @c1d0e7a00), the expansion of
/// `GGML_FP32_TO_FP16`.
///
/// Narrows a float to a half by scaling through the exponent range so that
/// rounding falls out of the hardware's own round-to-nearest-even, which is
/// how the C avoids implementing rounding by hand.
///
/// Parameters:
/// - `f`: the value to narrow.
///
/// Return: the half-precision bit pattern. NaNs collapse to 0x7E00, matching
/// the C exactly.
pub fn fp32ToFp16(f: f32) u16 {
    const scale_to_inf: f32 = @bitCast(@as(u32, 0x77800000)); // 0x1.0p+112
    const scale_to_zero: f32 = @bitCast(@as(u32, 0x08800000)); // 0x1.0p-110

    var base = (@abs(f) * scale_to_inf) * scale_to_zero;

    const w = fp32ToBits(f);
    const shl1_w = w +% w;
    const sign = w & 0x80000000;
    var bias = shl1_w & 0xFF000000;
    if (bias < 0x71000000) bias = 0x71000000;

    base = fp32FromBits((bias >> 1) +% 0x07800000) + base;
    const bits = fp32ToBits(base);
    const exp_bits = (bits >> 13) & 0x00007C00;
    const mantissa_bits = bits & 0x00000FFF;
    const nonsign = exp_bits +% mantissa_bits;
    return @intCast((sign >> 16) | (if (shl1_w > 0xFF000000)
        @as(u32, 0x7E00)
    else
        nonsign));
}

/// Ports `ggml_compute_bf16_to_fp32` (ggml-impl.h:594 @c1d0e7a00), the expansion of
/// `GGML_BF16_TO_FP32`.
///
/// A brain float is the top half of a float, so widening is a shift.
pub inline fn bf16ToFp32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// Ports `ggml_compute_fp32_to_bf16` (ggml-impl.h:611 @c1d0e7a00), the expansion of
/// `GGML_FP32_TO_BF16`.
///
/// Round-to-nearest-even on the discarded low half. Subnormals are preserved
/// rather than flushed, and NaNs are forced quiet, both matching Google Brain's
/// reference conversion as the C comment notes.
///
/// Parameters:
/// - `s`: the value to narrow.
///
/// Return: the brain-float bit pattern.
pub fn fp32ToBf16(s: f32) u16 {
    const u = fp32ToBits(s);
    if ((u & 0x7fffffff) > 0x7f800000) {
        // NaN: force quiet by setting the top mantissa bit.
        return @intCast((u >> 16) | 64);
    }
    return @intCast((u +% (0x7fff + ((u >> 16) & 1))) >> 16);
}

// -----------------------------------------------------------------------------
// Unit Tests

test "pad rounds up to the next multiple" {
    try std.testing.expectEqual(@as(usize, 0), pad(0, 32));
    try std.testing.expectEqual(@as(usize, 32), pad(1, 32));
    try std.testing.expectEqual(@as(usize, 32), pad(32, 32));
    try std.testing.expectEqual(@as(usize, 64), pad(33, 32));
}

test "bitset round-trips a single bit" {
    var words = [_]Bitset{0} ** 4;
    try std.testing.expect(!bitsetGet(&words, 37));
    bitsetSet(&words, 37);
    try std.testing.expect(bitsetGet(&words, 37));
    // Bit 37 lives in word 1, bit 5. Nothing else should have moved.
    try std.testing.expectEqual(@as(Bitset, 0), words[0]);
    try std.testing.expectEqual(@as(Bitset, 1) << 5, words[1]);
}

// Golden values, captured from the C implementations in `ggml.c` before any
// of `ggml.c` was ported. They are checksums rather than direct comparisons on
// purpose: once `runtime.zig` exports `ggml_fp16_to_fp32` and friends, a test
// that called those symbols would be comparing this code against itself and
// would pass no matter how wrong it became.
//
// Regenerate only from a C build, never from this code. The procedure is in
// the commit that introduced them.
const golden_fp16_to_fp32: u64 = 0x5c17cf6b1ad614ad;
const golden_bf16_to_fp32: u64 = 0x4b65f6efc80cd5ba;
const golden_fp32_to_fp16: u64 = 0x3081ce06bf36ae60;
const golden_fp32_to_bf16: u64 = 0xd60a7155d0aaade5;

/// The mixing step the golden checksums were built with.
///
/// Order-dependent and avalanching, so a single wrong conversion anywhere in
/// the swept domain changes the result.
fn mix(h: u64, v: u64) u64 {
    return h ^ (v +% 0x9e3779b97f4a7c15 +% (h << 6) +% (h >> 2));
}

/// The fp32 sweep used by the checksums: a prime stride, so every exponent and
/// both signs are visited while the low mantissa bits keep changing.
const sweep_stride: u32 = 9973;

test "fp16 to fp32 matches the C checksum over all 65536 inputs" {
    var h: u64 = 0;
    var bits: u32 = 0;
    while (bits <= std.math.maxInt(u16)) : (bits += 1) {
        const f = fp16ToFp32(@intCast(bits));
        h = mix(h, @as(u32, @bitCast(f)));
    }
    try std.testing.expectEqual(golden_fp16_to_fp32, h);
}

test "bf16 to fp32 matches the C checksum over all 65536 inputs" {
    var h: u64 = 0;
    var bits: u32 = 0;
    while (bits <= std.math.maxInt(u16)) : (bits += 1) {
        const f = bf16ToFp32(@intCast(bits));
        h = mix(h, @as(u32, @bitCast(f)));
    }
    try std.testing.expectEqual(golden_bf16_to_fp32, h);
}

test "fp32 narrowing matches the C checksums across the exponent range" {
    var h16: u64 = 0;
    var hbf: u64 = 0;
    var bits: u32 = 0;
    while (true) {
        const f: f32 = @bitCast(bits);
        h16 = mix(h16, fp32ToFp16(f));
        hbf = mix(hbf, fp32ToBf16(f));
        if (bits > std.math.maxInt(u32) - sweep_stride) break;
        bits += sweep_stride;
    }
    try std.testing.expectEqual(golden_fp32_to_fp16, h16);
    try std.testing.expectEqual(golden_fp32_to_bf16, hbf);
}

test "fp32 narrowing matches C on the awkward values" {
    // Read out of the C implementation directly. The boundaries matter: 65504
    // is the largest finite half, 65520 is the first value that rounds to
    // infinity, and a NaN must collapse to 0x7e00 rather than keep its payload.
    const cases = [_]struct { v: f32, fp16: u16, bf16: u16 }{
        .{ .v = 0.0, .fp16 = 0x0000, .bf16 = 0x0000 },
        .{ .v = -0.0, .fp16 = 0x8000, .bf16 = 0x8000 },
        .{ .v = 1.0, .fp16 = 0x3c00, .bf16 = 0x3f80 },
        .{ .v = -1.0, .fp16 = 0xbc00, .bf16 = 0xbf80 },
        .{ .v = 65504.0, .fp16 = 0x7bff, .bf16 = 0x4780 },
        .{ .v = 65520.0, .fp16 = 0x7c00, .bf16 = 0x4780 },
        .{ .v = 6.103515625e-5, .fp16 = 0x0400, .bf16 = 0x3880 },
        .{ .v = 5.960464477539063e-8, .fp16 = 0x0001, .bf16 = 0x3380 },
        .{ .v = std.math.floatMax(f32), .fp16 = 0x7c00, .bf16 = 0x7f80 },
        .{ .v = std.math.inf(f32), .fp16 = 0x7c00, .bf16 = 0x7f80 },
        .{ .v = -std.math.inf(f32), .fp16 = 0xfc00, .bf16 = 0xff80 },
        .{ .v = std.math.nan(f32), .fp16 = 0x7e00, .bf16 = 0x7fc0 },
        .{ .v = 1e-45, .fp16 = 0x0000, .bf16 = 0x0000 },
    };
    for (cases) |case| {
        try std.testing.expectEqual(case.fp16, fp32ToFp16(case.v));
        try std.testing.expectEqual(case.bf16, fp32ToBf16(case.v));
    }
}

test "fp16 widening and narrowing round-trip" {
    var bits: u32 = 0;
    while (bits <= std.math.maxInt(u16)) : (bits += 1) {
        const h: u16 = @intCast(bits);
        const widened = fp16ToFp32(h);
        if (std.math.isNan(widened)) continue; // NaN payloads are not preserved
        try std.testing.expectEqual(h, fp32ToFp16(widened));
    }
}
