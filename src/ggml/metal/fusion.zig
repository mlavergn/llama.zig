//! The Metal backend's fusion table: the one place a fusable subgraph is
//! declared.
//!
//! # Provenance
//!
//! Ports `llama.cpp/ggml/src/ggml-metal/ggml-metal-fusion.cpp`, with the
//! types and constants of `llama.cpp/ggml/src/ggml-metal/ggml-metal-fusion.h`,
//! at v0.4.1 (`b29c606e2`). Each declaration below names the C++ it
//! replaces and the line it began at.
//!
//! # Why one table
//!
//! The graph optimizer (`common.zig`'s `ggml_graph_optimize`, through
//! `ggml_metal_fusion_max`) packs fusable runs before it reorders, and the
//! op encoders (`ops.zig`, through `ggml_metal_fusion_next`) fuse them when
//! they encode. Before v0.4.1 each phase had its own idea of what fused;
//! now both consult `fusions`, so they cannot disagree.
//!
//! # Its contract is C, and two callers stay Objective-C
//!
//! All 17 of the C++'s exports are `extern "C"`. `ggml-metal-device.m`
//! owns a `ggml_metal_fusion_info` per device and `ggml-metal-context.m`
//! reads its flags, labels and counts; both stay Objective-C (Decision
//! 13) and call the `ggml_metal_fusion_info_*` functions below by name.
//! The struct itself is opaque to them, so `Info` is free to be a Zig
//! struct. `Fusion` is not: its address is what `count_fusion` matches
//! against the table, and it is declared in the header, so it keeps the
//! C's layout.

const std = @import("std");

const impl = @import("../impl.zig");
const c = impl.c;

const mc = @import("device_c.zig");
const types = @import("../types.zig");
const graph = @import("../graph.zig");

const Tensor = c.ggml_tensor;
const CGraph = impl.CGraph;

/// The allocator ported ggml uses — libc's, as in the rest of `metal/`.
const allocator = std.heap.c_allocator;

/// Ports `GGML_METAL_FUSION_MAX` (ggml-metal-fusion.h:20 @b29c606e2): the
/// most nodes one kernel fuses, and the longest run the optimizer packs.
pub const max = 16;

/// Ports `ggml_metal_fusion_mode` (ggml-metal-fusion.h:22 @b29c606e2).
pub const Mode = enum(c_uint) {
    /// structural checks only; used by the graph optimizer, at which point
    /// the graph tensors are not allocated yet, so buffer placement cannot
    /// be verified
    structural = 0,
    /// full checks, including buffer placement; used by the op encoders
    full = 1,
};

/// Ports `ggml_metal_fusion_id` (ggml-metal-fusion.h:31 @b29c606e2): which
/// kernel an encoder uses for a matched pattern.
pub const Id = enum(c_uint) {
    none = 0,
    /// NORM/RMS_NORM + MUL
    norm_mul = 1,
    /// NORM/RMS_NORM + MUL + ADD
    norm_mul_add = 2,
    /// ADD x N (N in [2, 7])
    add_chain = 3,
    /// MUL + SIN + SQR + MUL + ADD
    snake = 4,
    /// GATED_DELTA_NET + CPY (write snapshots into the recurrent cache)
    gdn_cache = 5,
};

/// The `check` callback's type: `nodes[j]` is the pattern's `j`-th node.
pub const CheckFn = *const fn (fusion: *const Fusion, nodes: [*]const *const Tensor, mode: Mode) callconv(.c) bool;

/// Ports `struct ggml_metal_fusion` (ggml-metal-fusion.h:40 @b29c606e2).
///
/// `extern` because the header declares it and `count_fusion` identifies
/// an entry by its address in `fusions`.
pub const Fusion = extern struct {
    id: Id,

    /// op sequence (fixed length)
    ops: [*]const c.enum_ggml_op,
    /// number of ops
    n_ops: c_int,

    /// if unsafe: the generic chain/shape + `ggml_can_fuse_subgraph`
    /// checks are skipped and `check` is the sole validator (used for
    /// patterns that are not elision chains, e.g. the gdn + cache-cpy
    /// write-through fusion)
    unsafe: bool,

    /// extra backend constraints on top of `ggml_can_fuse_subgraph`
    check: ?CheckFn,
};

// -----------------------------------------------------------------------------
// Helpers

/// `src[i]`, narrowed: a `[*c]` must go through `impl.one` before an array
/// field is indexed through it (`CLAUDE.md`).
inline fn src(t: *const Tensor, i: usize) *const Tensor {
    return impl.one(Tensor, t.src[i]);
}

/// Ports `ggml_metal_fusion_same_buffer` (ggml-metal-fusion.cpp:13
/// @b29c606e2): true if two tensors live in the same Metal buffer.
fn sameBuffer(a_opt: ?*const Tensor, b_opt: ?*const Tensor) bool {
    const a = a_opt orelse return false;
    const b = b_opt orelse return false;

    const ba = impl.one(c.ggml_backend_buffer, if (a.view_src != null) impl.one(Tensor, a.view_src).buffer else a.buffer);
    const bb = impl.one(c.ggml_backend_buffer, if (b.view_src != null) impl.one(Tensor, b.view_src).buffer else b.buffer);

    const ca: *mc.Buffer = @ptrCast(@alignCast(ba.context.?));
    const cb: *mc.Buffer = @ptrCast(@alignCast(bb.context.?));

    return mc.ggml_metal_buffer_get_id(ca, a).metal == mc.ggml_metal_buffer_get_id(cb, b).metal;
}

// -----------------------------------------------------------------------------
// Pattern checks

/// Ports `ggml_metal_fusion_check_norm` (ggml-metal-fusion.cpp:31
/// @b29c606e2).
///
/// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match
/// the norm input width, be contiguous rows, and the fused outputs must
/// stay F32.
fn checkNorm(fusion: *const Fusion, nodes: [*]const *const Tensor, mode: Mode) callconv(.c) bool {
    _ = mode;

    impl.assert(fusion.n_ops >= 2, "fusion->n_ops >= 2");

    var j: usize = 1;
    while (j < @as(usize, @intCast(fusion.n_ops))) : (j += 1) {
        // the fused MUL/ADD must read the previous node as src0
        if (nodes[j].src[0] != nodes[j - 1]) {
            return false;
        }

        // the weight/bias must have the same row width as the norm input
        if (src(nodes[j], 1).ne[0] != nodes[0].ne[0]) {
            return false;
        }

        if (!types.ggml_is_contiguous_rows(src(nodes[j], 1))) {
            return false;
        }

        if (nodes[j].type != c.GGML_TYPE_F32) {
            return false;
        }
    }

    return true;
}

/// Ports `ggml_metal_fusion_check_add_chain` (ggml-metal-fusion.cpp:64
/// @b29c606e2).
///
/// ADD x N: each ADD reads the previous ADD as src0, and all addends must
/// share layout (and, in FULL mode, live in the same Metal buffer).
fn checkAddChain(fusion: *const Fusion, nodes: [*]const *const Tensor, mode: Mode) callconv(.c) bool {
    impl.assert(fusion.n_ops >= 2, "fusion->n_ops >= 2");

    var j: usize = 1;
    while (j < @as(usize, @intCast(fusion.n_ops))) : (j += 1) {
        if (nodes[j].src[0] != nodes[j - 1]) {
            return false;
        }

        if (!impl.areSameLayout(src(nodes[j], 1), src(nodes[j - 1], 1))) {
            return false;
        }

        if (mode == .full) {
            if (!sameBuffer(src(nodes[j], 1), src(nodes[0], 1))) {
                return false;
            }
        }
    }

    return true;
}

/// Ports `ggml_metal_fusion_check_gdn_cache` (ggml-metal-fusion.cpp:94
/// @b29c606e2).
///
/// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state
/// snapshots into the recurrent cache, so the gdn kernel writes them
/// straight to the cache and the cpy is elided. The gdn output has other
/// consumers (the attn scores view), so unlike the other patterns this is
/// not an elision chain: the structural checks live entirely here
/// (`unsafe = true`).
fn checkGdnCache(fusion: *const Fusion, nodes: [*]const *const Tensor, mode: Mode) callconv(.c) bool {
    _ = fusion;

    const gdn = nodes[0];
    const cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn.type != c.GGML_TYPE_F32 or (gdn.flags & c.GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }

    if (cpy.op != c.GGML_OP_CPY or (cpy.flags & c.GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }

    const g2 = src(gdn, 2);
    const S_v: i64 = g2.ne[0];
    const H: i64 = g2.ne[1];
    const n_tokens: i64 = g2.ne[2];
    const n_seqs: i64 = g2.ne[3];
    const K: i64 = impl.getOpParamsI32(gdn, 0);
    const tail_off: usize = types.ggml_row_size(c.GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    const D: i64 = S_v * S_v * H;
    const n_written: i64 = @min(n_tokens, K);

    const s = src(cpy, 0); // gdn snapshot tail view
    const dst = src(cpy, 1); // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (s.op != c.GGML_OP_VIEW or s.view_src != gdn or
        s.view_offs != tail_off or !c.ggml_is_contiguous(s))
    {
        return false;
    }

    const expected_ne = [c.GGML_MAX_DIMS]i64{ D, n_seqs, n_written, 1 };
    if (dst.type != c.GGML_TYPE_F32 or
        !std.mem.eql(i64, &expected_ne, &dst.ne) or
        dst.nb[0] != types.ggml_type_size(c.GGML_TYPE_F32) or
        dst.nb[1] != types.ggml_row_size(c.GGML_TYPE_F32, D))
    {
        return false;
    }

    if (mode == .full) {
        // the cache must be allocated so the kernel can write straight to its buffer
        if (dst.data == null) {
            return false;
        }
    }

    return true;
}

/// Ports `ggml_metal_fusion_check_snake` (ggml-metal-fusion.cpp:150
/// @b29c606e2): MUL + SIN + SQR + MUL + ADD (snake activation).
fn checkSnake(fusion: *const Fusion, nodes: [*]const *const Tensor, mode: Mode) callconv(.c) bool {
    _ = fusion;
    _ = mode;

    const mul0 = nodes[0];
    const sin_node = nodes[1];
    const sqr = nodes[2];
    const mul1 = nodes[3];
    const add = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    const x = if (c.ggml_are_same_shape(mul0, src(mul0, 0))) src(mul0, 0) else src(mul0, 1);
    const a = if (x == src(mul0, 0)) src(mul0, 1) else src(mul0, 0);

    // mul1 reads sqr and inv_b in either operand order
    const inv_b = if (mul1.src[0] == sqr) src(mul1, 1) else src(mul1, 0);

    // closure check: the trailing add reads the same x as the leading mul
    const x_in_add = if (add.src[0] == mul1) src(add, 1) else src(add, 0);

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    const types_ok =
        (x.type == c.GGML_TYPE_F32 or x.type == c.GGML_TYPE_F16 or x.type == c.GGML_TYPE_BF16) and
        (a.type == c.GGML_TYPE_F32) and (inv_b.type == c.GGML_TYPE_F32) and
        (mul0.type == x.type) and (sin_node.type == x.type) and
        (sqr.type == x.type) and (mul1.type == x.type) and
        (add.type == x.type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    const shape_ok = c.ggml_are_same_shape(a, inv_b) and a.ne[0] == 1 and a.ne[1] == x.ne[1];
    const dim_ok =
        (x.ne[2] == 1) and (x.ne[3] == 1) and
        (add.ne[2] == 1) and (add.ne[3] == 1) and
        (a.ne[2] == 1) and (a.ne[3] == 1) and
        (inv_b.ne[2] == 1) and (inv_b.ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    const contig_ok =
        c.ggml_is_contiguous(x) and c.ggml_is_contiguous(add) and
        c.ggml_is_contiguous(a) and c.ggml_is_contiguous(inv_b);

    return types_ok and shape_ok and dim_ok and contig_ok and x_in_add == x;
}

// -----------------------------------------------------------------------------
// Patterns

// Ports `ops_norm_mul`, `ops_norm_mul_add`, `ops_rms_norm_mul`,
// `ops_rms_norm_mul_add` (ggml-metal-fusion.cpp:200, 201, 202, 203
// @b29c606e2), `ops_add_2`, `ops_add_3`, `ops_add_4`, `ops_add_5`,
// `ops_add_6`, `ops_add_7` (ggml-metal-fusion.cpp:205, 206, 207, 208, 209,
// 210 @b29c606e2), `ops_snake` (ggml-metal-fusion.cpp:211 @b29c606e2) and
// `ops_gdn_cache` (ggml-metal-fusion.cpp:213 @b29c606e2).
const ops_norm_mul = [_]c.enum_ggml_op{ c.GGML_OP_NORM, c.GGML_OP_MUL };
const ops_norm_mul_add = [_]c.enum_ggml_op{ c.GGML_OP_NORM, c.GGML_OP_MUL, c.GGML_OP_ADD };
const ops_rms_norm_mul = [_]c.enum_ggml_op{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL };
const ops_rms_norm_mul_add = [_]c.enum_ggml_op{ c.GGML_OP_RMS_NORM, c.GGML_OP_MUL, c.GGML_OP_ADD };

const ops_add_2: [2]c.enum_ggml_op = @splat(c.GGML_OP_ADD);
const ops_add_3: [3]c.enum_ggml_op = @splat(c.GGML_OP_ADD);
const ops_add_4: [4]c.enum_ggml_op = @splat(c.GGML_OP_ADD);
const ops_add_5: [5]c.enum_ggml_op = @splat(c.GGML_OP_ADD);
const ops_add_6: [6]c.enum_ggml_op = @splat(c.GGML_OP_ADD);
const ops_add_7: [7]c.enum_ggml_op = @splat(c.GGML_OP_ADD);
const ops_snake = [_]c.enum_ggml_op{ c.GGML_OP_MUL, c.GGML_OP_SIN, c.GGML_OP_SQR, c.GGML_OP_MUL, c.GGML_OP_ADD };

const ops_gdn_cache = [_]c.enum_ggml_op{ c.GGML_OP_GATED_DELTA_NET, c.GGML_OP_CPY };

fn entry(id: Id, comptime ops: []const c.enum_ggml_op, unsafe: bool, check: CheckFn) Fusion {
    return .{ .id = id, .ops = ops.ptr, .n_ops = @intCast(ops.len), .unsafe = unsafe, .check = check };
}

/// Ports `ggml_metal_fusions` (ggml-metal-fusion.cpp:215 @b29c606e2): the
/// single table of all fusions the Metal backend supports, in the C's
/// order — `ggml_metal_fusion_next` keeps the *first* longest match, and
/// the stats labels are indexed by position.
const fusions = [_]Fusion{
    entry(.norm_mul, &ops_norm_mul, false, &checkNorm),
    entry(.norm_mul_add, &ops_norm_mul_add, false, &checkNorm),
    entry(.norm_mul, &ops_rms_norm_mul, false, &checkNorm),
    entry(.norm_mul_add, &ops_rms_norm_mul_add, false, &checkNorm),
    entry(.add_chain, &ops_add_2, false, &checkAddChain),
    entry(.add_chain, &ops_add_3, false, &checkAddChain),
    entry(.add_chain, &ops_add_4, false, &checkAddChain),
    entry(.add_chain, &ops_add_5, false, &checkAddChain),
    entry(.add_chain, &ops_add_6, false, &checkAddChain),
    entry(.add_chain, &ops_add_7, false, &checkAddChain),
    entry(.snake, &ops_snake, false, &checkSnake),
    entry(.gdn_cache, &ops_gdn_cache, true, &checkGdnCache),
};

comptime {
    for (fusions) |f| std.debug.assert(f.n_ops <= max);
}

/// Ports `ggml_metal_fusion_all` (ggml-metal-fusion.cpp:230 @b29c606e2).
///
/// Parameters:
/// - `n`: receives the table's length.
///
/// Return: the table, static for the life of the process.
pub export fn ggml_metal_fusion_all(n: *c_int) callconv(.c) [*]const Fusion {
    n.* = fusions.len;

    return &fusions;
}

// -----------------------------------------------------------------------------
// Shared fusion info

/// Ports `struct ggml_metal_fusion_info` (ggml-metal-fusion.cpp:251
/// @b29c606e2).
///
/// The device owns one and every backend context on that device shares it,
/// so the counters accumulate across contexts. Opaque to the C callers.
/// The C's two `std::vector`s are slices allocated once by `labelsInit`,
/// which is also the only place the C fills them.
pub const Info = struct {
    /// NUL-terminated, one per table entry once `labels_set`
    labels: [][:0]u8 = &.{},
    counts: []u64 = &.{},
    enabled: bool,
    stats: bool,
    labels_set: bool,
    debug: c_int,
};

/// Ports `ggml_metal_fusion_label` (ggml-metal-fusion.cpp:238 @b29c606e2):
/// the pattern's op names joined with `+`.
fn label(fusion: *const Fusion) [:0]u8 {
    var buf: std.ArrayList(u8) = .empty;
    var j: usize = 0;
    while (j < @as(usize, @intCast(fusion.n_ops))) : (j += 1) {
        if (j > 0) {
            buf.append(allocator, '+') catch impl.abort("metal: out of memory building a fusion label");
        }
        buf.appendSlice(allocator, std.mem.span(types.ggml_op_name(fusion.ops[j]))) catch
            impl.abort("metal: out of memory building a fusion label");
    }
    return buf.toOwnedSliceSentinel(allocator, 0) catch impl.abort("metal: out of memory building a fusion label");
}

/// Ports `ggml_metal_fusion_info_init` (ggml-metal-fusion.cpp:260
/// @b29c606e2).
///
/// Parameters:
/// - `enabled`: whether the encoders try to fuse at all.
/// - `debug`: `GGML_METAL_FUSION_DEBUG`; above 0 the counters are kept.
///
/// Return: a new info, owned by the caller and freed with
/// `ggml_metal_fusion_info_free`.
pub export fn ggml_metal_fusion_info_init(enabled: bool, debug: c_int) callconv(.c) *Info {
    const finfo = allocator.create(Info) catch impl.abort("metal: out of memory allocating the fusion info");
    finfo.* = .{
        .enabled = enabled,
        .stats = debug > 0,
        .labels_set = false,
        .debug = debug,
    };

    if (finfo.stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    return finfo;
}

/// Ports `ggml_metal_fusion_info_free` (ggml-metal-fusion.cpp:274
/// @b29c606e2).
pub export fn ggml_metal_fusion_info_free(finfo: ?*Info) callconv(.c) void {
    const self = finfo orelse return; // `delete nullptr` is a no-op
    freeLabels(self);
    allocator.destroy(self);
}

fn freeLabels(finfo: *Info) void {
    for (finfo.labels) |l| allocator.free(l);
    if (finfo.labels.len > 0) allocator.free(finfo.labels);
    if (finfo.counts.len > 0) allocator.free(finfo.counts);
    finfo.labels = &.{};
    finfo.counts = &.{};
}

/// Ports `ggml_metal_fusion_info_enabled` (ggml-metal-fusion.cpp:278
/// @b29c606e2).
pub export fn ggml_metal_fusion_info_enabled(finfo: *const Info) callconv(.c) bool {
    return finfo.enabled;
}

/// Ports `ggml_metal_fusion_info_stats` (ggml-metal-fusion.cpp:282
/// @b29c606e2).
pub export fn ggml_metal_fusion_info_stats(finfo: *const Info) callconv(.c) bool {
    return finfo.stats;
}

/// Ports `ggml_metal_fusion_info_debug` (ggml-metal-fusion.cpp:286
/// @b29c606e2).
pub export fn ggml_metal_fusion_info_debug(finfo: *const Info) callconv(.c) c_int {
    return finfo.debug;
}

/// Ports `ggml_metal_fusion_info_n_fusions` (ggml-metal-fusion.cpp:290
/// @b29c606e2). Zero until the labels are built.
pub export fn ggml_metal_fusion_info_n_fusions(finfo: *const Info) callconv(.c) c_int {
    return @intCast(finfo.labels.len);
}

/// Ports `ggml_metal_fusion_info_label` (ggml-metal-fusion.cpp:294
/// @b29c606e2).
///
/// Return: the label of table entry `idx`, borrowed from `finfo`.
pub export fn ggml_metal_fusion_info_label(finfo: *const Info, idx: c_int) callconv(.c) [*:0]const u8 {
    impl.assert(idx >= 0 and idx < @as(c_int, @intCast(finfo.labels.len)), "idx >= 0 && idx < (int) finfo->labels.size()");
    return finfo.labels[@intCast(idx)].ptr;
}

/// Ports `ggml_metal_fusion_info_count` (ggml-metal-fusion.cpp:299
/// @b29c606e2).
pub export fn ggml_metal_fusion_info_count(finfo: *const Info, idx: c_int) callconv(.c) u64 {
    impl.assert(idx >= 0 and idx < @as(c_int, @intCast(finfo.counts.len)), "idx >= 0 && idx < (int) finfo->counts.size()");
    return finfo.counts[@intCast(idx)];
}

/// Ports `ggml_metal_fusion_info_count_fusion` (ggml-metal-fusion.cpp:304
/// @b29c606e2): one more firing of `fusion`, found in the table by
/// address. Single-threaded by construction: the context encodes on one
/// thread whenever stats are on.
pub export fn ggml_metal_fusion_info_count_fusion(finfo: *Info, fusion: ?*const Fusion) callconv(.c) void {
    if (!finfo.stats or fusion == null) {
        return;
    }

    var n: c_int = 0;
    const all = ggml_metal_fusion_all(&n);

    var idx: c_int = -1;
    var i: c_int = 0;
    while (i < n) : (i += 1) {
        if (&all[@intCast(i)] == fusion.?) {
            idx = i;
            break;
        }
    }

    if (idx >= 0 and idx < @as(c_int, @intCast(finfo.counts.len))) {
        finfo.counts[@intCast(idx)] += 1;
    }
}

/// Ports `ggml_metal_fusion_info_set_enabled` (ggml-metal-fusion.cpp:325
/// @b29c606e2).
pub export fn ggml_metal_fusion_info_set_enabled(finfo: *Info, enabled: bool) callconv(.c) void {
    finfo.enabled = enabled;
}

/// Ports `ggml_metal_fusion_info_labels_init` (ggml-metal-fusion.cpp:329
/// @b29c606e2): one label and one zeroed counter per table entry, once.
pub export fn ggml_metal_fusion_info_labels_init(finfo: *Info) callconv(.c) void {
    if (finfo.labels_set) {
        return;
    }

    var n: c_int = 0;
    const all = ggml_metal_fusion_all(&n);
    const len: usize = @intCast(n);

    freeLabels(finfo);
    finfo.counts = allocator.alloc(u64, len) catch impl.abort("metal: out of memory allocating the fusion counters");
    @memset(finfo.counts, 0);
    finfo.labels = allocator.alloc([:0]u8, len) catch impl.abort("metal: out of memory allocating the fusion labels");

    for (0..len) |i| {
        finfo.labels[i] = label(&all[i]);
    }

    finfo.labels_set = true;
}

/// Ports `ggml_metal_fusion_info_stats_init` (ggml-metal-fusion.cpp:348
/// @b29c606e2).
pub export fn ggml_metal_fusion_info_stats_init(finfo: *Info) callconv(.c) void {
    finfo.stats = true;
    ggml_metal_fusion_info_labels_init(finfo);
}

/// Ports `ggml_metal_fusion_info_stats_reset` (ggml-metal-fusion.cpp:353
/// @b29c606e2).
pub export fn ggml_metal_fusion_info_stats_reset(finfo: *Info) callconv(.c) void {
    @memset(finfo.counts, 0);
}

/// Ports `ggml_metal_fusion_info_stats_get` (ggml-metal-fusion.cpp:357
/// @b29c606e2).
///
/// Parameters:
/// - `labels`: receives up to `n` labels, borrowed from `finfo`; null asks
///   only for the count.
/// - `counts`: receives the matching counters, or null.
/// - `n`: capacity of both arrays.
///
/// Return: the number of fusions when `labels` is null, else how many
/// entries were filled.
pub export fn ggml_metal_fusion_info_stats_get(finfo: *const Info, labels: ?[*][*:0]const u8, counts: ?[*]u64, n: c_int) callconv(.c) c_int {
    const n_fusions: c_int = @intCast(finfo.labels.len);

    const out = labels orelse return n_fusions;

    const n_fill = @min(n, n_fusions);
    var i: c_int = 0;
    while (i < n_fill) : (i += 1) {
        const k: usize = @intCast(i);
        out[k] = finfo.labels[k].ptr;
        if (counts) |cs| {
            cs[k] = finfo.counts[k];
        }
    }

    return n_fill;
}

// -----------------------------------------------------------------------------
// Queries

/// Ports `ggml_metal_fusion_next` (ggml-metal-fusion.cpp:379 @b29c606e2).
///
/// Parameters:
/// - `gf`: the graph.
/// - `node_idxs`: `n_idxs` graph node indices — the non-empty nodes.
/// - `idx`: a position in `node_idxs`, where the pattern must start.
/// - `mode`: structural (optimizer) or full (encoder) checks.
/// - `n_out`: receives the number of nodes consumed, 1 when nothing fuses.
///
/// Return: the longest matching table entry, the first of equal length
/// wins; null when nothing fuses.
pub export fn ggml_metal_fusion_next(
    gf: *const CGraph,
    node_idxs: [*]const c_int,
    n_idxs: c_int,
    idx: c_int,
    mode: Mode,
    n_out: *c_int,
) callconv(.c) ?*const Fusion {
    var n: c_int = 0;
    const all = ggml_metal_fusion_all(&n);

    var res: ?*const Fusion = null;
    var best: c_int = 1;

    var i: c_int = 0;
    while (i < n) : (i += 1) {
        const fusion = &all[@intCast(i)];

        // only look for a longer match than the current best
        if (fusion.n_ops <= best) {
            continue;
        }
        if (idx + fusion.n_ops > n_idxs) {
            continue;
        }

        var nodes: [max]*const Tensor = undefined;
        const n_ops: usize = @intCast(fusion.n_ops);

        // the op sequence must match exactly
        var ok = true;
        for (0..n_ops) |j| {
            nodes[j] = gf.nodes[@intCast(node_idxs[@as(usize, @intCast(idx)) + j])].?;
            if (nodes[j].op != fusion.ops[j]) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        if (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            var j: usize = 1;
            while (j < n_ops and ok) : (j += 1) {
                if (nodes[j].src[0] != nodes[j - 1] and nodes[j].src[1] != nodes[j - 1]) {
                    ok = false;
                    break;
                }
                if (!c.ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }

            // all current fusions are single-output elision chains, so the last node is the only output
            // TODO: multi-output fusions: store pattern-relative offsets in the table and translate them here
            const outputs_buf = [1]c_int{node_idxs[@as(usize, @intCast(idx)) + n_ops - 1]};

            // structural subgraph checks (op sequence, elidable uses, view containment)
            if (!graph.ggml_can_fuse_subgraph_ext(gf, node_idxs + @as(usize, @intCast(idx)), fusion.n_ops, fusion.ops, &outputs_buf, 1)) {
                continue;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        if (fusion.check) |check| {
            if (!check(fusion, &nodes, mode)) continue;
        }

        best = fusion.n_ops;
        res = fusion;
    }

    n_out.* = best;

    return res;
}

/// Ports `ggml_metal_fusion_max` (ggml-metal-fusion.cpp:464 @b29c606e2).
///
/// The optimize phase's question: how many nodes from `idx` (a raw graph
/// index) could be fused, chaining patterns back to back. Matching runs on
/// the same view-transparent sequence the encoders see, so the answer is
/// the raw span to the last matched node, views packed along.
///
/// Return: at least 1.
pub export fn ggml_metal_fusion_max(gf: *const CGraph, idx: c_int) callconv(.c) c_int {
    // an empty/view node cannot start a pattern - pack it alone
    const first = gf.nodes[@intCast(idx)].?;
    if (impl.opIsEmpty(first.op) or types.ggml_is_empty(first)) {
        return 1;
    }

    // collect the non-empty node indices starting at idx
    var idxs: [max]c_int = undefined;
    var n_idxs: c_int = 0;
    var i = idx;
    while (i < gf.n_nodes and n_idxs < max) : (i += 1) {
        const t = gf.nodes[@intCast(i)].?;
        if (!impl.opIsEmpty(t.op) and !types.ggml_is_empty(t)) {
            idxs[@intCast(n_idxs)] = i;
            n_idxs += 1;
        }
    }
    if (n_idxs == 0) {
        return 1;
    }

    var total: c_int = 0;
    var i_f: c_int = 0;

    while (i_f < n_idxs and total < max) {
        var len: c_int = 1;
        const fusion = ggml_metal_fusion_next(gf, &idxs, n_idxs, i_f, .structural, &len);
        if (fusion == null or total + len > max) {
            break;
        }

        total += len;
        i_f += len;
    }

    if (i_f == 0) {
        return 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    return @min(max, idxs[@intCast(i_f - 1)] - idx + 1);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the table's labels are its op names joined by +" {
    const finfo = ggml_metal_fusion_info_init(true, 1);
    defer ggml_metal_fusion_info_free(finfo);

    try std.testing.expectEqual(@as(c_int, fusions.len), ggml_metal_fusion_info_n_fusions(finfo));
    try std.testing.expectEqualStrings("NORM+MUL", std.mem.span(ggml_metal_fusion_info_label(finfo, 0)));
    try std.testing.expectEqualStrings("GATED_DELTA_NET+CPY", std.mem.span(ggml_metal_fusion_info_label(finfo, fusions.len - 1)));

    ggml_metal_fusion_info_count_fusion(finfo, &fusions[4]);
    ggml_metal_fusion_info_count_fusion(finfo, &fusions[4]);
    try std.testing.expectEqual(@as(u64, 2), ggml_metal_fusion_info_count(finfo, 4));

    ggml_metal_fusion_info_stats_reset(finfo);
    try std.testing.expectEqual(@as(u64, 0), ggml_metal_fusion_info_count(finfo, 4));
}

test "without debug the info keeps no labels until stats are asked for" {
    const finfo = ggml_metal_fusion_info_init(false, 0);
    defer ggml_metal_fusion_info_free(finfo);

    try std.testing.expect(!ggml_metal_fusion_info_enabled(finfo));
    try std.testing.expect(!ggml_metal_fusion_info_stats(finfo));
    try std.testing.expectEqual(@as(c_int, 0), ggml_metal_fusion_info_stats_get(finfo, null, null, 0));

    ggml_metal_fusion_info_stats_init(finfo);
    try std.testing.expect(ggml_metal_fusion_info_stats(finfo));
    try std.testing.expectEqual(@as(c_int, fusions.len), ggml_metal_fusion_info_stats_get(finfo, null, null, 0));
}
