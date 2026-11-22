//! The graph scheduler: assigns every node in a graph to a backend, splits the
//! graph where the backend changes, and runs the splits in order.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-backend.cpp` (v1.0.0, `b1b770d90`),
//! lines 901 to 2200 — the half the C marks off with its own `// scheduler`
//! comment. The other half is `src/ggml/backend.zig`, whose header carries the
//! provenance note for both. Each declaration below names the C++ it replaces
//! and the line it began at.
//!
//! # What it does
//!
//! `ggml_backend_sched_split_graph` is the whole of the interesting work, and
//! it runs in six passes over the graph:
//!
//! 1. Assign backends to nodes whose inputs are already allocated somewhere.
//! 2. Expand those assignments to adjacent unassigned nodes, four times: GPU
//!    down, GPU up, everything down, everything up. The GPU passes skip the
//!    lowest-priority backend so the CPU is never chosen by proximity.
//! 3. Upgrade nodes to a higher-priority backend with the same buffer type,
//!    and give each still-unassigned node the backend supporting the most of
//!    its inputs.
//! 4. Propagate assignments to remaining sources, and from `view_src`.
//! 5. Cut the graph into splits wherever the backend changes, creating a copy
//!    of any input that lives on an incompatible backend.
//! 6. With pipeline parallelism (`n_copies > 1`), give every input leaf its
//!    per-copy tensors and list it in `graph_inputs` -- every one, consumed or
//!    not, so the graph's shape does not depend on which inputs a batch uses.
//!
//! The order matters and the passes are not independent; they are reproduced
//! step for step.
//!
//! # `SET_CAUSE` is compiled out
//!
//! The C keeps a per-node string explaining why it landed on a backend, behind
//! an `#if 0` whose live `#else` defines `SET_CAUSE`
//! (src/ggml-backend.cpp:1333 @b1b770d90) as nothing. With that arm off, `SET_CAUSE`
//! expands to nothing and `GET_CAUSE` to `""`. This port has the same, as an
//! empty `setCause` and a `""` from `getCause`, so the call sites stay
//! readable against the C rather than disappearing.

const std = @import("std");
const mod = @import("module.zig");
const stdlib = @import("stdlib");

/// Ports `GGML_SCHED_MAX_BACKENDS` (src/ggml-backend.cpp:1164 @b1b770d90).
const max_backends = 16;

/// Ports `GGML_SCHED_MAX_SPLIT_INPUTS` (src/ggml-backend.cpp:1168 @b1b770d90).
///
/// Only the *initial* capacity: both input arrays grow by doubling, with a
/// warning, rather than overflowing.
const max_split_inputs = 30;

/// Ports `GGML_SCHED_MAX_COPIES` (src/ggml-backend.cpp:1172 @b1b770d90).
///
/// `build/llamacpp.zig` passes `-DGGML_SCHED_MAX_COPIES=4`, which is also the
/// C's default, so the `#ifndef` arm and the build agree.
const max_copies = 4;

/// Ports `struct ggml_backend_sched_split` (src/ggml-backend.cpp:1175 @b1b770d90).
const Split = extern struct {
    backend_id: c_int,
    i_start: c_int,
    i_end: c_int,
    inputs: [*c]?*mod.impl.c.ggml_tensor,
    n_inputs: c_int,
    inputs_capacity: c_int,
    /// Graph view of this split.
    graph: mod.impl.CGraph,
};

/// Ports `struct ggml_backend_sched` (src/ggml-backend.cpp:1186 @b1b770d90).
///
/// Laid out field for field as the C, because `ggml_backend_sched_t` is an
/// opaque pointer that crosses the ABI and the C++ that still holds one must
/// see the same object.
const Sched = extern struct {
    /// True if the scheduler has been reset since the last graph split.
    is_reset: bool,
    is_alloc: bool,

    n_backends: c_int,

    backends: [max_backends]mod.impl.c.ggml_backend_t,
    bufts: [max_backends]mod.impl.c.ggml_backend_buffer_type_t,
    galloc: mod.impl.c.ggml_gallocr_t,

    /// Hash map of the nodes in the graph.
    hash_set: mod.impl.HashSet,
    /// `[hash_set.size]`
    hv_tensor_backend_ids: [*c]c_int,
    /// `[hash_set.size][n_backends][n_copies]`
    hv_tensor_copies: [*c]?*mod.impl.c.ggml_tensor,
    /// `[hash_set.size]` decides if input needs to be copied to each split.
    hv_tensor_requires_copy: [*c]bool,

    /// `[graph_size]`
    node_backend_ids: [*c]c_int,
    /// `[graph_size]`
    leaf_backend_ids: [*c]c_int,

    /// `[graph_size]`
    prev_node_backend_ids: [*c]c_int,
    /// `[graph_size]`
    prev_leaf_backend_ids: [*c]c_int,

    /// Copy of the graph with modified inputs.
    graph: mod.impl.CGraph,

    /// Graph splits.
    splits: [*c]Split,
    n_splits: c_int,
    splits_capacity: c_int,

    // Pipeline parallelism support.
    n_copies: c_int,
    cur_copy: c_int,
    next_copy: c_int,
    events: [max_backends][max_copies]mod.impl.c.ggml_backend_event_t,
    graph_inputs: [*c]?*mod.impl.c.ggml_tensor,
    n_graph_inputs: c_int,
    graph_inputs_capacity: c_int,

    ctx: ?*mod.impl.c.ggml_context,

    callback_eval: mod.impl.c.ggml_backend_sched_eval_callback,
    callback_eval_user_data: ?*anyopaque,

    callback_copy: mod.impl.c.ggml_backend_sched_copy_callback,
    callback_copy_user_data: ?*anyopaque,

    callback_realloc: mod.impl.c.ggml_backend_sched_realloc_callback,
    callback_realloc_user_data: ?*anyopaque,

    context_buffer: [*c]u8,
    context_buffer_size: usize,

    op_offload: bool,

    debug: c_int,

    // Graph size of the current and the previous scheduling, used by the
    // realloc callback and for debugging graph reallocations
    // [GGML_SCHED_DEBUG_REALLOC].
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    graph_size: c_int,
    prev_graph_size: c_int,

    debug_realloc: c_int,

    // Inputs of the next split read from the backend of the current split
    // before it computes, see `prefetchInputs`.
    /// Created on first use, null if the backend has no events.
    prefetch_events: [max_backends]mod.impl.c.ggml_backend_event_t,
    prefetch_tried: [max_backends]bool,
    prefetched: [max_split_inputs]?*mod.impl.c.ggml_tensor,
    n_prefetched: c_int,
    /// The split that reads the prefetched inputs.
    prefetch_split: c_int,
    /// The backend that copies them.
    prefetch_backend_id: c_int,
};

/// Narrows the opaque `ggml_backend_sched_t` the C ABI passes.
inline fn schedOf(p: mod.impl.c.ggml_backend_sched_t) *Sched {
    return @ptrCast(@alignCast(p.?));
}

// -----------------------------------------------------------------------------
// The four addressing macros
//
// Ports `hash_id`, `tensor_backend_id`, `tensor_id_copy` and `tensor_copy`
// (src/ggml-backend.cpp:1260, 1261, 1262, 1263 @b1b770d90). The C writes them as
// macros so they can appear on the left of an assignment; Zig returns a
// pointer where the C yields an lvalue, and the call sites dereference.

/// Ports the `hash_id` macro (src/ggml-backend.cpp:1263 @b1b770d90).
inline fn hashId(sched: *Sched, tensor: *mod.impl.c.ggml_tensor) usize {
    return mod.impl.hashFindOrInsert(&sched.hash_set, tensor);
}

/// Ports the `tensor_backend_id` macro (src/ggml-backend.cpp:1261 @b1b770d90),
/// as a pointer so it can be assigned through.
inline fn tensorBackendIdPtr(sched: *Sched, tensor: *mod.impl.c.ggml_tensor) *c_int {
    return &mod.impl.many(c_int, sched.hv_tensor_backend_ids)[hashId(sched, tensor)];
}

/// Reading form of `tensor_backend_id` (src/ggml-backend.cpp:1261 @b1b770d90).
inline fn tensorBackendId(sched: *Sched, tensor: *mod.impl.c.ggml_tensor) c_int {
    return tensorBackendIdPtr(sched, tensor).*;
}

/// Ports the `tensor_id_copy` macro (src/ggml-backend.cpp:1262 @b1b770d90).
inline fn tensorIdCopyPtr(sched: *Sched, id: usize, backend_id: c_int, copy_id: c_int) *?*mod.impl.c.ggml_tensor {
    const nb: usize = @intCast(sched.n_backends);
    const nc: usize = @intCast(sched.n_copies);
    const b: usize = @intCast(backend_id);
    const cp: usize = @intCast(copy_id);
    return &mod.impl.many(?*mod.impl.c.ggml_tensor, sched.hv_tensor_copies)[id * nb * nc + b * nc + cp];
}

/// Ports the `tensor_copy` macro (src/ggml-backend.cpp:1263 @b1b770d90).
inline fn tensorCopy(sched: *Sched, tensor: *mod.impl.c.ggml_tensor, backend_id: c_int, copy_id: c_int) ?*mod.impl.c.ggml_tensor {
    return tensorIdCopyPtr(sched, hashId(sched, tensor), backend_id, copy_id).*;
}

/// Ports the `SET_CAUSE` macro (src/ggml-backend.cpp:1333 @b1b770d90), the
/// `#else` arm that this build compiles: the debug arm is behind `#if 0`.
inline fn setCause(tensor: ?*mod.impl.c.ggml_tensor, comptime why: []const u8) void {
    _ = tensor;
    _ = why;
}

/// Ports the `GET_CAUSE` macro (src/ggml-backend.cpp:1334 @b1b770d90), likewise
/// the compiled arm.
inline fn getCause(tensor: ?*mod.impl.c.ggml_tensor) [*:0]const u8 {
    _ = tensor;
    return "";
}

// -----------------------------------------------------------------------------
// Growing the input arrays

/// Ports `ggml_backend_sched_split_inputs_grow` (src/ggml-backend.cpp:1265 @b1b770d90).
fn splitInputsGrow(split: *Split) void {
    var new_cap: c_int = max_split_inputs;
    if (split.inputs_capacity > 0) {
        new_cap = 2 * split.inputs_capacity;
        mod.impl.logDebug(
            "%s: increasing split inputs capacity from %d to %d\n",
            .{ "ggml_backend_sched_split_inputs_grow", split.inputs_capacity, new_cap },
        );
    }
    const bytes = @as(usize, @intCast(new_cap)) * @sizeOf(?*mod.impl.c.ggml_tensor);
    const pnew = std.c.realloc(@ptrCast(split.inputs), bytes);
    if (pnew == null) {
        mod.impl.logError("%s: failed to allocate %zu bytes\n", .{ "ggml_backend_sched_split_inputs_grow", bytes });
        mod.impl.abort("failed to grow split inputs container");
    }
    split.inputs = @ptrCast(@alignCast(pnew));
    split.inputs_capacity = new_cap;
}

/// Ports `ggml_backend_sched_graph_inputs_grow` (src/ggml-backend.cpp:1280 @b1b770d90).
fn graphInputsGrow(sched: *Sched) void {
    var new_cap: c_int = max_split_inputs;
    if (sched.graph_inputs_capacity > 0) {
        new_cap = 2 * sched.graph_inputs_capacity;
        mod.impl.logDebug(
            "%s: increasing graph inputs capacity from %d to %d\n",
            .{ "ggml_backend_sched_graph_inputs_grow", sched.graph_inputs_capacity, new_cap },
        );
    }
    const bytes = @as(usize, @intCast(new_cap)) * @sizeOf(?*mod.impl.c.ggml_tensor);
    const pnew = std.c.realloc(@ptrCast(sched.graph_inputs), bytes);
    if (pnew == null) {
        mod.impl.logError("%s: failed to allocate %zu bytes\n", .{ "ggml_backend_sched_graph_inputs_grow", bytes });
        mod.impl.abort("failed to grow graph inputs container");
    }
    sched.graph_inputs = @ptrCast(@alignCast(pnew));
    sched.graph_inputs_capacity = new_cap;
}

// -----------------------------------------------------------------------------
// Choosing a backend for a node

/// Ports `ggml_backend_sched_backend_id` (src/ggml-backend.cpp:1296 @b1b770d90).
///
/// Return: the backend's priority index, lower being higher priority, or -1.
fn backendId(sched: *Sched, b: mod.impl.c.ggml_backend_t) c_int {
    var i: c_int = 0;
    while (i < sched.n_backends) : (i += 1) {
        if (sched.backends[@intCast(i)] == b) {
            return i;
        }
    }
    return -1;
}

/// Ports `ggml_backend_sched_backend_from_buffer` (src/ggml-backend.cpp:1305 @b1b770d90).
///
/// Return: the highest-priority backend that both supports `tensor`'s buffer
/// type and can run `op`, or -1 when none can — in which case the weight will
/// have to be copied.
fn backendFromBuffer(sched: *Sched, tensor: *const mod.impl.c.ggml_tensor, op: *const mod.impl.c.ggml_tensor) c_int {
    const buffer = if (tensor.view_src != null) tensor.view_src.*.buffer else tensor.buffer;
    if (buffer == null) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    var i: c_int = 0;
    while (i < sched.n_backends) : (i += 1) {
        if (mod.backend.ggml_backend_supports_buft(sched.backends[@intCast(i)], buffer.*.buft) and
            mod.backend.ggml_backend_supports_op(sched.backends[@intCast(i)], op))
        {
            return i;
        }
    }

    if (!debug_off) {
        mod.impl.logDebug(
            "%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
            .{
                "ggml_backend_sched_backend_from_buffer",
                mod.impl.c.ggml_op_desc(tensor),
                mod.backend.ggml_backend_buffer_name(buffer),
                @as([*:0]const u8, @ptrCast(&tensor.name)),
            },
        );
    }

    return -1;
}

/// Mirrors `NDEBUG`, which gates the diagnostics in this file.
const debug_off = @import("builtin").mode != .debug;

/// Ports `ggml_backend_sched_backend_id_from_cur` (src/ggml-backend.cpp:1338 @b1b770d90).
///
/// Return: the backend this node should run on given where its data already
/// is, or -1 if nothing decides it yet. Aborts when a pre-allocated tensor
/// sits in a buffer whose backend cannot run its op, since it cannot be moved.
fn backendIdFromCur(sched: *Sched, tensor: *mod.impl.c.ggml_tensor) c_int {
    // assign pre-allocated nodes to their backend
    var cur_backend_id = backendFromBuffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        setCause(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor.view_src != null) {
        cur_backend_id = backendFromBuffer(sched, mod.impl.one(mod.impl.c.ggml_tensor, tensor.view_src), tensor);
        if (cur_backend_id != -1) {
            setCause(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor.buffer != null or (tensor.view_src != null and tensor.view_src.*.buffer != null)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        mod.impl.abort("pre-allocated tensor in a buffer that cannot run the operation");
    }

    // graph input
    if (tensor.flags & mod.impl.c.GGML_TENSOR_FLAG_INPUT != 0) {
        cur_backend_id = sched.n_backends - 1; // last backend (assumed CPU)
        setCause(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    var allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow and tensor.op != mod.impl.c.GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow and tensor.op != mod.impl.c.GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (0..mod.impl.c.GGML_MAX_SRC) |i| {
            const src_c = tensor.src[i];
            if (src_c == null) {
                continue;
            }
            const src = mod.impl.one(mod.impl.c.ggml_tensor, src_c);
            if (src.buffer != null and src.buffer.*.usage == mod.impl.c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                const src_backend_id = backendFromBuffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched.op_offload and src_backend_id == sched.n_backends - 1 and
                    mod.backend.ggml_backend_buffer_is_host(src.buffer))
                {
                    var b: c_int = 0;
                    while (b < src_backend_id) : (b += 1) {
                        if (mod.backend.ggml_backend_supports_op(sched.backends[@intCast(b)], tensor) and
                            mod.backend.ggml_backend_offload_op(sched.backends[@intCast(b)], tensor))
                        {
                            setCause(tensor, "1.off");
                            return b;
                        }
                    }
                }
                setCause(tensor, "1.wgt");
                return src_backend_id;
            }
        }
    }

    return -1;
}

/// Ports `fmt_size` (src/ggml-backend.cpp:1404 @b1b770d90).
///
/// Uses one process-wide buffer, exactly as the C's `static char buffer[128]`
/// does, so two calls in one format string would collide — and the C has that
/// hazard too. Only reachable from the debug printer below.
var fmt_size_buffer: [128]u8 = undefined;

fn fmtSize(size: usize) [*:0]const u8 {
    const s = if (size >= 1024 * 1024)
        std.mem.printSentinel(&fmt_size_buffer, "{d}M", .{size / 1024 / 1024}, 0)
    else
        std.mem.printSentinel(&fmt_size_buffer, "{d}K", .{size / 1024}, 0);
    const out = s catch blk: {
        // 128 bytes is far more than any size needs; the C has no fallback
        // because `snprintf` cannot fail here either.
        break :blk std.mem.printSentinel(&fmt_size_buffer, "?", .{}, 0) catch unreachable;
    };
    return out.ptr;
}

/// Ports `ggml_backend_sched_print_assignments` (src/ggml-backend.cpp:1414 @b1b770d90).
///
/// Only runs when `GGML_SCHED_DEBUG` is set in the environment.
fn printAssignments(sched: *Sched, graph: *mod.impl.CGraph) void {
    var cur_split: c_int = 0;
    var i: c_int = 0;
    while (i < graph.n_nodes) : (i += 1) {
        if (cur_split < sched.n_splits and i == sched.splits[@intCast(cur_split)].i_start) {
            const split = &mod.impl.many(Split, sched.splits)[@intCast(cur_split)];
            const split_backend = sched.backends[@intCast(split.backend_id)];
            mod.impl.logDebug(
                "\n## SPLIT #%d: %s # %d inputs",
                .{ cur_split, mod.backend.ggml_backend_name(split_backend), split.n_inputs },
            );
            var j: c_int = 0;
            while (j < split.n_inputs) : (j += 1) {
                if (j == 0) {
                    mod.impl.logDebug(": ", .{});
                }
                const in = split.inputs[@intCast(j)].?;
                mod.impl.logDebug("[%s (%5.5s)] ", .{
                    @as([*:0]const u8, @ptrCast(&in.name)),
                    fmtSize(mod.impl.c.ggml_nbytes(in)),
                });
            }
            mod.impl.logDebug("\n", .{});
            cur_split += 1;
        }
        const node = graph.nodes[@intCast(i)].?;
        if (mod.backend.isViewOp(node.op)) {
            continue;
        }
        if (sched.debug > 1) {
            const tensor_backend = ggml_backend_sched_get_tensor_backend(@ptrCast(sched), node);
            mod.impl.logDebug("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", .{
                i,
                mod.impl.c.ggml_op_desc(node),
                @as([*:0]const u8, @ptrCast(&node.name)),
                fmtSize(mod.impl.c.ggml_nbytes(node)),
                if (tensor_backend != null) mod.backend.ggml_backend_name(tensor_backend) else @as([*c]const u8, "NULL"),
                getCause(node),
                graph.use_counts[mod.impl.hashFind(&graph.visited_hash_set, node)],
                @as(c_int, if (node.flags & mod.impl.c.GGML_TENSOR_FLAG_COMPUTE != 0) 1 else 0),
            });
            for (0..mod.impl.c.GGML_MAX_SRC) |j| {
                const src_c = node.src[j];
                if (src_c == null) {
                    continue;
                }
                const src = mod.impl.one(mod.impl.c.ggml_tensor, src_c);
                const src_backend = ggml_backend_sched_get_tensor_backend(@ptrCast(sched), src);
                mod.impl.logDebug(" %20.20s (%5.5s) [%5.5s %8.8s]", .{
                    @as([*:0]const u8, @ptrCast(&src.name)),
                    fmtSize(mod.impl.c.ggml_nbytes(src)),
                    if (src_backend != null) mod.backend.ggml_backend_name(src_backend) else @as([*c]const u8, "NULL"),
                    getCause(src),
                });
            }
            mod.impl.logDebug("\n", .{});
        }
    }
}

/// Ports `ggml_backend_sched_buffer_supported` (src/ggml-backend.cpp:1454 @b1b770d90).
///
/// Return: whether `backend_id` can use the buffer type `t` already lives in,
/// or the one its assigned backend would give it.
fn bufferSupported(sched: *Sched, t: *mod.impl.c.ggml_tensor, backend_id: c_int) bool {
    const buf = if (t.view_src != null) t.view_src.*.buffer else t.buffer;
    var buft: mod.impl.c.ggml_backend_buffer_type_t = null;

    if (buf != null) {
        // the tensor is already allocated
        buft = buf.*.buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer
        // type of that backend
        var tbid = tensorBackendId(sched, t);
        if (tbid == -1 and t.view_src != null) {
            tbid = tensorBackendId(sched, mod.impl.one(mod.impl.c.ggml_tensor, t.view_src));
        }
        if (tbid != -1) {
            buft = sched.bufts[@intCast(tbid)];
        }
    }

    return buft != null and mod.backend.ggml_backend_supports_buft(sched.backends[@intCast(backend_id)], buft);
}

/// Ports `ggml_backend_sched_set_if_supported` (src/ggml-backend.cpp:1475 @b1b770d90).
fn setIfSupported(sched: *Sched, node: *mod.impl.c.ggml_tensor, cur_backend_id: c_int, node_backend_id: *c_int) void {
    if (mod.backend.ggml_backend_supports_op(sched.backends[@intCast(cur_backend_id)], node)) {
        node_backend_id.* = cur_backend_id;
        setCause(node, "2.sup");
    }
}

// -----------------------------------------------------------------------------
// Splitting the graph

/// `GGML_MAX_SRC` as a `usize`, for the dependency nodes' source slots.
const max_src: usize = mod.impl.c.GGML_MAX_SRC;

/// The allocation dependencies backends add from `graph_optimize`: for each
/// `until` node, the tensors to keep allocated until it has run, in the order
/// first added. The C++'s `std::unordered_map<ggml_tensor *,
/// std::vector<ggml_tensor *>>`; only its lookups and its total size are
/// read, never its iteration order.
const AllocDeps = std.AutoHashMapUnmanaged(*mod.impl.c.ggml_tensor, std.ArrayList(*mod.impl.c.ggml_tensor));

/// The lambda `ggml_backend_sched_split_graph` passes to `graph_optimize` as
/// `add_alloc_dep` (src/ggml-backend.cpp:1886 @b1b770d90): keeps `tensor`
/// allocated until `until` is computed; a repeat is ignored, so the longest
/// lifetime applies.
fn addAllocDep(user_data: ?*anyopaque, tensor: [*c]mod.impl.c.ggml_tensor, until: [*c]mod.impl.c.ggml_tensor) callconv(.c) void {
    const deps: *AllocDeps = @ptrCast(@alignCast(user_data.?));
    const gop = deps.getOrPut(std.heap.c_allocator, mod.impl.one(mod.impl.c.ggml_tensor, until)) catch mod.impl.abort("sched: out of memory");
    if (!gop.found_existing) gop.value_ptr.* = .empty;
    const t = mod.impl.one(mod.impl.c.ggml_tensor, tensor);
    for (gop.value_ptr.items) |kept| if (kept == t) return;
    gop.value_ptr.append(std.heap.c_allocator, t) catch mod.impl.abort("sched: out of memory");
}

/// Ports `ggml_backend_sched_split_graph` (src/ggml-backend.cpp:1483 @b1b770d90).
///
/// Assigns backends to ops and splits the graph into subgraphs that can be
/// computed on the same backend. The six passes are described in the file
/// header; each is kept in its own block, as in the C.
///
/// Parameters:
/// - `sched_`: the scheduler; its `splits`, `graph` and hash table are rebuilt.
/// - `graph`: the graph to split. Its nodes' `src[]` are rewritten in place to
///   point at the copies this creates.
export fn ggml_backend_sched_split_graph(sched_: mod.impl.c.ggml_backend_sched_t, graph: *mod.impl.CGraph) callconv(.c) void {
    const sched = schedOf(sched_);

    // reset splits
    sched.n_splits = 0;
    sched.n_graph_inputs = 0;
    sched.is_reset = false;

    const params: mod.impl.c.ggml_init_params = .{
        .mem_size = sched.context_buffer_size,
        .mem_buffer = sched.context_buffer,
        .no_alloc = true,
    };

    mod.impl.c.ggml_free(sched.ctx);

    sched.ctx = mod.impl.c.ggml_init(params);
    if (sched.ctx == null) {
        mod.impl.abort("ggml_backend_sched_split_graph: failed to initialize context");
    }

    graph.uid = mod.runtime.ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    {
        var i: c_int = 0;
        while (i < graph.n_leafs) : (i += 1) {
            const leaf = graph.leafs[@intCast(i)].?;
            const leaf_backend_id = tensorBackendIdPtr(sched, leaf);
            // do not overwrite user assignments
            if (leaf_backend_id.* == -1) {
                leaf_backend_id.* = backendIdFromCur(sched, leaf);
            }
        }

        i = 0;
        while (i < graph.n_nodes) : (i += 1) {
            const node = graph.nodes[@intCast(i)].?;
            const node_backend_id = tensorBackendIdPtr(sched, node);
            // do not overwrite user assignments
            if (node_backend_id.* == -1) {
                node_backend_id.* = backendIdFromCur(sched, node);
            }
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        var cur_backend_id: c_int = -1;
        var i: c_int = 0;
        while (i < graph.n_nodes) : (i += 1) {
            const node = graph.nodes[@intCast(i)].?;
            if (mod.backend.isViewOp(node.op)) {
                continue;
            }
            const node_backend_id = tensorBackendIdPtr(sched, node);
            if (node_backend_id.* != -1) {
                if (node_backend_id.* == sched.n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = node_backend_id.*;
                }
            } else if (cur_backend_id != -1) {
                setIfSupported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        var cur_backend_id: c_int = -1;
        var i: c_int = graph.n_nodes - 1;
        while (i >= 0) : (i -= 1) {
            const node = graph.nodes[@intCast(i)].?;
            if (mod.backend.isViewOp(node.op)) {
                continue;
            }
            const node_backend_id = tensorBackendIdPtr(sched, node);
            if (node_backend_id.* != -1) {
                if (node_backend_id.* == sched.n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = node_backend_id.*;
                }
            } else if (cur_backend_id != -1) {
                setIfSupported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        var cur_backend_id: c_int = -1;
        var i: c_int = 0;
        while (i < graph.n_nodes) : (i += 1) {
            const node = graph.nodes[@intCast(i)].?;
            if (mod.backend.isViewOp(node.op)) {
                continue;
            }
            const node_backend_id = tensorBackendIdPtr(sched, node);
            if (node_backend_id.* != -1) {
                cur_backend_id = node_backend_id.*;
            } else if (cur_backend_id != -1) {
                setIfSupported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        var cur_backend_id: c_int = -1;
        var i: c_int = graph.n_nodes - 1;
        while (i >= 0) : (i -= 1) {
            const node = graph.nodes[@intCast(i)].?;
            if (mod.backend.isViewOp(node.op)) {
                continue;
            }
            const node_backend_id = tensorBackendIdPtr(sched, node);
            if (node_backend_id.* != -1) {
                cur_backend_id = node_backend_id.*;
            } else if (cur_backend_id != -1) {
                setIfSupported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    {
        var i: c_int = 0;
        while (i < graph.n_nodes) : (i += 1) {
            const node = graph.nodes[@intCast(i)].?;
            if (mod.backend.isViewOp(node.op)) {
                continue;
            }
            const node_backend_id = tensorBackendIdPtr(sched, node);
            if (node_backend_id.* == -1) {
                // unassigned node: find the backend with the most supported inputs
                var n_supported_best: c_int = -1;
                var b: c_int = 0;
                while (b < sched.n_backends) : (b += 1) {
                    if (mod.backend.ggml_backend_supports_op(sched.backends[@intCast(b)], node)) {
                        var n_supported: c_int = 0;
                        for (0..mod.impl.c.GGML_MAX_SRC) |j| {
                            const src_c = node.src[j];
                            if (src_c == null) {
                                continue;
                            }
                            const src = mod.impl.one(mod.impl.c.ggml_tensor, src_c);
                            // **A deliberate difference.** The C writes this
                            // with `tensor_backend_id` applied to both `src`
                            // and `src->view_src` in one condition, at
                            // `n_supported` (src/ggml-backend.cpp:1640
                            // @b1b770d90), and `||`
                            // short-circuits, so the second term is evaluated
                            // whenever the first is false -- with `view_src`
                            // null for most tensors. `hash_id` then inserts a
                            // *null key* into a hash set of tensors, burning
                            // one slot per reset and returning an id whose
                            // stored value is -1, so the condition is false
                            // anyway. Guarding it changes nothing observable:
                            // the entry is never looked up again and no
                            // tensor's own id moves. Same call as
                            // `quantize_row_iq4_nl_ref` -- where the C's
                            // answer is accidental, reproducing it is not the
                            // goal.
                            const has_id = tensorBackendId(sched, src) != -1 or
                                (src.view_src != null and tensorBackendId(sched, mod.impl.one(mod.impl.c.ggml_tensor, src.view_src)) != -1);
                            if (has_id and bufferSupported(sched, src, b)) {
                                n_supported += 1;
                            }
                        }
                        if (n_supported > n_supported_best) {
                            n_supported_best = n_supported;
                            node_backend_id.* = b;
                            setCause(node, "3.best");
                        }
                    }
                }
            } else {
                // assigned node: upgrade to higher prio backend if possible
                var b: c_int = 0;
                while (b < node_backend_id.*) : (b += 1) {
                    if (sched.bufts[@intCast(b)] == sched.bufts[@intCast(node_backend_id.*)] and
                        mod.backend.ggml_backend_supports_op(sched.backends[@intCast(b)], node))
                    {
                        var supported = true;
                        for (0..mod.impl.c.GGML_MAX_SRC) |j| {
                            const src_c = node.src[j];
                            if (src_c == null) {
                                continue;
                            }
                            const src = mod.impl.one(mod.impl.c.ggml_tensor, src_c);
                            if (!bufferSupported(sched, src, b)) {
                                supported = false;
                                break;
                            }
                        }
                        if (supported) {
                            node_backend_id.* = b;
                            setCause(node, "3.upg");
                            break;
                        }
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    {
        var i: c_int = 0;
        while (i < graph.n_nodes) : (i += 1) {
            const node = graph.nodes[@intCast(i)].?;
            const cur_backend_id = tensorBackendIdPtr(sched, node);
            if (node.view_src != null and cur_backend_id.* == -1) {
                cur_backend_id.* = tensorBackendId(sched, mod.impl.one(mod.impl.c.ggml_tensor, node.view_src));
                setCause(node, "4.vsrc");
            }
            for (0..mod.impl.c.GGML_MAX_SRC) |j| {
                const src_c = node.src[j];
                if (src_c == null) {
                    continue;
                }
                const src = mod.impl.one(mod.impl.c.ggml_tensor, src_c);
                const src_backend_id = tensorBackendIdPtr(sched, src);
                if (src_backend_id.* == -1) {
                    if (src.view_src != null) {
                        // views are always on the same backend as the source
                        src_backend_id.* = tensorBackendId(sched, mod.impl.one(mod.impl.c.ggml_tensor, src.view_src));
                        setCause(src, "4.vsrc");
                    } else {
                        src_backend_id.* = cur_backend_id.*;
                        setCause(src, "4.cur");
                    }
                }
            }
            // if the node is still unassigned, assign it to the first backend that supports it
            var b: c_int = 0;
            while (b < sched.n_backends and cur_backend_id.* == -1) : (b += 1) {
                setIfSupported(sched, node, b, cur_backend_id);
            }
            mod.impl.assert(cur_backend_id.* != -1, "*cur_backend_id != -1");
        }
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        var i_split: c_int = 0;
        var split: *Split = &mod.impl.many(Split, sched.splits)[0];
        // find the backend of the first split, skipping view ops
        var i: c_int = 0;
        while (i < graph.n_nodes) : (i += 1) {
            const node = graph.nodes[@intCast(i)].?;
            if (!mod.backend.isViewOp(node.op)) {
                split.backend_id = tensorBackendId(sched, node);
                break;
            }
        }
        split.i_start = 0;
        split.n_inputs = 0;
        var cur_backend_id = split.backend_id;
        while (i < graph.n_nodes) : (i += 1) {
            const node = graph.nodes[@intCast(i)].?;

            if (mod.backend.isViewOp(node.op)) {
                continue;
            }

            const node_backend_id = tensorBackendId(sched, node);

            // all nodes should be assigned by now; this can happen if there is
            // no CPU fallback
            mod.impl.assert(node_backend_id != -1, "node_backend_id != -1");

            // check if we should start a new split based on the sources of the current node
            var need_new_split = false;
            if (node_backend_id == cur_backend_id and split.n_inputs > 0) {
                for (0..mod.impl.c.GGML_MAX_SRC) |j| {
                    const src_c = node.src[j];
                    if (src_c == null) {
                        continue;
                    }
                    const src = mod.impl.one(mod.impl.c.ggml_tensor, src_c);
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src.buffer != null and src.buffer.*.usage == mod.impl.c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        const src_backend_id = tensorBackendId(sched, src);
                        if (src_backend_id != cur_backend_id and !bufferSupported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id or need_new_split) {
                split.i_end = i;
                i_split += 1;
                if (i_split >= sched.splits_capacity) {
                    const old_cap = sched.splits_capacity;
                    sched.splits_capacity *= 2;
                    sched.splits = @ptrCast(@alignCast(std.c.realloc(
                        @ptrCast(sched.splits),
                        @as(usize, @intCast(sched.splits_capacity)) * @sizeOf(Split),
                    )));
                    mod.impl.assert(sched.splits != null, "sched->splits != NULL");
                    var k: c_int = old_cap;
                    while (k < sched.splits_capacity) : (k += 1) {
                        sched.splits[@intCast(k)] = std.mem.zeroes(Split);
                    }
                }
                split = &mod.impl.many(Split, sched.splits)[@intCast(i_split)];
                split.backend_id = node_backend_id;
                split.i_start = i;
                split.n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (0..mod.impl.c.GGML_MAX_SRC) |j| {
                const src_c = node.src[j];
                if (src_c == null) {
                    continue;
                }
                const src = mod.impl.one(mod.impl.c.ggml_tensor, src_c);

                const src_id = hashId(sched, src);
                const src_backend_id = sched.hv_tensor_backend_ids[src_id];
                // all inputs should be assigned by now
                mod.impl.assert(src_backend_id != -1, "src_backend_id != -1");

                if (src_backend_id != cur_backend_id and !bufferSupported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensorIdCopyPtr(sched, src_id, cur_backend_id, 0).* == null) {
                        const b = sched.backends[@intCast(cur_backend_id)];
                        var cp: c_int = 0;
                        while (cp < sched.n_copies) : (cp += 1) {
                            const tensor_copy = mod.impl.one(mod.impl.c.ggml_tensor, mod.backend.dupTensorLayout(sched.ctx, src));
                            _ = mod.impl.c.ggml_format_name(tensor_copy, "%s#%s#%d", mod.backend.ggml_backend_name(b), @as([*:0]const u8, @ptrCast(&src.name)), cp);
                            if (sched.n_copies > 1) {
                                _ = mod.impl.c.ggml_set_input(tensor_copy);
                                _ = mod.impl.c.ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensorIdCopyPtr(sched, src_id, cur_backend_id, cp).* = tensor_copy;
                            setCause(tensor_copy, "4.cpy");
                        }
                        const n_inputs = split.n_inputs;
                        split.n_inputs += 1;
                        if (n_inputs >= split.inputs_capacity) {
                            splitInputsGrow(split);
                        }
                        split.inputs[@intCast(n_inputs)] = src;
                    }
                    // inputs of nodes that are not computed (e.g. the unselected
                    // branches of ggml_build_forward_select) stay in the split so
                    // that the allocation does not change, but they do not need
                    // to be copied
                    if (node.flags & mod.impl.c.GGML_TENSOR_FLAG_COMPUTE != 0) {
                        mod.impl.many(bool, sched.hv_tensor_requires_copy)[src_id] = true;
                    }
                    node.src[j] = tensorIdCopyPtr(sched, src_id, cur_backend_id, sched.cur_copy).*;
                }
            }
        }
        split.i_end = graph.n_nodes;
        sched.n_splits = i_split + 1;
    }

    if (sched.debug != 0) {
        printAssignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    if (sched.n_copies > 1) {
        for (0..@intCast(graph.n_leafs)) |i| {
            const leaf = mod.impl.one(mod.impl.c.ggml_tensor, graph.leafs[i]);
            if ((leaf.flags & mod.impl.c.GGML_TENSOR_FLAG_INPUT) == 0) {
                continue;
            }

            const leaf_id = hashId(sched, leaf);
            const leaf_backend_id = tensorBackendId(sched, leaf);
            mod.impl.assert(leaf_backend_id != -1, "leaf_backend_id != -1"); // all leafs should be assigned by now

            if (tensorIdCopyPtr(sched, leaf_id, leaf_backend_id, 0).* == null) {
                const b = sched.backends[@intCast(leaf_backend_id)];
                var cp: c_int = 0;
                while (cp < sched.n_copies) : (cp += 1) {
                    var tensor_copy: *mod.impl.c.ggml_tensor = undefined;
                    if (cp == sched.cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } else {
                        tensor_copy = mod.impl.one(mod.impl.c.ggml_tensor, mod.backend.dupTensorLayout(sched.ctx, leaf));
                        _ = mod.impl.c.ggml_format_name(tensor_copy, "%s#%s#%d", mod.backend.ggml_backend_name(b), @as([*:0]const u8, @ptrCast(&leaf.name)), cp);
                    }
                    _ = mod.impl.c.ggml_set_input(tensor_copy);
                    _ = mod.impl.c.ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensorIdCopyPtr(sched, leaf_id, leaf_backend_id, cp).* = tensor_copy;
                    setCause(tensor_copy, "6.cpy");
                }
            }

            const n_graph_inputs = sched.n_graph_inputs;
            sched.n_graph_inputs += 1;
            if (n_graph_inputs >= sched.graph_inputs_capacity) {
                graphInputsGrow(sched);
            }
            sched.graph_inputs[@intCast(n_graph_inputs)] = leaf;
        }
    }

    // swap node_backend_ids and leaf_backend_ids with prevs
    {
        const tmp_nodes = sched.node_backend_ids;
        sched.node_backend_ids = sched.prev_node_backend_ids;
        sched.prev_node_backend_ids = tmp_nodes;

        const tmp_leafs = sched.leaf_backend_ids;
        sched.leaf_backend_ids = sched.prev_leaf_backend_ids;
        sched.prev_leaf_backend_ids = tmp_leafs;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    var alloc_deps: AllocDeps = .empty;
    defer {
        var it = alloc_deps.valueIterator();
        while (it.next()) |keep| keep.deinit(std.heap.c_allocator);
        alloc_deps.deinit(std.heap.c_allocator);
    }

    var opt_params: mod.impl.c.struct_ggml_backend_graph_optimize_params = .{
        .add_alloc_dep = addAllocDep,
        .user_data = &alloc_deps,
    };

    {
        var i: c_int = 0;
        while (i < sched.n_splits) : (i += 1) {
            const split = &mod.impl.many(Split, sched.splits)[@intCast(i)];
            split.graph = mod.graph.ggml_graph_view(graph, split.i_start, split.i_end);

            mod.backend.graphOptimize(sched.backends[@intCast(split.backend_id)], &split.graph, &opt_params);
        }
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    var n_dep_nodes: c_int = 0;
    {
        var it = alloc_deps.valueIterator();
        while (it.next()) |keep| {
            n_dep_nodes += @intCast((keep.items.len + max_src - 1) / max_src);
        }
    }

    var total_inputs = sched.n_graph_inputs;
    {
        var i: c_int = 0;
        while (i < sched.n_splits) : (i += 1) {
            total_inputs += sched.splits[@intCast(i)].n_inputs;
        }
    }
    const graph_size = @max(graph.n_nodes, graph.n_leafs) + total_inputs * 2 * sched.n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched.prev_graph_size = sched.graph_size;
    sched.graph_size = graph_size;

    if (sched.graph.size < graph_size) {
        sched.graph.size = graph_size;
        const bytes = @as(usize, @intCast(graph_size)) * @sizeOf(?*mod.impl.c.ggml_tensor);
        sched.graph.nodes = @ptrCast(@alignCast(std.c.realloc(@ptrCast(sched.graph.nodes), bytes)));
        sched.graph.leafs = @ptrCast(@alignCast(std.c.realloc(@ptrCast(sched.graph.leafs), bytes)));
        mod.impl.assert(sched.graph.nodes != null, "sched->graph.nodes != NULL");
        mod.impl.assert(sched.graph.leafs != null, "sched->graph.leafs != NULL");
    }
    sched.graph.n_nodes = 0;
    sched.graph.n_leafs = 0;

    const graph_copy = &sched.graph;

    var n_dep_nodes_added: c_int = 0;

    {
        var i: c_int = 0;
        while (i < sched.n_splits) : (i += 1) {
            const split = &mod.impl.many(Split, sched.splits)[@intCast(i)];

            // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
            var j: c_int = 0;
            while (j < split.n_inputs) : (j += 1) {
                mod.impl.assert(graph_copy.size > graph_copy.n_nodes + 1, "graph_copy->size > (graph_copy->n_nodes + 1)");

                const input = split.inputs[@intCast(j)].?;
                const input_id = hashId(sched, input);
                const input_cpy = tensorIdCopyPtr(sched, input_id, split.backend_id, sched.cur_copy).*;

                // add a dependency to the input source so that it is not freed before the copy is done
                const input_dep = mod.impl.one(mod.impl.c.ggml_tensor, mod.impl.c.ggml_view_tensor(sched.ctx, input));
                input_dep.src[0] = input;
                sched.node_backend_ids[@intCast(graph_copy.n_nodes)] = sched.hv_tensor_backend_ids[input_id];
                graph_copy.nodes[@intCast(graph_copy.n_nodes)] = input_dep;
                graph_copy.n_nodes += 1;

                // add a dependency to the input copy so that it is allocated at the start of the split
                sched.node_backend_ids[@intCast(graph_copy.n_nodes)] = split.backend_id;
                graph_copy.nodes[@intCast(graph_copy.n_nodes)] = input_cpy;
                graph_copy.n_nodes += 1;
            }

            j = split.i_start;
            while (j < split.i_end) : (j += 1) {
                mod.impl.assert(graph_copy.size > graph_copy.n_nodes, "graph_copy->size > graph_copy->n_nodes");
                sched.node_backend_ids[@intCast(graph_copy.n_nodes)] = tensorBackendId(sched, graph.nodes[@intCast(j)].?);
                graph_copy.nodes[@intCast(graph_copy.n_nodes)] = graph.nodes[@intCast(j)];
                graph_copy.n_nodes += 1;

                if (alloc_deps.count() == 0) {
                    continue;
                }

                // add a dependency node so that the kept tensors are not freed before this node is computed
                if (alloc_deps.get(graph.nodes[@intCast(j)].?)) |keep| {
                    var k: usize = 0;
                    while (k < keep.items.len) : (k += max_src) {
                        const dep = mod.impl.one(mod.impl.c.ggml_tensor, mod.impl.c.ggml_view_tensor(sched.ctx, keep.items[k]));
                        var s: usize = 0;
                        while (s < max_src and k + s < keep.items.len) : (s += 1) {
                            dep.src[s] = keep.items[k + s];
                        }
                        mod.impl.assert(graph_copy.size > graph_copy.n_nodes, "graph_copy->size > graph_copy->n_nodes");
                        sched.node_backend_ids[@intCast(graph_copy.n_nodes)] = split.backend_id;
                        graph_copy.nodes[@intCast(graph_copy.n_nodes)] = dep;
                        graph_copy.n_nodes += 1;
                        n_dep_nodes_added += 1;
                    }
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    mod.impl.assert(n_dep_nodes_added == n_dep_nodes, "n_dep_nodes_added == n_dep_nodes");

    if (sched.n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        var i: c_int = 0;
        while (i < sched.n_graph_inputs) : (i += 1) {
            const input = sched.graph_inputs[@intCast(i)].?;
            const id = hashId(sched, input);
            const bid = tensorBackendId(sched, input);
            var cp: c_int = 0;
            while (cp < sched.n_copies) : (cp += 1) {
                const input_cpy = tensorIdCopyPtr(sched, id, bid, cp).*;
                sched.leaf_backend_ids[@intCast(graph_copy.n_leafs)] = bid;
                mod.impl.assert(graph_copy.size > graph_copy.n_leafs, "graph_copy->size > graph_copy->n_leafs");
                graph_copy.leafs[@intCast(graph_copy.n_leafs)] = input_cpy;
                graph_copy.n_leafs += 1;
            }
        }

        i = 0;
        while (i < sched.n_splits) : (i += 1) {
            const split = &mod.impl.many(Split, sched.splits)[@intCast(i)];
            const bid = split.backend_id;
            var j: c_int = 0;
            while (j < split.n_inputs) : (j += 1) {
                const input = split.inputs[@intCast(j)].?;
                const id = hashId(sched, input);
                var cp: c_int = 0;
                while (cp < sched.n_copies) : (cp += 1) {
                    const input_cpy = tensorIdCopyPtr(sched, id, bid, cp).*;
                    sched.leaf_backend_ids[@intCast(graph_copy.n_leafs)] = bid;
                    mod.impl.assert(graph_copy.size > graph_copy.n_leafs, "graph_copy->size > graph_copy->n_leafs");
                    graph_copy.leafs[@intCast(graph_copy.n_leafs)] = input_cpy;
                    graph_copy.n_leafs += 1;
                }
            }
        }
    }

    // add leafs from the original graph
    {
        var i: c_int = 0;
        while (i < graph.n_leafs) : (i += 1) {
            const leaf = graph.leafs[@intCast(i)].?;
            sched.leaf_backend_ids[@intCast(graph_copy.n_leafs)] = tensorBackendId(sched, leaf);
            mod.impl.assert(graph_copy.size > graph_copy.n_leafs, "graph_copy->size > graph_copy->n_leafs");
            graph_copy.leafs[@intCast(graph_copy.n_leafs)] = leaf;
            graph_copy.n_leafs += 1;
        }
    }

    // set ids for all splits
    {
        var i: c_int = 0;
        while (i < sched.n_splits) : (i += 1) {
            sched.splits[@intCast(i)].graph.uid = mod.runtime.ggml_graph_next_uid();
        }
    }
}

/// Ports `ggml_backend_sched_alloc_splits` (src/ggml-backend.cpp:2029 @b1b770d90).
///
/// Return: false when the reserve itself fails, or when the allocator cannot
/// fit the graph even after it. A change of backend ids alone does not force a realloc — only a
/// change that also changes the *buffer type* does.
fn allocSplits(sched: *Sched) bool {
    var backend_ids_changed = false;
    {
        var i: c_int = 0;
        while (i < sched.graph.n_nodes) : (i += 1) {
            const a = sched.node_backend_ids[@intCast(i)];
            const b = sched.prev_node_backend_ids[@intCast(i)];
            if (a != b and sched.bufts[@intCast(a)] != sched.bufts[@intCast(b)]) {
                backend_ids_changed = true;
                break;
            }
        }
    }
    if (!backend_ids_changed) {
        var i: c_int = 0;
        while (i < sched.graph.n_leafs) : (i += 1) {
            const a = sched.leaf_backend_ids[@intCast(i)];
            const b = sched.prev_leaf_backend_ids[@intCast(i)];
            if (a != b and sched.bufts[@intCast(a)] != sched.bufts[@intCast(b)]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed or !mod.impl.c.ggml_gallocr_alloc_graph(sched.galloc, @ptrCast(&sched.graph))) {
        if (!debug_off) {
            mod.impl.logDebug(
                "%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n",
                .{ "ggml_backend_sched_alloc_splits", @as(c_int, if (backend_ids_changed) 1 else 0) },
            );
        }

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        const graph_size_changed = sched.prev_graph_size != sched.graph_size;

        if (sched.callback_realloc) |cb| {
            cb(@ptrCast(sched), @ptrCast(&sched.graph), backend_ids_changed, graph_size_changed, sched.callback_realloc_user_data);
        }

        if (sched.debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const unexpected = !backend_ids_changed and !graph_size_changed;

            if (unexpected or sched.debug_realloc > 1) {
                mod.impl.abort("ggml_backend_sched_alloc_splits: unexpected graph reallocation");
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        var i: c_int = 0;
        while (i < sched.n_backends) : (i += 1) {
            mod.backend.ggml_backend_synchronize(sched.backends[@intCast(i)]);
        }

        if (!mod.impl.c.ggml_gallocr_reserve_n(sched.galloc, @ptrCast(&sched.graph), sched.node_backend_ids, sched.leaf_backend_ids)) {
            mod.impl.logError("%s: failed to reserve graph buffers\n", .{"ggml_backend_sched_alloc_splits"});
            return false;
        }
        if (!mod.impl.c.ggml_gallocr_alloc_graph(sched.galloc, @ptrCast(&sched.graph))) {
            mod.impl.logError("%s: failed to allocate graph\n", .{"ggml_backend_sched_alloc_splits"});
            return false;
        }
    }

    return true;
}

/// Ports `ggml_backend_sched_is_host_weight` (src/ggml-backend.cpp:2091 @b1b770d90).
///
/// Return: true for a weight that lives in a host buffer -- the inputs the
/// copy callback is offered, and the ones `computeSplits` copies last.
fn isHostWeight(t: *const mod.impl.c.ggml_tensor) bool {
    return t.buffer != null and
        mod.backend.ggml_backend_buffer_get_usage(t.buffer) == mod.impl.c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS and
        mod.backend.ggml_backend_buffer_is_host(t.buffer);
}

/// Ports `ggml_backend_sched_copy_input` (src/ggml-backend.cpp:2097 @b1b770d90).
///
/// Copies one input of `split` to the split's backend. A host weight is first
/// offered to the copy callback, which may copy part of it itself (libllama's
/// MoE cache copies only the experts a batch selects) and return true.
fn copyInput(sched: *Sched, split: *Split, input: *mod.impl.c.ggml_tensor) void {
    const split_backend_id = split.backend_id;
    const split_backend = sched.backends[@intCast(split_backend_id)];
    const input_backend = ggml_backend_sched_get_tensor_backend(@ptrCast(sched), input);
    const input_cpy = tensorCopy(sched, input, split_backend_id, sched.cur_copy).?;

    // read before the previous split, wait for the copy only
    if (split == &mod.impl.many(Split, sched.splits)[@intCast(sched.prefetch_split)]) {
        var i: c_int = 0;
        while (i < sched.n_prefetched) : (i += 1) {
            if (sched.prefetched[@intCast(i)] == input) {
                mod.backend.ggml_backend_event_synchronize(sched.prefetch_events[@intCast(sched.prefetch_backend_id)]);
                return;
            }
        }
    }

    if (input.flags & mod.impl.c.GGML_TENSOR_FLAG_INPUT != 0) {
        // inputs from the user must be copied immediately to prevent the user
        // overwriting the data before the copy is done
        if (sched.events[@intCast(split_backend_id)][@intCast(sched.cur_copy)] != null) {
            mod.backend.ggml_backend_event_synchronize(sched.events[@intCast(split_backend_id)][@intCast(sched.cur_copy)]);
        } else {
            mod.backend.ggml_backend_synchronize(split_backend);
        }
        mod.backend.ggml_backend_tensor_copy(input, input_cpy);
        return;
    }

    // wait for the split backend to finish using the input before overwriting it
    if (sched.events[@intCast(split_backend_id)][@intCast(sched.cur_copy)] != null) {
        mod.backend.ggml_backend_event_wait(split_backend, sched.events[@intCast(split_backend_id)][@intCast(sched.cur_copy)]);
    } else {
        mod.backend.ggml_backend_synchronize(split_backend);
    }

    if (sched.callback_copy) |cb| {
        if (isHostWeight(input) and
            cb(split_backend, input, input_cpy, @ptrCast(&split.graph), sched.callback_copy_user_data))
        {
            return;
        }
    }

    // try async copy, but if not possible, we can still use a sync copy
    // without synchronizing the dst backend, since we handle the
    // synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not
    // have direct access to the backend interface
    const async_ok = if (split_backend.*.iface.cpy_tensor_async) |f|
        f(input_backend, split_backend, input, input_cpy)
    else
        false;
    if (!async_ok) {
        mod.backend.ggml_backend_synchronize(input_backend);
        if (sched.events[@intCast(split_backend_id)][@intCast(sched.cur_copy)] != null) {
            mod.backend.ggml_backend_event_synchronize(sched.events[@intCast(split_backend_id)][@intCast(sched.cur_copy)]);
        } else {
            mod.backend.ggml_backend_synchronize(split_backend);
        }
        mod.backend.ggml_backend_tensor_copy(input, input_cpy);
    }
}

/// Ports `ggml_backend_sched_prefetch_inputs` (src/ggml-backend.cpp:2152 @b1b770d90)
/// (#31419, #32100).
///
/// The inputs of the next split that are computed on the backend of this
/// split before it, and copied to host memory, are copied before this split
/// computes, so that the next split waits for these copies instead of this
/// split -- e.g. a CPU split runs while a GPU split computes, if it reads only
/// the results of the earlier splits.
fn prefetchInputs(sched: *Sched, split_id: c_int) void {
    sched.n_prefetched = 0;
    if (split_id + 1 >= sched.n_splits) {
        return;
    }
    const splits = mod.impl.many(Split, sched.splits);
    const split = &splits[@intCast(split_id)];
    const next = &splits[@intCast(split_id + 1)];
    const backend_id = split.backend_id;
    const be = sched.backends[@intCast(backend_id)];
    if (be.*.iface.get_tensor_async == null) {
        return;
    }

    var i: c_int = 0;
    while (i < next.n_inputs and sched.n_prefetched < max_split_inputs) : (i += 1) {
        const input = next.inputs[@intCast(i)].?;
        if ((input.flags & mod.impl.c.GGML_TENSOR_FLAG_INPUT != 0) or tensorBackendId(sched, input) != backend_id) {
            continue;
        }
        const input_cpy = tensorCopy(sched, input, next.backend_id, sched.cur_copy).?;
        if (input_cpy.buffer == null or !mod.backend.ggml_backend_buffer_is_host(input_cpy.buffer)) {
            continue;
        }

        // only results of earlier splits, the copy callback of this split runs
        // later and can write other tensors (e.g. the MoE cache)
        const base: *const mod.impl.c.ggml_tensor = if (input.view_src != null) mod.impl.one(mod.impl.c.ggml_tensor, input.view_src) else input;
        if (base.buffer == null or mod.backend.ggml_backend_buffer_get_usage(base.buffer) != mod.impl.c.GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            continue;
        }

        // not if this split computes or writes the input
        var written = false;
        var j: c_int = 0;
        while (j < split.graph.n_nodes and !written) : (j += 1) {
            const node = mod.impl.one(mod.impl.c.ggml_tensor, split.graph.nodes[@intCast(j)]);
            written = !mod.backend.isViewOp(node.op) and (node == base or node.view_src == base);
        }
        if (written) {
            continue;
        }

        if (!sched.prefetch_tried[@intCast(backend_id)]) {
            sched.prefetch_tried[@intCast(backend_id)] = true;
            sched.prefetch_events[@intCast(backend_id)] = mod.backend.ggml_backend_event_new(be.*.device);
        }
        if (sched.prefetch_events[@intCast(backend_id)] == null) {
            return;
        }

        mod.backend.ggml_backend_tensor_get_async(be, input, input_cpy.data, 0, mod.impl.c.ggml_nbytes(input));
        sched.prefetched[@intCast(sched.n_prefetched)] = input;
        sched.n_prefetched += 1;
    }

    if (sched.n_prefetched > 0) {
        sched.prefetch_split = split_id + 1;
        sched.prefetch_backend_id = backend_id;
        mod.backend.ggml_backend_event_record(sched.prefetch_events[@intCast(backend_id)], be);
    }
}

/// Ports `ggml_backend_sched_compute_splits` (src/ggml-backend.cpp:2210 @b1b770d90).
///
/// Runs each split on its backend, copying inputs across first. Two details
/// carry the complexity:
///
/// - **Synchronisation is per split and per copy.** A split with no inputs
///   still waits on the previous split's backend, because the allocator may
///   have reused buffer regions across them.
/// - **Host weights are copied last**, so the copy callback can read the
///   split's other inputs (the MoE expert ids) already on the split backend.
///   The selective expert copy itself lives in libllama's MoE cache.
fn computeSplits(sched: *Sched) mod.impl.c.enum_ggml_status {
    const splits = mod.impl.many(Split, sched.splits);

    var prev_backend_id: c_int = -1;

    var split_id: c_int = 0;
    while (split_id < sched.n_splits) : (split_id += 1) {
        const split = &splits[@intCast(split_id)];
        const split_backend_id = split.backend_id;
        const split_backend = sched.backends[@intCast(split_backend_id)];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        if (split.n_inputs == 0 and prev_backend_id >= 0 and prev_backend_id != split_backend_id) {
            if (sched.events[@intCast(prev_backend_id)][@intCast(sched.cur_copy)] != null) {
                mod.backend.ggml_backend_event_synchronize(sched.events[@intCast(prev_backend_id)][@intCast(sched.cur_copy)]);
            } else {
                mod.backend.ggml_backend_synchronize(sched.backends[@intCast(prev_backend_id)]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback
        // can read the other inputs of the split
        var input_id: c_int = 0;
        while (input_id < split.n_inputs) : (input_id += 1) {
            const input = split.inputs[@intCast(input_id)].?;
            if (!isHostWeight(input) and mod.impl.many(bool, sched.hv_tensor_requires_copy)[hashId(sched, input)]) {
                copyInput(sched, split, input);
            }
        }
        // after the copies above, which may read host memory reused by the
        // copies of the next split before the weights, whose copy callback
        // may queue large uploads
        prefetchInputs(sched, split_id);
        input_id = 0;
        while (input_id < split.n_inputs) : (input_id += 1) {
            const input = split.inputs[@intCast(input_id)].?;
            if (isHostWeight(input) and mod.impl.many(bool, sched.hv_tensor_requires_copy)[hashId(sched, input)]) {
                copyInput(sched, split, input);
            }
        }

        if (sched.callback_eval == null) {
            const ec = mod.backend.ggml_backend_graph_compute_async(split_backend, &split.graph);
            if (ec != mod.impl.c.GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            var j0: c_int = 0;
            while (j0 < split.graph.n_nodes) : (j0 += 1) {
                var t = split.graph.nodes[@intCast(j0)];

                // check if the user needs data from this node
                var need = sched.callback_eval.?(t, true, sched.callback_eval_user_data);

                var j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need and j1 < split.graph.n_nodes - 1) {
                    j1 += 1;
                    t = split.graph.nodes[@intCast(j1)];
                    need = sched.callback_eval.?(t, true, sched.callback_eval_user_data);
                }

                var gv = mod.graph.ggml_graph_view(&split.graph, j0, j1 + 1);

                const ec = mod.backend.ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != mod.impl.c.GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if
                // they want to synchronize
                mod.backend.ggml_backend_synchronize(split_backend);

                if (need and !sched.callback_eval.?(t, false, sched.callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        if (sched.events[@intCast(split_backend_id)][@intCast(sched.cur_copy)] != null) {
            mod.backend.ggml_backend_event_record(sched.events[@intCast(split_backend_id)][@intCast(sched.cur_copy)], split_backend);
        }

        prev_backend_id = split_backend_id;
    }

    return mod.impl.c.GGML_STATUS_SUCCESS;
}

// -----------------------------------------------------------------------------
// Public API

/// Ports `ggml_backend_sched_new` (src/ggml-backend.cpp:2299 @b1b770d90).
///
/// Parameters:
/// - `backends`: in priority order, highest first. The **last must be a CPU
///   device** — asserted, because several passes treat `n_backends - 1` as the
///   fallback.
/// - `bufts`: one buffer type per backend, or null to take each backend's
///   default.
/// - `n_backends`: at most `GGML_SCHED_MAX_BACKENDS`.
/// - `graph_size`: sizes the hash table and the context buffer.
/// - `parallel`: enables `GGML_SCHED_MAX_COPIES` pipelined copies and the
///   per-backend events that order them.
/// - `op_offload`: lets a higher-priority backend claim an op whose weights
///   are on the CPU.
///
/// Return: the scheduler, released with `ggml_backend_sched_free`.
export fn ggml_backend_sched_new(
    backends: [*c]mod.impl.c.ggml_backend_t,
    bufts: [*c]mod.impl.c.ggml_backend_buffer_type_t,
    n_backends: c_int,
    graph_size: usize,
    parallel: bool,
    op_offload: bool,
) callconv(.c) mod.impl.c.ggml_backend_sched_t {
    mod.impl.assert(n_backends > 0, "n_backends > 0");
    mod.impl.assert(n_backends <= max_backends, "n_backends <= GGML_SCHED_MAX_BACKENDS");
    mod.impl.assert(
        mod.backend.ggml_backend_dev_type(mod.backend.ggml_backend_get_device(backends[@intCast(n_backends - 1)])) == mod.impl.c.GGML_BACKEND_DEVICE_TYPE_CPU,
        "the last backend must be a CPU device",
    );

    const sched: *Sched = @ptrCast(@alignCast(std.c.calloc(1, @sizeOf(Sched)).?));

    sched.debug = envInt("GGML_SCHED_DEBUG", 0);

    sched.debug_realloc = 0;
    // `GGML_SCHED_NO_REALLOC` is not defined by this build, so the `#ifdef`
    // arm that would set this to 1 is not compiled.
    sched.debug_realloc = envInt("GGML_SCHED_DEBUG_REALLOC", sched.debug_realloc);

    sched.n_backends = n_backends;
    sched.n_copies = if (parallel) max_copies else 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched.hash_set = mod.graph.ggml_hash_set_new(graph_size);
    sched.hv_tensor_backend_ids = @ptrCast(@alignCast(std.c.malloc(sched.hash_set.size * @sizeOf(c_int))));
    sched.hv_tensor_copies = @ptrCast(@alignCast(std.c.malloc(
        sched.hash_set.size * @as(usize, @intCast(sched.n_backends)) * @as(usize, @intCast(sched.n_copies)) * @sizeOf(?*mod.impl.c.ggml_tensor),
    )));
    sched.hv_tensor_requires_copy = @ptrCast(@alignCast(std.c.malloc(sched.hash_set.size * @sizeOf(bool))));

    const ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const nodes_size = graph_size + ggml_sched_max_splits * max_split_inputs * 2;
    sched.node_backend_ids = @ptrCast(@alignCast(std.c.calloc(nodes_size, @sizeOf(c_int))));
    sched.leaf_backend_ids = @ptrCast(@alignCast(std.c.calloc(nodes_size, @sizeOf(c_int))));
    sched.prev_node_backend_ids = @ptrCast(@alignCast(std.c.calloc(nodes_size, @sizeOf(c_int))));
    sched.prev_leaf_backend_ids = @ptrCast(@alignCast(std.c.calloc(nodes_size, @sizeOf(c_int))));

    sched.graph_size = 0;
    sched.prev_graph_size = 0;

    sched.context_buffer_size = ggml_sched_max_splits * max_split_inputs * 2 * @sizeOf(mod.impl.c.ggml_tensor) +
        mod.impl.c.ggml_graph_overhead_custom(graph_size, false);
    sched.context_buffer = @ptrCast(@alignCast(std.c.malloc(sched.context_buffer_size)));

    const initial_splits_capacity: c_int = 16;
    sched.splits = @ptrCast(@alignCast(std.c.calloc(@intCast(initial_splits_capacity), @sizeOf(Split))));
    sched.splits_capacity = initial_splits_capacity;

    sched.graph_inputs_capacity = max_split_inputs;
    sched.graph_inputs = @ptrCast(@alignCast(std.c.calloc(@intCast(sched.graph_inputs_capacity), @sizeOf(?*mod.impl.c.ggml_tensor))));

    var b: c_int = 0;
    while (b < n_backends) : (b += 1) {
        sched.backends[@intCast(b)] = backends[@intCast(b)];
        sched.bufts[@intCast(b)] = if (bufts != null)
            bufts[@intCast(b)]
        else
            mod.backend.ggml_backend_get_default_buffer_type(backends[@intCast(b)]);
        mod.impl.assert(
            mod.backend.ggml_backend_supports_buft(backends[@intCast(b)], sched.bufts[@intCast(b)]),
            "ggml_backend_supports_buft(backends[b], sched->bufts[b])",
        );

        if (sched.n_copies > 1) {
            var cp: c_int = 0;
            while (cp < sched.n_copies) : (cp += 1) {
                sched.events[@intCast(b)][@intCast(cp)] = mod.backend.ggml_backend_event_new(backends[@intCast(b)].*.device);
            }
        }
    }

    sched.galloc = mod.impl.c.ggml_gallocr_new_n(&sched.bufts, n_backends);
    sched.op_offload = op_offload;

    ggml_backend_sched_reset(@ptrCast(sched));

    return @ptrCast(sched);
}

/// Reads an integer from the environment, as the C's `getenv`/`atoi` pair does.
///
/// Return: the parsed value, or `fallback` when unset or unparseable —
/// `atoi` returns 0 on garbage, and so does this.
fn envInt(comptime name: [*:0]const u8, fallback: c_int) c_int {
    const raw = stdlib.env.getenv(name) orelse return fallback;
    return stdlib.num.atoiC(raw);
}

/// Ports `ggml_backend_sched_free` (src/ggml-backend.cpp:2372 @b1b770d90).
export fn ggml_backend_sched_free(sched_: mod.impl.c.ggml_backend_sched_t) callconv(.c) void {
    if (sched_ == null) {
        return;
    }
    const sched = schedOf(sched_);
    var b: c_int = 0;
    while (b < sched.n_backends) : (b += 1) {
        var cp: c_int = 0;
        while (cp < sched.n_copies) : (cp += 1) {
            mod.backend.ggml_backend_event_free(sched.events[@intCast(b)][@intCast(cp)]);
        }
        mod.backend.ggml_backend_event_free(sched.prefetch_events[@intCast(b)]);
    }
    mod.impl.c.ggml_gallocr_free(sched.galloc);
    mod.impl.c.ggml_free(sched.ctx);
    mod.graph.ggml_hash_set_free(&sched.hash_set);
    var i: c_int = 0;
    while (i < sched.splits_capacity) : (i += 1) {
        std.c.free(@ptrCast(sched.splits[@intCast(i)].inputs));
    }
    std.c.free(@ptrCast(sched.splits));
    std.c.free(@ptrCast(sched.graph_inputs));
    std.c.free(@ptrCast(sched.hv_tensor_backend_ids));
    std.c.free(@ptrCast(sched.hv_tensor_copies));
    std.c.free(@ptrCast(sched.hv_tensor_requires_copy));
    std.c.free(@ptrCast(sched.node_backend_ids));
    std.c.free(@ptrCast(sched.leaf_backend_ids));
    std.c.free(@ptrCast(sched.prev_node_backend_ids));
    std.c.free(@ptrCast(sched.prev_leaf_backend_ids));
    std.c.free(@ptrCast(sched.context_buffer));
    std.c.free(@ptrCast(sched.graph.nodes));
    std.c.free(@ptrCast(sched.graph.leafs));
    std.c.free(sched);
}

/// Ports `ggml_backend_sched_reset` (src/ggml-backend.cpp:2403 @b1b770d90).
///
/// Clears the per-run state. Idempotent: `is_reset` guards the expensive part
/// so a second call costs nothing.
export fn ggml_backend_sched_reset(sched_: mod.impl.c.ggml_backend_sched_t) callconv(.c) void {
    mod.impl.assert(sched_ != null, "sched");
    const sched = schedOf(sched_);
    // reset state for the next run
    if (!sched.is_reset) {
        mod.graph.ggml_hash_set_reset(&sched.hash_set);
        @memset(@as([*]u8, @ptrCast(sched.hv_tensor_backend_ids))[0 .. sched.hash_set.size * @sizeOf(c_int)], 0xFF);
        @memset(@as([*]u8, @ptrCast(sched.hv_tensor_copies))[0 .. sched.hash_set.size *
            @as(usize, @intCast(sched.n_backends)) * @as(usize, @intCast(sched.n_copies)) * @sizeOf(?*mod.impl.c.ggml_tensor)], 0);
        @memset(@as([*]u8, @ptrCast(sched.hv_tensor_requires_copy))[0 .. sched.hash_set.size * @sizeOf(bool)], 0);
        sched.is_reset = true;
    }
    sched.is_alloc = false;
}

/// Ports `ggml_backend_sched_reserve_size` (src/ggml-backend.cpp:2416 @b1b770d90).
export fn ggml_backend_sched_reserve_size(sched_: mod.impl.c.ggml_backend_sched_t, measure_graph: *mod.impl.CGraph, sizes: [*c]usize) callconv(.c) void {
    mod.impl.assert(sched_ != null, "sched");
    const sched = schedOf(sched_);
    mod.impl.assert(@as(c_int, @intCast(sched.hash_set.size)) >= measure_graph.n_nodes + measure_graph.n_leafs, "hash_set.size >= n_nodes + n_leafs");
    mod.impl.assert(sizes != null, "sizes");

    ggml_backend_sched_reset(sched_);

    ggml_backend_sched_synchronize(sched_);

    ggml_backend_sched_split_graph(sched_, measure_graph);

    mod.impl.c.ggml_gallocr_reserve_n_size(sched.galloc, @ptrCast(&sched.graph), sched.node_backend_ids, sched.leaf_backend_ids, sizes);
}

/// Ports `ggml_backend_sched_reserve` (src/ggml-backend.cpp:2430 @b1b770d90).
///
/// Return: false if the allocator could not reserve for `measure_graph`.
export fn ggml_backend_sched_reserve(sched_: mod.impl.c.ggml_backend_sched_t, measure_graph: *mod.impl.CGraph) callconv(.c) bool {
    mod.impl.assert(sched_ != null, "sched");
    const sched = schedOf(sched_);
    mod.impl.assert(@as(c_int, @intCast(sched.hash_set.size)) >= measure_graph.n_nodes + measure_graph.n_leafs, "hash_set.size >= n_nodes + n_leafs");

    ggml_backend_sched_synchronize(sched_);

    ggml_backend_sched_split_graph(sched_, measure_graph);

    if (!mod.impl.c.ggml_gallocr_reserve_n(sched.galloc, @ptrCast(&sched.graph), sched.node_backend_ids, sched.leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched_);

    return true;
}

/// Ports `ggml_backend_sched_alloc_graph` (src/ggml-backend.cpp:2447 @b1b770d90).
///
/// Advances `cur_copy` to the next pipeline slot, splits, and allocates.
export fn ggml_backend_sched_alloc_graph(sched_: mod.impl.c.ggml_backend_sched_t, graph: *mod.impl.CGraph) callconv(.c) bool {
    mod.impl.assert(sched_ != null, "sched");
    const sched = schedOf(sched_);
    mod.impl.assert(@as(c_int, @intCast(sched.hash_set.size)) >= graph.n_nodes + graph.n_leafs, "hash_set.size >= n_nodes + n_leafs");
    mod.impl.assert(!sched.is_alloc, "!sched->is_alloc");

    sched.cur_copy = sched.next_copy;
    sched.next_copy = @mod(sched.next_copy + 1, sched.n_copies);

    ggml_backend_sched_split_graph(sched_, graph);

    if (!allocSplits(sched)) {
        return false;
    }

    sched.is_alloc = true;

    return true;
}

/// Ports `ggml_backend_sched_graph_compute` (src/ggml-backend.cpp:2466 @b1b770d90).
export fn ggml_backend_sched_graph_compute(sched_: mod.impl.c.ggml_backend_sched_t, graph: *mod.impl.CGraph) callconv(.c) mod.impl.c.enum_ggml_status {
    const err = ggml_backend_sched_graph_compute_async(sched_, graph);
    ggml_backend_sched_synchronize(sched_);
    return err;
}

/// Ports `ggml_backend_sched_graph_compute_async` (src/ggml-backend.cpp:2472 @b1b770d90).
export fn ggml_backend_sched_graph_compute_async(sched_: mod.impl.c.ggml_backend_sched_t, graph: *mod.impl.CGraph) callconv(.c) mod.impl.c.enum_ggml_status {
    mod.impl.assert(sched_ != null, "sched");
    const sched = schedOf(sched_);
    if (!sched.is_reset and !sched.is_alloc) {
        ggml_backend_sched_reset(sched_);
    }

    if (!sched.is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched_, graph)) {
            return mod.impl.c.GGML_STATUS_ALLOC_FAILED;
        }
    }

    return computeSplits(sched);
}

/// Ports `ggml_backend_sched_synchronize` (src/ggml-backend.cpp:2487 @b1b770d90).
///
/// Resets `next_copy` to 0 when nothing is allocated, so generation keeps
/// using the same copy every step — a changing copy would change the graph
/// and disable CUDA graphs.
export fn ggml_backend_sched_synchronize(sched_: mod.impl.c.ggml_backend_sched_t) callconv(.c) void {
    mod.impl.assert(sched_ != null, "sched");
    const sched = schedOf(sched_);
    var i: c_int = 0;
    while (i < sched.n_backends) : (i += 1) {
        mod.backend.ggml_backend_synchronize(sched.backends[@intCast(i)]);
    }
    if (!sched.is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a
        // synchronization; this ensures that during generation the same copy
        // is used every time, which avoids changes in the graph that could
        // cause CUDA or other graphs to be disabled
        sched.next_copy = 0;
    }
}

/// Ports `ggml_backend_sched_set_eval_callback` (src/ggml-backend.cpp:2500 @b1b770d90).
export fn ggml_backend_sched_set_eval_callback(
    sched_: mod.impl.c.ggml_backend_sched_t,
    callback: mod.impl.c.ggml_backend_sched_eval_callback,
    user_data: ?*anyopaque,
) callconv(.c) void {
    mod.impl.assert(sched_ != null, "sched");
    const sched = schedOf(sched_);
    sched.callback_eval = callback;
    sched.callback_eval_user_data = user_data;
}

/// Ports `ggml_backend_sched_set_copy_callback` (src/ggml-backend.cpp:2506 @b1b770d90).
export fn ggml_backend_sched_set_copy_callback(
    sched_: mod.impl.c.ggml_backend_sched_t,
    callback: mod.impl.c.ggml_backend_sched_copy_callback,
    user_data: ?*anyopaque,
) callconv(.c) void {
    mod.impl.assert(sched_ != null, "sched");
    const sched = schedOf(sched_);
    sched.callback_copy = callback;
    sched.callback_copy_user_data = user_data;
}

/// Ports `ggml_backend_sched_set_realloc_callback` (src/ggml-backend.cpp:2512 @b1b770d90).
export fn ggml_backend_sched_set_realloc_callback(
    sched_: mod.impl.c.ggml_backend_sched_t,
    callback: mod.impl.c.ggml_backend_sched_realloc_callback,
    user_data: ?*anyopaque,
) callconv(.c) void {
    mod.impl.assert(sched_ != null, "sched");
    const sched = schedOf(sched_);
    sched.callback_realloc = callback;
    sched.callback_realloc_user_data = user_data;
}

/// Ports `ggml_backend_sched_get_n_splits` (src/ggml-backend.cpp:2518 @b1b770d90).
export fn ggml_backend_sched_get_n_splits(sched_: mod.impl.c.ggml_backend_sched_t) callconv(.c) c_int {
    mod.impl.assert(sched_ != null, "sched");
    return schedOf(sched_).n_splits;
}

/// Ports `ggml_backend_sched_get_n_copies` (src/ggml-backend.cpp:2523 @b1b770d90).
export fn ggml_backend_sched_get_n_copies(sched_: mod.impl.c.ggml_backend_sched_t) callconv(.c) c_int {
    mod.impl.assert(sched_ != null, "sched");
    return schedOf(sched_).n_copies;
}

/// Ports `ggml_backend_sched_get_n_backends` (src/ggml-backend.cpp:2528 @b1b770d90).
export fn ggml_backend_sched_get_n_backends(sched_: mod.impl.c.ggml_backend_sched_t) callconv(.c) c_int {
    mod.impl.assert(sched_ != null, "sched");
    return schedOf(sched_).n_backends;
}

/// Ports `ggml_backend_sched_get_backend` (src/ggml-backend.cpp:2533 @b1b770d90).
export fn ggml_backend_sched_get_backend(sched_: mod.impl.c.ggml_backend_sched_t, i: c_int) callconv(.c) mod.impl.c.ggml_backend_t {
    mod.impl.assert(sched_ != null, "sched");
    const sched = schedOf(sched_);
    mod.impl.assert(i >= 0 and i < sched.n_backends, "i >= 0 && i < sched->n_backends");
    return sched.backends[@intCast(i)];
}

/// Ports `ggml_backend_sched_get_buffer_type` (src/ggml-backend.cpp:2539 @b1b770d90).
export fn ggml_backend_sched_get_buffer_type(sched_: mod.impl.c.ggml_backend_sched_t, b: mod.impl.c.ggml_backend_t) callconv(.c) mod.impl.c.ggml_backend_buffer_type_t {
    mod.impl.assert(sched_ != null, "sched");
    const sched = schedOf(sched_);
    const backend_index = backendId(sched, b);
    mod.impl.assert(backend_index >= 0 and backend_index < sched.n_backends, "backend_index in range");

    return sched.bufts[@intCast(backend_index)];
}

/// Ports `ggml_backend_sched_get_buffer_size` (src/ggml-backend.cpp:2547 @b1b770d90).
export fn ggml_backend_sched_get_buffer_size(sched_: mod.impl.c.ggml_backend_sched_t, b: mod.impl.c.ggml_backend_t) callconv(.c) usize {
    mod.impl.assert(sched_ != null, "sched");
    const sched = schedOf(sched_);
    const backend_index = backendId(sched, b);
    mod.impl.assert(backend_index >= 0 and backend_index < sched.n_backends, "backend_index in range");

    return mod.impl.c.ggml_gallocr_get_buffer_size(sched.galloc, backend_index);
}

/// Ports `ggml_backend_sched_set_tensor_backend` (src/ggml-backend.cpp:2555 @b1b770d90).
///
/// A user assignment. Clears `is_reset` so the next split keeps it — pass 1
/// only fills in nodes still at -1.
export fn ggml_backend_sched_set_tensor_backend(sched_: mod.impl.c.ggml_backend_sched_t, node: *mod.impl.c.ggml_tensor, b: mod.impl.c.ggml_backend_t) callconv(.c) void {
    mod.impl.assert(sched_ != null, "sched");
    const sched = schedOf(sched_);
    const backend_index = backendId(sched, b);
    mod.impl.assert(backend_index >= 0 and backend_index < sched.n_backends, "backend_index in range");
    tensorBackendIdPtr(sched, node).* = backend_index;
    setCause(node, "usr");
    sched.is_reset = false;
}

/// Ports `ggml_backend_sched_get_tensor_backend` (src/ggml-backend.cpp:2564 @b1b770d90).
///
/// Return: the backend the node is assigned to, or null if it has none yet.
export fn ggml_backend_sched_get_tensor_backend(sched_: mod.impl.c.ggml_backend_sched_t, node: *mod.impl.c.ggml_tensor) callconv(.c) mod.impl.c.ggml_backend_t {
    mod.impl.assert(sched_ != null, "sched");
    const sched = schedOf(sched_);
    const backend_index = tensorBackendId(sched, node);
    if (backend_index == -1) {
        return null;
    }
    return sched.backends[@intCast(backend_index)];
}

// -----------------------------------------------------------------------------
// Imported from the sibling ported files
//
// `ggml-impl.h` declares the hash set but cannot be imported, so these are
// redeclared here as `context.zig` and `backend.zig` do.

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the split and scheduler structs match the C's layout" {
    // `ggml_backend_sched_t` is an opaque pointer across the ABI, and C++ that
    // has not been ported yet still allocates and walks these. A field added
    // or reordered here would corrupt silently rather than fail to link.
    try std.testing.expectEqual(@as(usize, 16), max_backends);
    try std.testing.expectEqual(@as(usize, 4), max_copies);
    try std.testing.expectEqual(@as(usize, 30), max_split_inputs);

    // The C's `graph` member is a by-value `ggml_cgraph`, so `Split` is larger
    // than its scalar fields suggest; catching a mismatch here is the point.
    try std.testing.expect(@sizeOf(Split) > @sizeOf(mod.impl.CGraph));
    try std.testing.expectEqual(@offsetOf(Split, "backend_id"), 0);
}

test "envInt falls back when unset and yields 0 on garbage, as atoi does" {
    // Neither name is set in a test run, so both take the fallback.
    try std.testing.expectEqual(@as(c_int, 0), envInt("GGML_SCHED_DEBUG_NOT_SET_XYZ", 0));
    try std.testing.expectEqual(@as(c_int, 7), envInt("GGML_SCHED_DEBUG_NOT_SET_XYZ", 7));
}

test "setCause and getCause are the compiled-out arm" {
    // The C's debug arm is behind `#if 0`; this is the `#else`. If the cause
    // strings are ever turned on, this test is what says the port did not
    // follow.
    setCause(null, "test");
    try std.testing.expectEqualStrings("", std.mem.span(getCause(null)));
}

// -----------------------------------------------------------------------------
// Why the assignment passes are gated elsewhere
//
// **The five assignment passes have no unit test here, deliberately.** They are
// gated by `make sched-diff`, which builds `harness/sched_dump.c` against the
// reference C and against this port and diffs the result.
//
// They need *some* gate, because no existing one could see them. Measured by
// injection: turning off pass 4's `view_src` propagation leaves
// `make parity-cli` at 6/6 and `make graph-diff` at 131/131. Parity has to miss
// it — at `--temp 0` the sampler takes an argmax, and `make backend-ops` has
// shown Metal and CPU agree on all 21,093 op configurations, so moving an op
// between backends shifts the last bits and not the chosen token. Token parity
// measures *what* was computed, never *where*.
//
// The gate lives in C rather than here because its expected values have to come
// from the reference implementation. A Zig test would have to hardcode them,
// and a hardcoded copy of the C's answer is not a check on the C — it is a
// check on whoever transcribed it. An earlier version of this file did exactly
// that, with a second copy of the stub-backend fixture; the two fixtures could
// drift apart with both still green, so it was removed.
