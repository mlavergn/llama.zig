//! Computation graphs: building them, copying them, and differentiating them.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/ggml/src/ggml.c`      — the graph section beginning at line
//!                                      6513, plus the hash-set lifecycle
//!                                      functions that sit with it
//! - `llama.cpp/ggml/src/ggml-impl.h` — the `static inline` graph helpers,
//!                                      which no caller can link against and
//!                                      which every reader of a graph needs
//!
//! both at v0.3.0 (`c1d0e7a00`). Each declaration names the C function it
//! replaces and the line it began at.
//!
//! # A graph is one allocation
//!
//! `ggml_new_graph_custom` does not allocate the nodes, the leafs, the hash
//! table, and the gradient arrays separately. It computes the total size, asks
//! the context for one block, and then hands out aligned slices of it. So
//! `ggml_graph_nbytes` and the allocation path have to walk the same fields in
//! the same order with the same alignments -- if they ever disagree, the graph
//! runs off the end of its own object. The C guards this with an `assert` at
//! the end of the allocation; that assert is reproduced here.
//!
//! # The hash set is the index for everything
//!
//! `visited_hash_set` is not just a visited-marker. A tensor's slot in it is
//! also the index of its gradient in `grads`, its accumulator in `grad_accs`,
//! and its use count in `use_counts`. That is why so much of this file threads
//! a `usize` position around rather than a pointer.
//!
//! A found index is only a *hit* when the used bit is also set -- `hashFind`
//! returns where a key would go as readily as where it is. Every read here
//! checks both, as the C does.

const std = @import("std");
const impl = @import("impl.zig");
const types = @import("types.zig");
const context = @import("context.zig");
const ops = @import("ops.zig");
const runtime = @import("runtime.zig");
const c = impl.c;

const Context = context.Context;
const Tensor = c.ggml_tensor;
const CGraph = impl.CGraph;
const HashSet = impl.HashSet;

extern fn fclose(stream: *std.c.FILE) c_int;
extern fn fprintf(stream: *std.c.FILE, fmt: [*:0]const u8, ...) c_int;
extern fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;

extern fn ggml_backend_buffer_get_usage(buffer: ?*anyopaque) c.enum_ggml_backend_buffer_usage;
extern fn ggml_backend_tensor_set(tensor: *Tensor, data: ?*const anyopaque, offset: usize, size: usize) void;

// -----------------------------------------------------------------------------
// The hash set
//
// Open addressing with linear probing. The probe helpers themselves live in
// `impl.zig`, because they are `static inline` in `ggml-impl.h` and callers
// outside this file need them; only the lifecycle is here.

/// Ports `ggml_hash_size` (ggml.c:6531 @c1d0e7a00).
///
/// Rounds up to the next prime past a power of two. Primes matter because the
/// hash is just the tensor's address shifted right by four -- allocations are
/// aligned, so the low bits carry no information and a power-of-two modulus
/// would collide heavily.
///
/// Parameters:
/// - `min_sz`: the smallest acceptable table size.
///
/// Return: the first prime in the table that is at least `min_sz`, or
/// `min_sz | 1` when `min_sz` exceeds every prime listed.
pub export fn ggml_hash_size(min_sz: usize) usize {
    // next primes after powers of two
    const primes = [_]usize{
        2,        3,        5,        11,        17,        37,        67,         131,
        257,      521,      1031,     2053,      4099,      8209,      16411,      32771,
        65537,    131101,   262147,   524309,    1048583,   2097169,   4194319,    8388617,
        16777259, 33554467, 67108879, 134217757, 268435459, 536870923, 1073741827, 2147483659,
    };

    // find the smallest prime that is larger or equal than min_sz
    var l: usize = 0;
    var r: usize = primes.len;
    while (l < r) {
        const m = (l + r) / 2;
        if (primes[m] < min_sz) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return if (l < primes.len) primes[l] else min_sz | 1;
}

/// Ports `ggml_hash_set_new` (ggml.c:6513 @c1d0e7a00).
///
/// Return: a set sized to the next prime at or above `size`, owning two heap
/// allocations. The caller frees it with `ggml_hash_set_free`.
pub export fn ggml_hash_set_new(size: usize) HashSet {
    const n = ggml_hash_size(size);
    return .{
        .size = n,
        .keys = @ptrCast(@alignCast(impl.ggmlMalloc(@sizeOf(*Tensor) * n))),
        .used = @ptrCast(@alignCast(impl.ggmlCalloc(impl.bitsetSize(n), @sizeOf(impl.Bitset)))),
    };
}

/// Ports `ggml_hash_set_reset` (ggml.c:6522 @c1d0e7a00).
///
/// Clears only the used bits. The keys are left as they are: a slot is
/// unreadable until its bit is set again, so there is nothing to scrub.
pub export fn ggml_hash_set_reset(hash_set: *HashSet) void {
    @memset(hash_set.used[0..impl.bitsetSize(hash_set.size)], 0);
}

/// Ports `ggml_hash_set_free` (ggml.c:6526 @c1d0e7a00).
pub export fn ggml_hash_set_free(hash_set: *HashSet) void {
    std.c.free(@ptrCast(hash_set.used));
    std.c.free(@ptrCast(hash_set.keys));
}

// -----------------------------------------------------------------------------
// Allocating a graph

/// Ports `incr_ptr_aligned` (ggml.c:7344 @c1d0e7a00).
///
/// Bump allocation over an address rather than a pointer, which lets the
/// sizing pass and the allocation pass share one routine: the C starts the
/// sizing pass from a null pointer and the allocation pass from the real
/// address, and both walk the identical sequence.
///
/// Parameters:
/// - `p`: running address, advanced past the block.
/// - `size`: bytes to reserve.
/// - `alignment`: power-of-two alignment for the block.
///
/// Return: the aligned address of the block.
fn incrPtrAligned(p: *usize, size: usize, alignment: usize) usize {
    const ptr = impl.pad(p.*, alignment);
    p.* = ptr + size;
    return ptr;
}

/// Ports `ggml_graph_nbytes` (ggml.c:7351 @c1d0e7a00).
///
/// Must walk the same fields in the same order as `ggml_new_graph_custom`.
fn graphNbytes(size: usize, grads: bool) usize {
    const hash_size = ggml_hash_size(size * 2);
    var p: usize = 0;
    _ = incrPtrAligned(&p, @sizeOf(CGraph), 1);
    _ = incrPtrAligned(&p, size * @sizeOf(*Tensor), @sizeOf(*Tensor)); // nodes
    _ = incrPtrAligned(&p, size * @sizeOf(*Tensor), @sizeOf(*Tensor)); // leafs
    _ = incrPtrAligned(&p, hash_size * @sizeOf(i32), @sizeOf(i32)); // use_counts
    _ = incrPtrAligned(&p, hash_size * @sizeOf(*Tensor), @sizeOf(*Tensor)); // hash keys
    if (grads) {
        _ = incrPtrAligned(&p, hash_size * @sizeOf(*Tensor), @sizeOf(*Tensor)); // grads
        _ = incrPtrAligned(&p, hash_size * @sizeOf(*Tensor), @sizeOf(*Tensor)); // grad_accs
    }
    _ = incrPtrAligned(&p, impl.bitsetSize(hash_size) * @sizeOf(impl.Bitset), @sizeOf(impl.Bitset));

    return p;
}

/// Ports `ggml_graph_overhead_custom` (ggml.c:7369 @c1d0e7a00).
pub export fn ggml_graph_overhead_custom(size: usize, grads: bool) usize {
    return types.object_size + impl.pad(graphNbytes(size, grads), c.GGML_MEM_ALIGN);
}

/// Ports `ggml_graph_overhead` (ggml.c:7373 @c1d0e7a00).
pub export fn ggml_graph_overhead() usize {
    return ggml_graph_overhead_custom(c.GGML_DEFAULT_GRAPH_SIZE, false);
}

/// Ports `ggml_new_graph_custom` (ggml.c:7377 @c1d0e7a00).
///
/// Parameters:
/// - `ctx`: context the graph is carved out of.
/// - `size`: maximum node count; `leafs` gets the same capacity.
/// - `grads`: when true, allocates the gradient and accumulator arrays. They
///   are indexed by hash slot, not by node, so they are `hash_size` long
///   rather than `size` long.
///
/// Return: the new graph, owned by `ctx` and invalid once `ctx` is freed.
pub export fn ggml_new_graph_custom(ctx: *Context, size: usize, grads: bool) *CGraph {
    const obj_size = graphNbytes(size, grads);
    const obj = context.newObject(ctx, c.GGML_OBJECT_TYPE_GRAPH, obj_size).?;
    const cgraph: *CGraph = @ptrCast(@alignCast(@as([*]u8, @ptrCast(ctx.mem_buffer.?)) + obj.offs));

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    const hash_size = ggml_hash_size(size * 2);

    var p: usize = @intFromPtr(cgraph) + @sizeOf(CGraph);

    const nodes_ptr = incrPtrAligned(&p, size * @sizeOf(*Tensor), @sizeOf(*Tensor));
    const leafs_ptr = incrPtrAligned(&p, size * @sizeOf(*Tensor), @sizeOf(*Tensor));
    const use_counts_ptr = incrPtrAligned(&p, hash_size * @sizeOf(i32), @sizeOf(i32));
    const hash_keys_ptr = incrPtrAligned(&p, hash_size * @sizeOf(*Tensor), @sizeOf(*Tensor));
    const grads_ptr = if (grads) incrPtrAligned(&p, hash_size * @sizeOf(*Tensor), @sizeOf(*Tensor)) else 0;
    const grad_accs_ptr = if (grads) incrPtrAligned(&p, hash_size * @sizeOf(*Tensor), @sizeOf(*Tensor)) else 0;

    const hash_used = incrPtrAligned(&p, impl.bitsetSize(hash_size) * @sizeOf(impl.Bitset), @sizeOf(impl.Bitset));

    // check that we allocated the correct amount of memory
    std.debug.assert(obj_size == p - @intFromPtr(cgraph));

    cgraph.* = .{
        .size = @intCast(size),
        .n_nodes = 0,
        .n_leafs = 0,
        .nodes = @ptrFromInt(nodes_ptr),
        .grads = @ptrFromInt(grads_ptr),
        .grad_accs = @ptrFromInt(grad_accs_ptr),
        .leafs = @ptrFromInt(leafs_ptr),
        .use_counts = @ptrFromInt(use_counts_ptr),
        .visited_hash_set = .{
            .size = hash_size,
            .used = @ptrFromInt(hash_used),
            .keys = @ptrFromInt(hash_keys_ptr),
        },
        .order = impl.eval_order_left_to_right,
        .uid = 0,
    };

    ggml_hash_set_reset(&cgraph.visited_hash_set);
    if (grads) {
        @memset(cgraph.grads[0..hash_size], null);
        @memset(cgraph.grad_accs[0..hash_size], null);
    }

    return cgraph;
}

/// Ports `ggml_new_graph` (ggml.c:7422 @c1d0e7a00).
pub export fn ggml_new_graph(ctx: *Context) *CGraph {
    return ggml_new_graph_custom(ctx, c.GGML_DEFAULT_GRAPH_SIZE, false);
}

/// Ports `ggml_graph_view` (ggml.c:7426 @c1d0e7a00).
///
/// A window onto `cgraph0`'s nodes, returned by value and owning nothing. Its
/// `size` is zero, which marks it as unable to accept new nodes; gradients are
/// dropped because they are indexed by the hash set, and a view that shared
/// the parent's set would alias its gradient array.
pub export fn ggml_graph_view(cgraph0: *CGraph, first: c_int, last: c_int) CGraph {
    return .{
        .size = 0,
        .n_nodes = last - first,
        .n_leafs = 0,
        .nodes = cgraph0.nodes + @as(usize, @intCast(first)),
        .grads = null, // gradients would need visited_hash_set
        .grad_accs = null,
        .leafs = null,
        .use_counts = cgraph0.use_counts,
        .visited_hash_set = cgraph0.visited_hash_set,
        .order = cgraph0.order,
        .uid = 0,
    };
}

/// Ports `ggml_graph_cpy` (ggml.c:7444 @c1d0e7a00).
///
/// Copies node and leaf pointers, then rebuilds `dst`'s hash set by inserting
/// every key `src` had in use. The slots will not line up between the two
/// sets -- they depend on the table size -- so use counts and gradients are
/// translated slot by slot rather than copied wholesale.
pub export fn ggml_graph_cpy(src: *CGraph, dst: *CGraph) void {
    impl.assert(dst.size >= src.n_leafs, "dst->size >= src->n_leafs");
    impl.assert(dst.size >= src.n_nodes, "dst->size >= src->n_nodes");
    impl.assert(dst.visited_hash_set.size >= src.visited_hash_set.size, "dst->visited_hash_set.size >= src->visited_hash_set.size");

    dst.n_leafs = src.n_leafs;
    dst.n_nodes = src.n_nodes;
    dst.order = src.order;

    for (0..@intCast(src.n_leafs)) |i| {
        dst.leafs[i] = src.leafs[i];
    }

    for (0..@intCast(src.n_nodes)) |i| {
        dst.nodes[i] = src.nodes[i];
    }

    for (0..src.visited_hash_set.size) |i| {
        // copy all hashset keys (tensors) that are in use
        if (impl.bitsetGet(src.visited_hash_set.used, i)) {
            const new_hash_pos = impl.hashInsert(&dst.visited_hash_set, src.visited_hash_set.keys[i].?);
            dst.use_counts[new_hash_pos] = src.use_counts[i];
        }
    }

    if (dst.grads != null) {
        @memset(dst.grads[0..dst.visited_hash_set.size], null);
        @memset(dst.grad_accs[0..dst.visited_hash_set.size], null);
    }
    if (src.grads != null) {
        impl.assert(dst.grads != null, "dst->grads != NULL");
        impl.assert(dst.grad_accs != null, "dst->grad_accs != NULL");
        for (0..@intCast(src.n_nodes)) |i| {
            const igrad_src = impl.hashFind(&src.visited_hash_set, src.nodes[i].?);
            const igrad_dst = impl.hashFind(&dst.visited_hash_set, dst.nodes[i].?);

            impl.assert(igrad_src != impl.hashset_full, "igrad_src != GGML_HASHSET_FULL");
            impl.assert(impl.bitsetGet(src.visited_hash_set.used, igrad_src), "ggml_bitset_get(src->visited_hash_set.used, igrad_src)");
            impl.assert(igrad_dst != impl.hashset_full, "igrad_dst != GGML_HASHSET_FULL");
            impl.assert(impl.bitsetGet(dst.visited_hash_set.used, igrad_dst), "ggml_bitset_get(dst->visited_hash_set.used, igrad_dst)");

            dst.grads[igrad_dst] = src.grads[igrad_src];
            dst.grad_accs[igrad_dst] = src.grad_accs[igrad_src];
        }
    }
}

/// Ports `ggml_graph_dup` (ggml.c:7491 @c1d0e7a00).
pub export fn ggml_graph_dup(ctx: *Context, cgraph: *CGraph, force_grads: bool) *CGraph {
    const result = ggml_new_graph_custom(ctx, @intCast(cgraph.size), cgraph.grads != null or force_grads);
    ggml_graph_cpy(cgraph, result);
    return result;
}

/// Ports `ggml_graph_reset` (ggml.c:7510 @c1d0e7a00).
///
/// Zeroes the gradient accumulators for another backward pass, except a loss
/// tensor's, which is seeded to one -- differentiating a loss with respect to
/// itself. AdamW momenta are cleared too, since they live in the node's own
/// sources rather than in the graph.
pub export fn ggml_graph_reset(cgraph: ?*CGraph) void {
    const g = cgraph orelse return;
    impl.assert(g.grads != null, "cgraph->grads != NULL");

    for (0..@intCast(g.n_nodes)) |i| {
        const node = g.nodes[i].?;
        const grad_acc = ggml_graph_get_grad_acc(g, node);

        if (node.op == c.GGML_OP_OPT_STEP_ADAMW) {
            // clear momenta
            _ = ops.ggml_set_zero(impl.one(Tensor, node.src[2]));
            _ = ops.ggml_set_zero(impl.one(Tensor, node.src[3]));
        }

        // initial gradients of loss should be 1, 0 otherwise
        if (grad_acc) |acc| {
            if ((node.flags & c.GGML_TENSOR_FLAG_LOSS) != 0) {
                impl.assert(acc.type == c.GGML_TYPE_F32, "grad_acc->type == GGML_TYPE_F32");
                impl.assert(types.ggml_is_scalar(acc), "ggml_is_scalar(grad_acc)");

                const onef: f32 = 1.0;
                if (acc.buffer != null) {
                    ggml_backend_tensor_set(acc, &onef, 0, @sizeOf(f32));
                } else {
                    impl.assert(acc.data != null, "grad_acc->data");
                    @as(*f32, @ptrCast(@alignCast(acc.data.?))).* = onef;
                }
            } else {
                _ = ops.ggml_set_zero(acc);
            }
        }
    }
}

/// Ports `ggml_graph_clear` (ggml.c:7546 @c1d0e7a00).
pub export fn ggml_graph_clear(cgraph: *CGraph) void {
    cgraph.n_leafs = 0;
    cgraph.n_nodes = 0;
    ggml_hash_set_reset(&cgraph.visited_hash_set);
}

// -----------------------------------------------------------------------------
// Reading a graph

/// Ports `ggml_graph_size` (ggml.c:7552 @c1d0e7a00).
pub export fn ggml_graph_size(cgraph: *CGraph) c_int {
    return cgraph.size;
}

/// Ports `ggml_graph_node` (ggml.c:7556 @c1d0e7a00).
///
/// A negative `i` counts back from the end, so `-1` is the last node.
pub export fn ggml_graph_node(cgraph: *CGraph, i: c_int) *Tensor {
    if (i < 0) {
        impl.assert(cgraph.n_nodes + i >= 0, "cgraph->n_nodes + i >= 0");
        return cgraph.nodes[@intCast(cgraph.n_nodes + i)].?;
    }

    impl.assert(i < cgraph.n_nodes, "i < cgraph->n_nodes");
    return cgraph.nodes[@intCast(i)].?;
}

/// Ports `ggml_graph_nodes` (ggml.c:7566 @c1d0e7a00).
pub export fn ggml_graph_nodes(cgraph: *CGraph) [*c]?*Tensor {
    return cgraph.nodes;
}

/// Ports `ggml_graph_n_nodes` (ggml.c:7570 @c1d0e7a00).
pub export fn ggml_graph_n_nodes(cgraph: *CGraph) c_int {
    return cgraph.n_nodes;
}

/// Ports `ggml_graph_add_node` (ggml.c:7574 @c1d0e7a00).
///
/// Appends without visiting parents, so the caller is asserting the sources
/// are already in the graph.
pub export fn ggml_graph_add_node(cgraph: *CGraph, tensor: *Tensor) void {
    impl.assert(cgraph.size > cgraph.n_nodes, "cgraph->size > cgraph->n_nodes");
    cgraph.nodes[@intCast(cgraph.n_nodes)] = tensor;
    cgraph.n_nodes += 1;
}

/// Ports `ggml_graph_get_tensor` (ggml.c:7580 @c1d0e7a00).
///
/// Linear scan of leafs then nodes.
///
/// Return: the first tensor whose name matches, or null. Borrowed from the
/// graph.
pub export fn ggml_graph_get_tensor(cgraph: *const CGraph, name: [*:0]const u8) ?*Tensor {
    for (0..@intCast(cgraph.n_leafs)) |i| {
        const leaf = cgraph.leafs[i].?;
        if (std.mem.orderZ(u8, @ptrCast(&leaf.name), name) == .eq) return leaf;
    }

    for (0..@intCast(cgraph.n_nodes)) |i| {
        const node = cgraph.nodes[i].?;
        if (std.mem.orderZ(u8, @ptrCast(&node.name), name) == .eq) return node;
    }

    return null;
}

/// Ports `ggml_graph_get_grad` (ggml.c:7600 @c1d0e7a00).
///
/// Three conditions, all required: the key was found, its slot is in use, and
/// the graph has gradients at all.
pub export fn ggml_graph_get_grad(cgraph: *const CGraph, node: *const Tensor) ?*Tensor {
    const igrad = impl.hashFind(&cgraph.visited_hash_set, node);
    return if (igrad != impl.hashset_full and
        impl.bitsetGet(cgraph.visited_hash_set.used, igrad) and
        cgraph.grads != null) cgraph.grads[igrad] else null;
}

/// Ports `ggml_graph_get_grad_acc` (ggml.c:7605 @c1d0e7a00).
pub export fn ggml_graph_get_grad_acc(cgraph: *const CGraph, node: *const Tensor) ?*Tensor {
    const igrad = impl.hashFind(&cgraph.visited_hash_set, node);
    return if (igrad != impl.hashset_full and
        impl.bitsetGet(cgraph.visited_hash_set.used, igrad) and
        cgraph.grad_accs != null) cgraph.grad_accs[igrad] else null;
}

// -----------------------------------------------------------------------------
// Building the forward graph
//
// A depth-first walk from the output back through `src[]`, recording each
// tensor once. Tensors with no op become leafs; everything else becomes a
// node, and the topological order falls out of the post-order traversal.

/// Ports `ggml_visit_parents_graph` (ggml.c:7136 @c1d0e7a00).
///
/// Parameters:
/// - `cgraph`: graph being built.
/// - `node`: tensor to visit.
/// - `compute`: whether this subtree is actually evaluated. A node reached
///   only as a shape reference is recorded but not marked, which is what
///   `ggml_build_forward_order` exists to do.
///
/// Return: `node`'s slot in the visited hash set.
fn visitParents(cgraph: *CGraph, node: *Tensor, compute: bool) usize {
    if (node.op != c.GGML_OP_NONE and compute) {
        node.flags |= c.GGML_TENSOR_FLAG_COMPUTE;
    }

    const node_hash_pos = impl.hashFind(&cgraph.visited_hash_set, node);
    impl.assert(node_hash_pos != impl.hashset_full, "node_hash_pos != GGML_HASHSET_FULL");

    if (impl.bitsetGet(cgraph.visited_hash_set.used, node_hash_pos)) {
        // already visited

        if (compute) {
            // A subtree first reached without `compute` may be reached again
            // with it, and the flag has to spread even though the nodes are
            // already recorded.
            for (0..c.GGML_MAX_SRC) |i| {
                if (node.src[i] != null) {
                    const src = impl.one(Tensor, node.src[i]);
                    if ((src.flags & c.GGML_TENSOR_FLAG_COMPUTE) == 0) {
                        _ = visitParents(cgraph, src, true);
                    }
                }
            }
        }

        return node_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph.visited_hash_set.keys[node_hash_pos] = node;
    impl.bitsetSet(cgraph.visited_hash_set.used, node_hash_pos);
    cgraph.use_counts[node_hash_pos] = 0;

    for (0..c.GGML_MAX_SRC) |i| {
        const k = switch (cgraph.order) {
            impl.eval_order_left_to_right => i,
            impl.eval_order_right_to_left => c.GGML_MAX_SRC - 1 - i,
            else => i, // unknown order, just fall back to using i
        };

        if (node.src[k] != null) {
            const src_hash_pos = visitParents(cgraph, impl.one(Tensor, node.src[k]), compute);

            // Update the use count for this operand.
            cgraph.use_counts[src_hash_pos] += 1;
        }
    }

    if (node.op == c.GGML_OP_NONE and (node.flags & c.GGML_TENSOR_FLAG_PARAM) == 0) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        impl.assert(cgraph.n_leafs < cgraph.size, "cgraph->n_leafs < cgraph->size");

        if (node.name[0] == 0) {
            _ = context.ggml_format_name(node, "leaf_%d", cgraph.n_leafs);
        }

        cgraph.leafs[@intCast(cgraph.n_leafs)] = node;
        cgraph.n_leafs += 1;
    } else {
        impl.assert(cgraph.n_nodes < cgraph.size, "cgraph->n_nodes < cgraph->size");

        if (node.name[0] == 0) {
            _ = context.ggml_format_name(node, "node_%d", cgraph.n_nodes);
        }

        cgraph.nodes[@intCast(cgraph.n_nodes)] = node;
        cgraph.n_nodes += 1;
    }

    return node_hash_pos;
}

/// Ports `ggml_build_forward_impl` (ggml.c:7204 @c1d0e7a00).
fn buildForwardImpl(cgraph: *CGraph, tensor: *Tensor, expand: bool, compute: bool) void {
    if (!expand) {
        ggml_graph_clear(cgraph);
    }

    const n_old = cgraph.n_nodes;

    _ = visitParents(cgraph, tensor, compute);

    const n_new = cgraph.n_nodes - n_old;
    impl.printDebug("%s: visited %d new nodes\n", .{ "ggml_build_forward_impl", n_new });

    if (n_new > 0) {
        // the last added node should always be starting point
        impl.assert(cgraph.nodes[@intCast(cgraph.n_nodes - 1)] == tensor, "cgraph->nodes[cgraph->n_nodes - 1] == tensor");
    }
}

/// Ports `ggml_build_forward_select` (ggml.c:7223 @c1d0e7a00).
///
/// Records every tensor in `tensors` but marks only `tensors[idx]` for
/// computation, so alternative branches share the graph's node numbering
/// without being evaluated.
///
/// Return: `tensors[idx]`, borrowed from the caller's array.
pub export fn ggml_build_forward_select(
    cgraph: *CGraph,
    tensors: [*c]?*Tensor,
    n_tensors: c_int,
    idx: c_int,
) *Tensor {
    impl.assert(idx >= 0 and idx < n_tensors, "idx >= 0 && idx < n_tensors");

    for (0..@intCast(n_tensors)) |i| {
        buildForwardImpl(cgraph, tensors[i].?, true, i == @as(usize, @intCast(idx)));
    }

    return tensors[@intCast(idx)].?;
}

/// Ports `ggml_build_forward_expand` (ggml.c:7237 @c1d0e7a00).
///
/// The normal entry point: add `tensor` and everything it depends on.
pub export fn ggml_build_forward_expand(cgraph: *CGraph, tensor: *Tensor) void {
    buildForwardImpl(cgraph, tensor, true, true);
}

/// Ports `ggml_build_forward_order` (ggml.c:7241 @c1d0e7a00).
///
/// Records the subtree without marking it for computation, fixing its place in
/// the evaluation order for a later `ggml_build_forward_expand`.
pub export fn ggml_build_forward_order(cgraph: *CGraph, tensor: *Tensor) void {
    buildForwardImpl(cgraph, tensor, true, false);
}

// -----------------------------------------------------------------------------
// Differentiating a graph
//
// One reverse pass over the nodes. For each, `computeBackward` adds the nodes
// that produce its sources' gradients, appending them to the same graph.
//
// Gradients are addressed by hash slot, not by node index, so the four
// `*OrSet` helpers below all take an `isrc` and all end by expanding the graph
// to include whatever they just built.

/// Ports `ggml_add_or_set` (ggml.c:6582 @c1d0e7a00).
fn addOrSet(ctx: *Context, cgraph: *CGraph, isrc: usize, tensor: *Tensor) void {
    const src = cgraph.visited_hash_set.keys[isrc].?;
    if (cgraph.grads[isrc]) |g| {
        cgraph.grads[isrc] = ops.addImpl(ctx, g, tensor, cgraph.grad_accs[isrc] != null);
    } else {
        cgraph.grads[isrc] = tensor;
    }
    _ = context.ggml_format_name(cgraph.grads[isrc].?, "grad for %s", &src.name);
    ggml_build_forward_expand(cgraph, cgraph.grads[isrc].?);
}

/// Ports `ggml_acc_or_set` (ggml.c:6598 @c1d0e7a00).
fn accOrSet(
    ctx: *Context,
    cgraph: *CGraph,
    isrc: usize,
    tensor: *Tensor,
    nb1: usize,
    nb2: usize,
    nb3: usize,
    offset: usize,
) void {
    const src = cgraph.visited_hash_set.keys[isrc].?;
    if (cgraph.grads[isrc]) |g| {
        cgraph.grads[isrc] = ops.accImpl(ctx, g, tensor, nb1, nb2, nb3, offset, cgraph.grad_accs[isrc] != null);
    } else {
        // FIXME this is going to produce NaN if a contains inf/NaN -- the
        // comment is the C's, and the behaviour is kept.
        const a_zero = ops.ggml_scale(ctx, src, 0.0);
        cgraph.grads[isrc] = ops.accImpl(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
    }
    _ = context.ggml_format_name(cgraph.grads[isrc].?, "grad for %s", &cgraph.visited_hash_set.keys[isrc].?.name);
    ggml_build_forward_expand(cgraph, cgraph.grads[isrc].?);
}

/// Ports `ggml_add1_or_set` (ggml.c:6619 @c1d0e7a00).
fn add1OrSet(ctx: *Context, cgraph: *CGraph, isrc: usize, tensor: *Tensor) void {
    const src = cgraph.visited_hash_set.keys[isrc].?;
    if (cgraph.grads[isrc]) |g| {
        cgraph.grads[isrc] = ops.add1Impl(ctx, g, tensor, cgraph.grad_accs[isrc] != null);
    } else {
        cgraph.grads[isrc] = ops.ggml_repeat(ctx, tensor, src);
    }
    _ = context.ggml_format_name(cgraph.grads[isrc].?, "grad for %s", &src.name);
    ggml_build_forward_expand(cgraph, cgraph.grads[isrc].?);
}

/// Ports `ggml_sub_or_set` (ggml.c:6635 @c1d0e7a00).
fn subOrSet(ctx: *Context, cgraph: *CGraph, isrc: usize, tensor: *Tensor) void {
    const src = cgraph.visited_hash_set.keys[isrc].?;
    if (cgraph.grads[isrc]) |g| {
        cgraph.grads[isrc] = ops.subImpl(ctx, g, tensor, cgraph.grad_accs[isrc] != null);
    } else {
        cgraph.grads[isrc] = ops.unary.neg(ctx, tensor);
    }
    _ = context.ggml_format_name(cgraph.grads[isrc].?, "grad for %s", &src.name);
    ggml_build_forward_expand(cgraph, cgraph.grads[isrc].?);
}

/// Ports `ggml_compute_backward` (ggml.c:6651 @c1d0e7a00).
///
/// Adds the nodes computing the gradients of `cgraph.nodes[i]`'s sources.
/// One `switch` over every differentiable op -- ggml's entire autodiff.
///
/// Parameters:
/// - `ctx`: context the new nodes are allocated from.
/// - `cgraph`: graph being differentiated, appended to in place.
/// - `i`: index of the node to differentiate.
/// - `grads_needed`: per-hash-slot flags marking which tensors want gradients.
///
/// Return: nothing. Aborts on an op with no backward rule.
fn computeBackward(ctx: *Context, cgraph: *CGraph, i: c_int, grads_needed: [*]const bool) void {
    const tensor = cgraph.nodes[@intCast(i)].?;
    const grad = ggml_graph_get_grad(cgraph, tensor) orelse return;

    const src0: ?*Tensor = if (tensor.src[0] != null) impl.one(Tensor, tensor.src[0]) else null;
    const src1: ?*Tensor = if (tensor.src[1] != null) impl.one(Tensor, tensor.src[1]) else null;
    const src2: ?*Tensor = if (tensor.src[2] != null) impl.one(Tensor, tensor.src[2]) else null;
    const hash_set = &cgraph.visited_hash_set;
    const isrc0 = if (src0) |s| impl.hashFind(hash_set, s) else impl.hashset_full;
    const isrc1 = if (src1) |s| impl.hashFind(hash_set, s) else impl.hashset_full;
    const isrc2 = if (src2) |s| impl.hashFind(hash_set, s) else impl.hashset_full;
    const src0_needs_grads = src0 != null and isrc0 != impl.hashset_full and impl.bitsetGet(hash_set.used, isrc0) and grads_needed[isrc0];
    const src1_needs_grads = src1 != null and isrc1 != impl.hashset_full and impl.bitsetGet(hash_set.used, isrc1) and grads_needed[isrc1];
    const src2_needs_grads = src2 != null and isrc2 != impl.hashset_full and impl.bitsetGet(hash_set.used, isrc2) and grads_needed[isrc2];

    switch (tensor.op) {
        c.GGML_OP_DUP => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, grad);
        },
        c.GGML_OP_ADD => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, grad);
            if (src1_needs_grads) {
                var tmp = grad;
                if (!types.ggml_are_same_shape(src0.?, src1.?)) {
                    tmp = ops.ggml_repeat_back(ctx, tmp, src1.?);
                }
                addOrSet(ctx, cgraph, isrc1, tmp);
            }
        },
        c.GGML_OP_ADD1 => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, grad);
            // TODO: should probably be sum instead of mean -- the C's comment.
            if (src1_needs_grads) addOrSet(ctx, cgraph, isrc1, ops.ggml_mean(ctx, grad));
        },
        c.GGML_OP_ACC => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, grad);
            if (src1_needs_grads) {
                const nb1: usize = @intCast(impl.getOpParamsI32(tensor, 0));
                const nb2: usize = @intCast(impl.getOpParamsI32(tensor, 1));
                const nb3: usize = @intCast(impl.getOpParamsI32(tensor, 2));
                const offset: usize = @intCast(impl.getOpParamsI32(tensor, 3));

                const tensor_grad_view = ops.ggml_view_4d(
                    ctx,
                    grad,
                    src1.?.ne[0],
                    src1.?.ne[1],
                    src1.?.ne[2],
                    src1.?.ne[3],
                    nb1,
                    nb2,
                    nb3,
                    offset,
                );

                addOrSet(ctx, cgraph, isrc1, ops.ggml_reshape(ctx, ops.ggml_cont(ctx, tensor_grad_view), src1.?));
            }
        },
        c.GGML_OP_SUB => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, grad);
            if (src1_needs_grads) subOrSet(ctx, cgraph, isrc1, grad);
        },
        c.GGML_OP_MUL => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_mul(ctx, grad, src1.?));
            if (src1_needs_grads) {
                var tmp = ops.ggml_mul(ctx, src0.?, grad);
                if (!types.ggml_are_same_shape(src0.?, src1.?)) {
                    tmp = ops.ggml_repeat_back(ctx, tmp, src1.?);
                }
                addOrSet(ctx, cgraph, isrc1, tmp);
            }
        },
        c.GGML_OP_DIV => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_div(ctx, grad, src1.?));
            if (src1_needs_grads) subOrSet(ctx, cgraph, isrc1, ops.ggml_mul(ctx, grad, ops.ggml_div(ctx, tensor, src1.?)));
        },
        c.GGML_OP_SQR => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_scale(ctx, ops.ggml_mul(ctx, src0.?, grad), 2.0));
        },
        c.GGML_OP_SQRT => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_scale(ctx, ops.ggml_div(ctx, grad, tensor), 0.5));
        },
        c.GGML_OP_LOG => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_div(ctx, grad, src0.?));
        },
        c.GGML_OP_SIN => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_mul(ctx, grad, ops.ggml_cos(ctx, src0.?)));
        },
        c.GGML_OP_COS => {
            if (src0_needs_grads) subOrSet(ctx, cgraph, isrc0, ops.ggml_mul(ctx, grad, ops.ggml_sin(ctx, src0.?)));
        },
        c.GGML_OP_SUM => {
            if (src0_needs_grads) add1OrSet(ctx, cgraph, isrc0, grad);
        },
        c.GGML_OP_SUM_ROWS => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_repeat(ctx, grad, src0.?));
        },
        c.GGML_OP_MEAN => {
            if (src0_needs_grads) {
                add1OrSet(ctx, cgraph, isrc0, ops.scaleImpl(ctx, grad, 1.0 / @as(f32, @floatFromInt(src0.?.ne[0])), 0.0, false));
            }
        },
        c.GGML_OP_REPEAT => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_repeat_back(ctx, grad, src0.?));
        },
        c.GGML_OP_REPEAT_BACK => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_repeat(ctx, grad, src0.?));
        },
        c.GGML_OP_RMS_NORM => {
            if (src0_needs_grads) {
                const eps = impl.getOpParamsF32(tensor, 0);
                addOrSet(ctx, cgraph, isrc0, ops.ggml_rms_norm_back(ctx, grad, src0.?, eps));
            }
        },
        c.GGML_OP_MUL_MAT => {
            // https://cs231n.github.io/optimization-2/#staged
            //   tensor.shape [m,p,qq,rr]
            //   src0.shape   [n,m,q1,r1]
            //   src1.shape   [n,p,qq,rr]
            if (src0_needs_grads) {
                impl.assert(grad.ne[2] == src1.?.ne[2], "grad->ne[2] == src1->ne[2]");
                impl.assert(grad.ne[3] == src1.?.ne[3], "grad->ne[3] == src1->ne[3]");
                var tmp = ops.ggml_out_prod(ctx, src1.?, grad); // [n,m,qq,rr]
                if (!types.ggml_are_same_shape(tmp, src0.?)) {
                    impl.assert(tmp.ne[0] == src0.?.ne[0], "tmp->ne[0] == src0->ne[0]");
                    impl.assert(tmp.ne[1] == src0.?.ne[1], "tmp->ne[1] == src0->ne[1]");
                    impl.assert(tmp.ne[3] == 1, "tmp->ne[3] == 1");

                    const nr2 = @divTrunc(tmp.ne[2], src0.?.ne[2]);
                    const nb2 = tmp.nb[2] * @as(usize, @intCast(nr2));
                    const nb3 = tmp.nb[2];

                    tmp = ops.ggml_view_4d(ctx, tmp, src0.?.ne[0], src0.?.ne[1], src0.?.ne[2], nr2, tmp.nb[1], nb2, nb3, 0);
                    tmp = ops.ggml_repeat_back(ctx, tmp, src0.?);
                }
                addOrSet(ctx, cgraph, isrc0, tmp);
            }
            if (src1_needs_grads) {
                // When src0 is bigger than the gradient -- the usual case in
                // llama -- transposing the gradient and taking an outer
                // product beats transposing src0.
                addOrSet(ctx, cgraph, isrc1, ops.ggml_out_prod(ctx, src0.?, ops.ggml_transpose(ctx, grad)));
            }
        },
        c.GGML_OP_SCALE => {
            if (src0_needs_grads) {
                const s = impl.getOpParamsF32(tensor, 0);
                addOrSet(ctx, cgraph, isrc0, ops.scaleImpl(ctx, grad, s, 0.0, false));
            }
        },
        c.GGML_OP_SET => {
            const nb1: usize = @intCast(impl.getOpParamsI32(tensor, 0));
            const nb2: usize = @intCast(impl.getOpParamsI32(tensor, 1));
            const nb3: usize = @intCast(impl.getOpParamsI32(tensor, 2));
            const offset: usize = @intCast(impl.getOpParamsI32(tensor, 3));

            var tensor_grad_view: ?*Tensor = null;

            if (src0_needs_grads or src1_needs_grads) {
                impl.assert(src0.?.type == tensor.type, "src0->type == tensor->type");
                impl.assert(cgraph.grads[isrc0] == null or cgraph.grads[isrc0].?.type == grad.type, "!cgraph->grads[isrc0] || cgraph->grads[isrc0]->type == grad->type");
                impl.assert(cgraph.grads[isrc1] == null or !src1_needs_grads or cgraph.grads[isrc1].?.type == grad.type, "!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type");

                tensor_grad_view = ops.ggml_view_4d(
                    ctx,
                    grad,
                    src1.?.ne[0],
                    src1.?.ne[1],
                    src1.?.ne[2],
                    src1.?.ne[3],
                    nb1,
                    nb2,
                    nb3,
                    offset,
                );
            }

            if (src0_needs_grads) {
                const tmp = ops.unary.neg(ctx, tensor_grad_view.?);
                addOrSet(ctx, cgraph, isrc0, ops.accImpl(ctx, grad, tmp, nb1, nb2, nb3, offset, false));
            }

            if (src1_needs_grads) {
                addOrSet(ctx, cgraph, isrc1, ops.ggml_reshape(ctx, ops.ggml_cont(ctx, tensor_grad_view.?), src1.?));
            }
        },
        c.GGML_OP_CPY => {
            // cpy overwrites the value of src1 with src0 and returns view(src1),
            // which is mathematically tensor = src0*1 + src1*0. So src0's
            // gradient passes through and src1's is zero -- a no-op.
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_reshape(ctx, grad, src0.?));
        },
        c.GGML_OP_CONT => {
            // same as cpy
            if (src0_needs_grads) {
                impl.assert(cgraph.grads[isrc0] == null or types.ggml_is_contiguous(cgraph.grads[isrc0].?), "!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0])");
                impl.assert(types.ggml_is_contiguous(grad), "ggml_is_contiguous(grad)");
                impl.assert(types.ggml_nelements(tensor) == types.ggml_nelements(src0.?), "ggml_nelements(tensor) == ggml_nelements(src0)");
                addOrSet(ctx, cgraph, isrc0, if (types.ggml_are_same_shape(tensor, src0.?)) grad else ops.ggml_reshape(ctx, grad, src0.?));
            }
        },
        c.GGML_OP_RESHAPE => {
            if (src0_needs_grads) {
                const grad_cont = if (types.ggml_is_contiguous(grad)) grad else ops.ggml_cont(ctx, grad);
                addOrSet(ctx, cgraph, isrc0, ops.ggml_reshape(ctx, grad_cont, src0.?));
            }
        },
        c.GGML_OP_VIEW => {
            if (src0_needs_grads) {
                // A view's offset is a `size_t` written over the first two
                // i32 slots, not an i32 -- reading it as one would truncate
                // any view past 2 GiB.
                var offset: usize = undefined;
                @memcpy(std.mem.asBytes(&offset), std.mem.asBytes(&tensor.op_params)[0..@sizeOf(usize)]);

                var nb1 = tensor.nb[1];
                var nb2 = tensor.nb[2];
                var nb3 = tensor.nb[3];

                if (cgraph.grads[isrc0]) |g| {
                    if (src0.?.type != g.type) {
                        // gradient is typically F32, but src0 could be other type
                        const ng = types.ggml_element_size(g);
                        const n0 = types.ggml_element_size(src0.?);
                        impl.assert(offset % n0 == 0, "offset % n0 == 0");
                        impl.assert(nb1 % n0 == 0, "nb1 % n0 == 0");
                        impl.assert(nb2 % n0 == 0, "nb2 % n0 == 0");
                        impl.assert(nb3 % n0 == 0, "nb3 % n0 == 0");
                        offset = (offset / n0) * ng;
                        nb1 = (nb1 / n0) * ng;
                        nb2 = (nb2 / n0) * ng;
                        nb3 = (nb3 / n0) * ng;
                    }
                }

                accOrSet(ctx, cgraph, isrc0, grad, nb1, nb2, nb3, offset);
            }
        },
        c.GGML_OP_PERMUTE => {
            if (src0_needs_grads) {
                // Invert the permutation: whichever slot each axis landed in
                // going forward is where it comes back from.
                const axis0: usize = @intCast(impl.getOpParamsI32(tensor, 0) & 0x3);
                const axis1: usize = @intCast(impl.getOpParamsI32(tensor, 1) & 0x3);
                const axis2: usize = @intCast(impl.getOpParamsI32(tensor, 2) & 0x3);
                const axis3: usize = @intCast(impl.getOpParamsI32(tensor, 3) & 0x3);
                var axb = [_]c_int{ 0, 0, 0, 0 }; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                addOrSet(ctx, cgraph, isrc0, ops.ggml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        },
        c.GGML_OP_TRANSPOSE => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_transpose(ctx, grad));
        },
        c.GGML_OP_GET_ROWS => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_get_rows_back(ctx, grad, src1.?, src0.?));
            // src1 holds row indices, which are not differentiable: no-op.
        },
        c.GGML_OP_DIAG_MASK_INF => {
            if (src0_needs_grads) {
                // ggml_diag_mask_inf_impl() shouldn't be here
                // ref: https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992
                const n_past = impl.getOpParamsI32(tensor, 0);
                addOrSet(ctx, cgraph, isrc0, ops.diagMaskZero(ctx, grad, n_past));
            }
        },
        c.GGML_OP_DIAG_MASK_ZERO => {
            if (src0_needs_grads) {
                const n_past = impl.getOpParamsI32(tensor, 0);
                addOrSet(ctx, cgraph, isrc0, ops.diagMaskZero(ctx, grad, n_past));
            }
        },
        c.GGML_OP_SOFT_MAX => {
            if (src0_needs_grads) {
                const scale = impl.getOpParamsF32(tensor, 0);
                const max_bias = impl.getOpParamsF32(tensor, 1);
                addOrSet(ctx, cgraph, isrc0, ops.ggml_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
            }
            impl.assert(src1 == null or !src1_needs_grads, "backward pass for softmax mask not implemented");
        },
        c.GGML_OP_ROPE => {
            if (src0_needs_grads) {
                const n_dims = impl.getOpParamsI32(tensor, 1);
                const mode = impl.getOpParamsI32(tensor, 2);
                const n_ctx_orig = impl.getOpParamsI32(tensor, 4);
                const freq_base = impl.getOpParamsF32(tensor, 5);
                const freq_scale = impl.getOpParamsF32(tensor, 6);
                const ext_factor = impl.getOpParamsF32(tensor, 7);
                const attn_factor = impl.getOpParamsF32(tensor, 8);
                const beta_fast = impl.getOpParamsF32(tensor, 9);
                const beta_slow = impl.getOpParamsF32(tensor, 10);
                var sections = [_]c_int{ 0, 0, 0, 0 };
                for (0..4) |k| sections[k] = impl.getOpParamsI32(tensor, 11 + k);

                // Multi-section RoPE is distinguished by the position tensor's
                // shape, not by a flag: one position per section means mrope.
                const rope_back = if (grad.ne[2] == src1.?.ne[0])
                    ops.ggml_rope_ext_back(ctx, grad, src1.?, src2, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
                else
                    ops.ggml_rope_multi_back(ctx, grad, src1.?, src2, n_dims, &sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                addOrSet(ctx, cgraph, isrc0, rope_back);
            }
            impl.assert(src2 == null or !src2_needs_grads, "gradients for freq factors not implemented");
        },
        c.GGML_OP_IM2COL => {
            if (src1_needs_grads) {
                const s0 = impl.getOpParamsI32(tensor, 0);
                const s1 = impl.getOpParamsI32(tensor, 1);
                const p0 = impl.getOpParamsI32(tensor, 2);
                const p1 = impl.getOpParamsI32(tensor, 3);
                const d0 = impl.getOpParamsI32(tensor, 4);
                const d1 = impl.getOpParamsI32(tensor, 5);
                const is_2d = impl.getOpParamsI32(tensor, 6) == 1;

                addOrSet(ctx, cgraph, isrc1, ops.ggml_im2col_back(ctx, grad, src0.?, &src1.?.ne, s0, s1, p0, p1, d0, d1, is_2d));
            }
        },
        c.GGML_OP_POOL_2D => {
            if (src0_needs_grads) {
                const op: c.enum_ggml_op_pool = @intCast(impl.getOpParamsI32(tensor, 0));
                const k0 = impl.getOpParamsI32(tensor, 1);
                const k1 = impl.getOpParamsI32(tensor, 2);
                const s0 = impl.getOpParamsI32(tensor, 3);
                const s1 = impl.getOpParamsI32(tensor, 4);
                // The padding is a float in the signature but was stored as
                // an i32 by `ggml_pool_2d`, so it round-trips through an
                // integer and comes back truncated. The C declares these
                // `int32_t` and lets the call implicitly widen them; the
                // conversion is explicit here but the value is the same.
                const p0: f32 = @floatFromInt(impl.getOpParamsI32(tensor, 5));
                const p1: f32 = @floatFromInt(impl.getOpParamsI32(tensor, 6));

                addOrSet(ctx, cgraph, isrc0, ops.ggml_pool_2d_back(ctx, grad, src0.?, op, k0, k1, s0, s1, p0, p1));
            }
        },
        // WIN_PART and WIN_UNPART fall through to UNARY in the C, which then
        // reads a unary op out of op_params they never wrote. Neither is
        // differentiable and neither is reachable from a training graph; the
        // fallthrough is reproduced rather than corrected.
        c.GGML_OP_WIN_PART, c.GGML_OP_WIN_UNPART, c.GGML_OP_UNARY => {
            switch (types.ggml_get_unary_op(tensor)) {
                c.GGML_UNARY_OP_ABS => {
                    if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_mul(ctx, ops.unary.sgn(ctx, src0.?), grad));
                },
                c.GGML_UNARY_OP_SGN => {}, // piecewise constant: no-op
                c.GGML_UNARY_OP_NEG => {
                    if (src0_needs_grads) subOrSet(ctx, cgraph, isrc0, grad);
                },
                c.GGML_UNARY_OP_STEP => {}, // piecewise constant: no-op
                c.GGML_UNARY_OP_RELU => {
                    if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_mul(ctx, ops.unary.step(ctx, src0.?), grad));
                },
                c.GGML_UNARY_OP_SILU => {
                    if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_silu_back(ctx, grad, src0.?));
                },
                c.GGML_UNARY_OP_EXP => {
                    if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_mul(ctx, tensor, grad));
                },
                c.GGML_UNARY_OP_EXPM1 => {
                    if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_mul(ctx, grad, ops.unary.exp(ctx, src0.?)));
                },
                c.GGML_UNARY_OP_SOFTPLUS => {
                    if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_mul(ctx, grad, ops.unary.sigmoid(ctx, src0.?)));
                },
                else => {
                    impl.logError("%s: unsupported unary op for backward pass: %s\n", .{
                        "ggml_compute_backward",
                        types.ggml_unary_op_name(types.ggml_get_unary_op(tensor)),
                    });
                    impl.abort("fatal error");
                },
            }
        },
        c.GGML_OP_CROSS_ENTROPY_LOSS => {
            if (src0_needs_grads) addOrSet(ctx, cgraph, isrc0, ops.ggml_cross_entropy_loss_back(ctx, grad, src0.?, src1.?));
            impl.assert(!src1_needs_grads, "backward pass for labels not implemented");
        },
        c.GGML_OP_GLU => {
            switch (types.ggml_get_glu_op(tensor)) {
                c.GGML_GLU_OP_SWIGLU => {
                    if (src0_needs_grads) {
                        impl.assert(src1 != null, "backward pass only implemented for split swiglu");
                        addOrSet(ctx, cgraph, isrc0, ops.ggml_silu_back(ctx, ops.ggml_mul(ctx, grad, src1.?), src0.?));
                    }
                    if (src1_needs_grads) addOrSet(ctx, cgraph, isrc1, ops.ggml_mul(ctx, ops.unary.silu(ctx, src0.?), grad));
                },
                else => impl.abort("unsupported glu op for backward pass"),
            }
        },
        c.GGML_OP_NONE => {}, // no-op
        else => {
            impl.logError("%s: unsupported ggml op for backward pass: %s\n", .{
                "ggml_compute_backward",
                types.ggml_op_name(tensor.op),
            });
            impl.abort("fatal error");
        },
    }

    impl.assert(!src0_needs_grads or types.ggml_are_same_shape(src0.?, cgraph.grads[isrc0].?), "ggml_are_same_shape(src0, cgraph->grads[isrc0])");
    impl.assert(!src1_needs_grads or types.ggml_are_same_shape(src1.?, cgraph.grads[isrc1].?), "ggml_are_same_shape(src1, cgraph->grads[isrc1])");
    impl.assert(!src2_needs_grads or types.ggml_are_same_shape(src2.?, cgraph.grads[isrc2].?), "ggml_are_same_shape(src2, cgraph->grads[isrc2])");
}

/// Ports `ggml_build_backward_expand` (ggml.c:7245 @c1d0e7a00).
///
/// Appends the backward pass to a graph that already holds the forward one.
///
/// Runs in two sweeps. The first goes forward, marking which tensors need
/// gradients -- a node needs one if it is a parameter, is the loss, or has a
/// source that needs one -- and attaching accumulators. The second goes
/// backward calling `computeBackward`, which is only correct because the first
/// sweep has already settled `grads_needed` for every node.
///
/// Parameters:
/// - `ctx`: context the gradient nodes are allocated from.
/// - `cgraph`: graph holding the forward pass, extended in place.
/// - `grad_accs`: optional per-node accumulators; null lets ggml allocate them
///   for loss tensors only.
pub export fn ggml_build_backward_expand(ctx: *Context, cgraph: *CGraph, grad_accs: [*c]?*Tensor) void {
    impl.assert(cgraph.n_nodes > 0, "cgraph->n_nodes > 0");
    impl.assert(cgraph.grads != null, "cgraph->grads");
    impl.assert(cgraph.grad_accs != null, "cgraph->grad_accs");

    const n_nodes_f = cgraph.n_nodes;

    @memset(cgraph.grads[0..cgraph.visited_hash_set.size], null);
    @memset(cgraph.grad_accs[0..cgraph.visited_hash_set.size], null);
    const grads_needed: [*]bool = @ptrCast(@alignCast(std.c.calloc(cgraph.visited_hash_set.size, @sizeOf(bool)).?));
    defer std.c.free(grads_needed);

    {
        var any_params = false;
        var any_loss = false;
        for (0..@intCast(n_nodes_f)) |i| {
            const node = cgraph.nodes[i].?;
            any_params = any_params or (node.flags & c.GGML_TENSOR_FLAG_PARAM) != 0;
            any_loss = any_loss or (node.flags & c.GGML_TENSOR_FLAG_LOSS) != 0;
        }
        impl.assert(any_params, "no trainable parameters found, did you forget to call ggml_set_param?");
        impl.assert(any_loss, "no training loss found, did you forget to call ggml_set_loss?");
    }

    for (0..@intCast(n_nodes_f)) |i| {
        const node = cgraph.nodes[i].?;

        if (node.type == c.GGML_TYPE_I32) continue;

        var node_needs_grad = (node.flags & c.GGML_TENSOR_FLAG_PARAM) != 0 or (node.flags & c.GGML_TENSOR_FLAG_LOSS) != 0;
        var ignore_src = [_]bool{false} ** c.GGML_MAX_SRC;
        switch (node.op) {
            // gradients in node->src[0] for one reason or another have no
            // effect on output gradients
            c.GGML_OP_IM2COL, c.GGML_OP_IM2COL_BACK => ignore_src[0] = true, // only used for its shape
            c.GGML_OP_UNARY => {
                // SGN and STEP unary ops are piecewise constant
                const uop = types.ggml_get_unary_op(node);
                if (uop == c.GGML_UNARY_OP_SGN or uop == c.GGML_UNARY_OP_STEP) ignore_src[0] = true;
            },
            // gradients in node->src[1] for one reason or another have no
            // effect on output gradients
            c.GGML_OP_CPY, // gradients in CPY target are irrelevant
            c.GGML_OP_GET_ROWS, // row indices not differentiable
            c.GGML_OP_GET_ROWS_BACK, // same as for GET_ROWS
            c.GGML_OP_ROPE, // positions not differentiable
            => ignore_src[1] = true,
            else => {},
        }
        for (0..c.GGML_MAX_SRC) |j| {
            if (node.src[j] == null) continue;
            const src = impl.one(Tensor, node.src[j]);
            if (ignore_src[j] or !grads_needed[impl.hashFind(&cgraph.visited_hash_set, src)]) continue;
            impl.assert(src.type == c.GGML_TYPE_F32 or src.type == c.GGML_TYPE_F16, "node->src[j]->type == GGML_TYPE_F32 || node->src[j]->type == GGML_TYPE_F16");
            node_needs_grad = true;
            break;
        }
        if (!node_needs_grad) continue;

        // inplace operations are currently not supported
        impl.assert(node.view_src == null or node.op == c.GGML_OP_CPY or node.op == c.GGML_OP_VIEW or
            node.op == c.GGML_OP_RESHAPE or node.op == c.GGML_OP_PERMUTE or node.op == c.GGML_OP_TRANSPOSE, "!node->view_src || ...");

        const ihash = impl.hashFind(&cgraph.visited_hash_set, node);
        impl.assert(ihash != impl.hashset_full, "ihash != GGML_HASHSET_FULL");
        impl.assert(impl.bitsetGet(cgraph.visited_hash_set.used, ihash), "ggml_bitset_get(cgraph->visited_hash_set.used, ihash)");
        if (grad_accs != null and grad_accs[i] != null) {
            cgraph.grad_accs[ihash] = grad_accs[i];
            cgraph.grads[ihash] = cgraph.grad_accs[ihash];
        } else if ((node.flags & c.GGML_TENSOR_FLAG_LOSS) != 0) {
            // loss tensors always need a gradient accumulator
            cgraph.grad_accs[ihash] = context.ggml_new_tensor(ctx, c.GGML_TYPE_F32, c.GGML_MAX_DIMS, &node.ne);
            cgraph.grads[ihash] = cgraph.grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    // Reverse order: a node's gradient is complete only once every consumer
    // downstream of it has contributed.
    var i: c_int = n_nodes_f - 1;
    while (i >= 0) : (i -= 1) {
        computeBackward(ctx, cgraph, i, grads_needed);
    }
}

// -----------------------------------------------------------------------------
// Fusion
//
// A backend asks whether a run of nodes can be executed as one kernel. The
// answer is no if any intermediate result is needed outside the run, which is
// what the use-count and view-source checks below establish.

/// Ports `ggml_node_list_find_tensor` (ggml.c:7638 @c1d0e7a00).
///
/// Return: the position of `tensor` within `idxs`, or -1. A node index past
/// the end of the graph also yields -1, which is how a caller's stale index
/// is rejected rather than trusted.
fn nodeListFindTensor(cgraph: *const CGraph, idxs: [*c]const c_int, count: c_int, tensor: *const Tensor) c_int {
    impl.assert(idxs != null, "cgraph && idxs");
    for (0..@intCast(count)) |i| {
        const node_idx = idxs[i];

        if (node_idx >= cgraph.n_nodes) return -1;
        if (cgraph.nodes[@intCast(node_idx)] == tensor) return @intCast(i);
    }
    return -1;
}

/// Ports `ggml_is_constant` (ggml.c:7656 @c1d0e7a00).
///
/// A weight: it lives in a buffer marked as weights and is not a trainable
/// parameter, so it cannot change during this graph's execution.
fn isConstant(tensor: *const Tensor) bool {
    return tensor.buffer != null and
        ggml_backend_buffer_get_usage(tensor.buffer) == c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS and
        (tensor.flags & c.GGML_TENSOR_FLAG_PARAM) == 0;
}

/// Ports `ggml_can_fuse_subgraph_ext` (ggml.c:7660 @c1d0e7a00).
///
/// Parameters:
/// - `cgraph`: the graph.
/// - `node_idxs`: `count` node indices, in order, forming the candidate run.
/// - `ops_wanted`: the op each of those nodes must have.
/// - `outputs`: which of the run's nodes may be read from outside it.
/// - `num_outputs`: length of `outputs`.
///
/// Return: true when the run can be replaced by a single fused kernel.
pub export fn ggml_can_fuse_subgraph_ext(
    cgraph: *const CGraph,
    node_idxs: [*c]const c_int,
    count: c_int,
    ops_wanted: [*c]const c.enum_ggml_op,
    outputs: [*c]const c_int,
    num_outputs: c_int,
) bool {
    impl.assert(outputs != null and num_outputs > 0, "outputs && num_outputs > 0");

    for (0..@intCast(count)) |i| {
        if (node_idxs[i] >= cgraph.n_nodes) return false;

        const node = cgraph.nodes[@intCast(node_idxs[i])].?;

        if (node.op != ops_wanted[i]) return false;

        if ((node.flags & c.GGML_TENSOR_FLAG_COMPUTE) == 0) return false;

        // A declared output may be read from anywhere, so the checks below
        // do not apply to it.
        if (nodeListFindTensor(cgraph, outputs, num_outputs, node) != -1) continue;

        if ((node.flags & c.GGML_TENSOR_FLAG_OUTPUT) != 0) return false;

        // Every use of this node must be inside the run, or fusing it would
        // discard a value someone else still reads.
        var subgraph_uses: i32 = 0;
        for (i + 1..@intCast(count)) |j| {
            const other_node = cgraph.nodes[@intCast(node_idxs[j])].?;
            for (0..c.GGML_MAX_SRC) |src_idx| {
                if (other_node.src[src_idx] == node) subgraph_uses += 1;
            }
        }

        if (subgraph_uses != nodeGetUseCount(cgraph, node_idxs[i])) return false;

        // If node is a view, check that view_src and all its parent view_srcs
        // are within the subgraph. External view sources are allowed only for
        // weight tensors, which are constant for this graph execution.
        var view_src = node.view_src;
        while (view_src != null) {
            const vs = impl.one(Tensor, view_src);
            if (nodeListFindTensor(cgraph, node_idxs, count, vs) == -1 and !isConstant(vs)) return false;
            view_src = vs.view_src;
        }
    }

    return true;
}

/// Ports `ggml_node_get_use_count` (ggml-impl.h:629 @c1d0e7a00).
///
/// `static inline` in the header, so it has to be reproduced rather than
/// called. Lives here rather than in `impl.zig` because it reads a graph.
pub fn nodeGetUseCount(cgraph: *const CGraph, node_idx: c_int) i32 {
    const node = cgraph.nodes[@intCast(node_idx)].?;

    const hash_pos = impl.hashFind(&cgraph.visited_hash_set, node);
    if (!impl.bitsetGet(cgraph.visited_hash_set.used, hash_pos)) return 0;
    return cgraph.use_counts[hash_pos];
}

/// Ports `ggml_node_has_n_uses` (ggml-impl.h:641 @c1d0e7a00).
///
/// Parameters:
/// - `cgraph`: the graph.
/// - `node_idx`: index of the node to test.
/// - `n_uses`: the use count being replaced by a fused kernel.
///
/// Return: true when fusing this node away would discard nothing.
pub fn nodeHasNUses(cgraph: *const CGraph, node_idx: c_int, n_uses: i32) bool {
    const node = cgraph.nodes[@intCast(node_idx)].?;

    if (nodeGetUseCount(cgraph, node_idx) != n_uses) return false;

    // A view means some other node might still read the value through the
    // view source.
    if (node.view_src != null) return false;

    // The caller asked for this node's output, so it has to survive.
    if ((node.flags & c.GGML_TENSOR_FLAG_OUTPUT) != 0) return false;

    return true;
}

/// Ports `ggml_can_fuse_ext` (ggml-impl.h:669 @c1d0e7a00).
///
/// Parameters:
/// - `cgraph`: the graph.
/// - `node_idxs`: `num_ops` node indices, in the order they would fuse.
/// - `ops_wanted`: the op each of those nodes must have.
/// - `num_ops`: length of both arrays.
///
/// Return: true when the run can be replaced by a single fused kernel.
pub fn canFuseExt(
    cgraph: *const CGraph,
    node_idxs: []const c_int,
    ops_wanted: []const c.enum_ggml_op,
    num_ops: usize,
) bool {
    for (0..num_ops) |i| {
        if (node_idxs[i] >= cgraph.n_nodes) return false;

        const node = cgraph.nodes[@intCast(node_idxs[i])].?;
        if (node.op != ops_wanted[i]) return false;
        if ((node.flags & c.GGML_TENSOR_FLAG_COMPUTE) == 0) return false;

        // Every node but the last must feed only the next one.
        if (i < num_ops - 1 and !nodeHasNUses(cgraph, node_idxs[i], 1)) return false;

        if (i > 0) {
            const prev = cgraph.nodes[@intCast(node_idxs[i - 1])].?;
            if (node.src[0] != prev and node.src[1] != prev) return false;
            if (!types.ggml_are_same_shape(node, prev)) return false;
        }
    }
    return true;
}

/// Ports `ggml_can_fuse` (ggml-impl.h:699 @c1d0e7a00).
///
/// The same test as `canFuseExt` for a run of consecutive nodes.
///
/// Parameters:
/// - `cgraph`: the graph.
/// - `node_idx`: index of the first node in the run.
/// - `ops_wanted`: the op each node in the run must have; at most 32.
///
/// Return: true when the run can be replaced by a single fused kernel.
pub fn canFuse(cgraph: *const CGraph, node_idx: c_int, ops_wanted: []const c.enum_ggml_op) bool {
    std.debug.assert(ops_wanted.len < 32);

    if (node_idx + @as(c_int, @intCast(ops_wanted.len)) > cgraph.n_nodes) return false;

    var idxs: [32]c_int = undefined;
    for (0..ops_wanted.len) |i| idxs[i] = node_idx + @as(c_int, @intCast(i));

    return canFuseExt(cgraph, &idxs, ops_wanted, ops_wanted.len);
}

// -----------------------------------------------------------------------------
// Printing

/// Ports `ggml_graph_print` (ggml.c:7610 @c1d0e7a00).
pub export fn ggml_graph_print(cgraph: *const CGraph) void {
    impl.logInfo("=== GRAPH ===\n", .{});

    impl.logInfo("n_nodes = %d\n", .{cgraph.n_nodes});
    for (0..@intCast(cgraph.n_nodes)) |i| {
        const node = cgraph.nodes[i].?;

        impl.logInfo(" - %3d: [ %5lld, %5lld, %5lld] %16s %s\n", .{
            @as(c_int, @intCast(i)),
            node.ne[0],
            node.ne[1],
            node.ne[2],
            types.ggml_op_name(node.op),
            if ((node.flags & c.GGML_TENSOR_FLAG_PARAM) != 0)
                @as([*:0]const u8, "x")
            else if (ggml_graph_get_grad(cgraph, node) != null) "g" else " ",
        });
    }

    impl.logInfo("n_leafs = %d\n", .{cgraph.n_leafs});
    for (0..@intCast(cgraph.n_leafs)) |i| {
        const node = cgraph.leafs[i].?;

        impl.logInfo(" - %3d: [ %5lld, %5lld] %8s %16s\n", .{
            @as(c_int, @intCast(i)),
            node.ne[0],
            node.ne[1],
            types.ggml_op_name(node.op),
            context.ggml_get_name(node),
        });
    }

    impl.logInfo("========================================\n", .{});
}

/// Ports `ggml_graph_find` (ggml.c:7720 @c1d0e7a00).
///
/// A null graph returns true, which is what makes the colour choice in
/// `ggml_graph_dump_dot` treat "no forward graph supplied" as "everything is
/// in it".
fn graphFind(cgraph: ?*const CGraph, node: *const Tensor) bool {
    const g = cgraph orelse return true;

    for (0..@intCast(g.n_nodes)) |i| {
        if (g.nodes[i] == node) return true;
    }

    return false;
}

/// Ports `ggml_graph_get_parent` (ggml.c:7734 @c1d0e7a00).
fn graphGetParent(cgraph: *const CGraph, node: *const Tensor) ?*Tensor {
    for (0..@intCast(cgraph.n_nodes)) |i| {
        const parent = cgraph.nodes[i].?;
        if (ggml_graph_get_grad(cgraph, parent) == node) return parent;
    }

    return null;
}

/// Ports `ggml_graph_dump_dot_node_edge` (ggml.c:7747 @c1d0e7a00).
fn dumpDotNodeEdge(fp: *std.c.FILE, gb: *const CGraph, node: *Tensor, parent: *Tensor, label: [*:0]const u8) void {
    const gparent = graphGetParent(gb, node);
    const gparent0 = graphGetParent(gb, parent);
    _ = fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n", if (gparent0) |g| g else parent, if (gparent) |g| g else node, if (gparent != null) @as([*:0]const u8, "empty") else "vee", if (gparent != null) @as([*:0]const u8, "dashed") else "solid", label);
}

/// Ports `ggml_graph_dump_dot_leaf_edge` (ggml.c:7758 @c1d0e7a00).
fn dumpDotLeafEdge(fp: *std.c.FILE, node: *Tensor, parent: *Tensor, label: [*:0]const u8) void {
    _ = fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n", parent, node, label);
}

/// Ports `ggml_graph_dump_dot` (ggml.c:7765 @c1d0e7a00).
///
/// Writes the graph as Graphviz DOT. Nodes are coloured by role: yellow for
/// parameters, green for tensors in the forward graph that have gradients,
/// light blue for those that have gradients but are not in it, white
/// otherwise, and pink for leafs.
///
/// Parameters:
/// - `gb`: the graph to draw, typically the backward one.
/// - `cgraph`: the forward graph, used only to pick colours. May be null.
/// - `filename`: written through `ggml_fopen`, truncating.
pub export fn ggml_graph_dump_dot(gb: *const CGraph, cgraph: ?*const CGraph, filename: [*:0]const u8) void {
    var color: [16]u8 = undefined;

    const fp = runtime.ggml_fopen(filename, "w");
    impl.assert(fp != null, "fp");
    const f = fp.?;

    _ = fprintf(f, "digraph G {\n");
    _ = fprintf(f, "  newrank = true;\n");
    _ = fprintf(f, "  rankdir = TB;\n");

    for (0..@intCast(gb.n_nodes)) |i| {
        const node = gb.nodes[i].?;
        const grad = ggml_graph_get_grad(gb, node);

        // A node that is some other node's gradient is drawn as part of that
        // node's record, not on its own.
        if (graphGetParent(gb, node) != null) continue;

        if ((node.flags & c.GGML_TENSOR_FLAG_PARAM) != 0) {
            _ = snprintf(&color, color.len, "yellow");
        } else if (grad != null) {
            if (graphFind(cgraph, node)) {
                _ = snprintf(&color, color.len, "green");
            } else {
                _ = snprintf(&color, color.len, "lightblue");
            }
        } else {
            _ = snprintf(&color, color.len, "white");
        }

        _ = fprintf(f, "  \"%p\" [ style = filled; fillcolor = %s; shape = record; label=\"", node, &color);

        if (node.name[0] != 0) {
            _ = fprintf(f, "%s (%s)|", &node.name, types.ggml_type_name(node.type));
        } else {
            _ = fprintf(f, "(%s)|", types.ggml_type_name(node.type));
        }

        if (types.ggml_is_matrix(node)) {
            _ = fprintf(f, "%d [%lld, %lld] | <x>%s", @as(c_int, @intCast(i)), node.ne[0], node.ne[1], types.ggml_op_symbol(node.op));
        } else {
            _ = fprintf(f, "%d [%lld, %lld, %lld] | <x>%s", @as(c_int, @intCast(i)), node.ne[0], node.ne[1], node.ne[2], types.ggml_op_symbol(node.op));
        }

        if (grad) |g| {
            _ = fprintf(f, " | <g>%s\"; ]\n", types.ggml_op_symbol(g.op));
        } else {
            _ = fprintf(f, "\"; ]\n");
        }
    }

    for (0..@intCast(gb.n_leafs)) |i| {
        const node = gb.leafs[i].?;

        _ = snprintf(&color, color.len, "pink");

        _ = fprintf(f, "  \"%p\" [ style = filled; fillcolor = %s; shape = record; label=\"<x>", node, &color);

        if (node.name[0] != 0) {
            _ = fprintf(f, "%s (%s)|", &node.name, types.ggml_type_name(node.type));
        } else {
            _ = fprintf(f, "(%s)|", types.ggml_type_name(node.type));
        }

        _ = fprintf(f, "CONST %d [%lld, %lld]", @as(c_int, @intCast(i)), node.ne[0], node.ne[1]);
        if (types.ggml_nelements(node) < 5 and node.data != null) {
            _ = fprintf(f, " | (");
            for (0..@intCast(types.ggml_nelements(node))) |j| {
                // FIXME: use ggml-backend to obtain the tensor data -- the C's
                // note. Until then every element prints as a placeholder.
                _ = fprintf(f, "#");
                if (j < @as(usize, @intCast(types.ggml_nelements(node) - 1))) _ = fprintf(f, ", ");
            }
            _ = fprintf(f, ")");
        }
        _ = fprintf(f, "\"; ]\n");
    }

    for (0..@intCast(gb.n_nodes)) |i| {
        const node = gb.nodes[i].?;

        for (0..c.GGML_MAX_SRC) |j| {
            if (node.src[j] != null) {
                var label: [16]u8 = undefined;
                _ = snprintf(&label, label.len, "src %d", @as(c_int, @intCast(j)));
                dumpDotNodeEdge(f, gb, node, impl.one(Tensor, node.src[j]), @ptrCast(&label));
            }
        }
    }

    for (0..@intCast(gb.n_leafs)) |i| {
        const node = gb.leafs[i].?;

        for (0..c.GGML_MAX_SRC) |j| {
            if (node.src[j] != null) {
                var label: [16]u8 = undefined;
                _ = snprintf(&label, label.len, "src %d", @as(c_int, @intCast(j)));
                dumpDotLeafEdge(f, node, impl.one(Tensor, node.src[j]), @ptrCast(&label));
            }
        }
    }

    _ = fprintf(f, "}\n");

    _ = fclose(f);

    impl.logInfo("%s: dot -Tpng %s -o %s.png && open %s.png\n", .{ "ggml_graph_dump_dot", filename, filename, filename });
}
