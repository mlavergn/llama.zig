//! Memory-range tracking and the graph reordering built on it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-metal/ggml-metal-common.cpp` at
//! v0.4.1 (`b29c606e2`). Each declaration below names the C++ it replaces
//! and the line it began at.
//!
//! # Nothing here touches Metal
//!
//! Its own header opens "helper functions for ggml-metal that are too
//! difficult to implement in Objective-C", and the body is exactly that:
//! interval arithmetic over buffer addresses, and a reordering pass the C
//! itself notes "is generic and not specific to metal". That makes it the
//! right first unit of the Metal group — it proves the swap mechanism for
//! `ggml-metal/` without any of the Metal API.
//!
//! # Why reordering is worth doing at all
//!
//! Two ops can run concurrently when neither writes memory the other
//! touches. `MemRanges` is the set of intervals the current concurrent
//! group already reads and writes; `ggml_graph_optimize` walks the graph
//! pulling later nodes forward into that group whenever they fit, and
//! starts a new group when one does not.
//!
//! # Its gate is `make node-diff ARGS=--gpu`
//!
//! A reorder is observable: `node-diff` hashes every node's output **in
//! graph order**, so a different permutation from the reference's shows up
//! as a mismatch even when every value is individually right. Measured
//! before porting — Metal node hashes are reproducible run to run and
//! across builds, 2750 nodes at one and four threads.

const std = @import("std");
const impl = @import("../impl.zig");
const fusion = @import("fusion.zig");
const backend = @import("../backend.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;
const CGraph = impl.CGraph;

/// The C `new`s and `delete`s this, and nothing hands it an allocator.
/// libc's is what a C entry point can reach — the same reasoning as
/// `backend_reg.zig`, `backend.zig` and `cpu_backend.zig`.
const allocator = std.heap.c_allocator;

/// Ports `enum ggml_mem_range_type` (ggml-metal-common.h:14 @b29c606e2).
pub const RangeType = enum(c_int) {
    src = 0,
    dst = 1,
};

/// Ports `struct ggml_mem_range` (ggml-metal-common.cpp:28 @b29c606e2): an
/// interval `[p0, p1)` within the buffer `pb`.
const MemRange = struct {
    /// Buffer id.
    pb: u64,
    /// Begin.
    p0: u64,
    /// End.
    p1: u64,
    pt: RangeType,
};

/// Ports `ggml_metal_op_mul_mat_use_mm` (ggml-metal-common.cpp:10 @b29c606e2).
///
/// Whether `MUL_MAT` takes the matrix-matrix kernel rather than a
/// matrix-vector one: the break-even the encoder and the Objective-C
/// `supports_op` must agree on, which is why it lives here and not in the
/// encoder. The matrix-matrix kernel needs simdgroup matrices (A14/M1 and
/// later).
pub export fn ggml_metal_op_mul_mat_use_mm(op: *const c.ggml_tensor, has_simdgroup_mm: bool) callconv(.c) bool {
    const src0 = impl.one(c.ggml_tensor, op.src[0]);
    const src1 = impl.one(c.ggml_tensor, op.src[1]);
    const ne00 = src0.ne[0];
    const ne11 = src1.ne[1];

    return !c.ggml_is_transposed(src0) and
        !c.ggml_is_transposed(src1) and
        has_simdgroup_mm and ne00 >= 64 and ne11 > 8;
}

/// Ports `ggml_metal_op_mul_mat_id_use_mm` (ggml-metal-common.cpp:19 @b29c606e2).
///
/// `MUL_MAT_ID`'s counterpart: the matrix-matrix kernel once the batch
/// (`ne21`, rows of the expert ids) reaches 32.
pub export fn ggml_metal_op_mul_mat_id_use_mm(op: *const c.ggml_tensor, has_simdgroup_mm: bool) callconv(.c) bool {
    const ne00 = impl.one(c.ggml_tensor, op.src[0]).ne[0];
    const ne21 = impl.one(c.ggml_tensor, op.src[2]).ne[1];

    return has_simdgroup_mm and ne00 >= 64 and ne21 >= 32;
}

/// Ports `struct ggml_mem_ranges` (ggml-metal-common.cpp:37 @b29c606e2).
///
/// Opaque to every caller — the C's `ggml_mem_ranges_t` is a pointer to an
/// incomplete type — so the layout is ours to choose.
pub const MemRanges = struct {
    ranges: std.ArrayList(MemRange),
    debug: c_int = 0,
};

/// Ports `ggml_mem_ranges_init` (ggml-metal-common.cpp:43 @b29c606e2).
///
/// Parameters:
/// - `debug`: the C's debug level; above 2, every range is logged.
///
/// Return: a new range set, owned by the caller and released with
/// `ggml_mem_ranges_free`.
pub export fn ggml_mem_ranges_init(debug: c_int) callconv(.c) ?*MemRanges {
    // `new` in the C, so a failure terminates rather than returning null.
    const res = allocator.create(MemRanges) catch impl.abort("out of memory allocating a memory-range set");
    res.* = .{ .ranges = .empty, .debug = debug };
    // The C's `reserve(256)`. A failure here alone is not fatal -- the
    // list grows on demand -- so unlike the rest of this file it is
    // allowed to pass.
    res.ranges.ensureTotalCapacity(allocator, 256) catch {};
    return res;
}

/// Ports `ggml_mem_ranges_free` (ggml-metal-common.cpp:52 @b29c606e2).
pub export fn ggml_mem_ranges_free(mrs: ?*MemRanges) callconv(.c) void {
    const m = mrs orelse return;
    m.ranges.deinit(allocator);
    allocator.destroy(m);
}

/// Ports `ggml_mem_ranges_reset` (ggml-metal-common.cpp:56 @b29c606e2).
pub export fn ggml_mem_ranges_reset(mrs: *MemRanges) callconv(.c) void {
    mrs.ranges.clearRetainingCapacity();
}

/// Ports the `ggml_mem_range` overload of `ggml_mem_ranges_add`
/// (ggml-metal-common.cpp:60 @b29c606e2).
///
/// The C has two functions of this name, separated by their parameter
/// type; Zig has no overloading, so the private one is spelled out.
///
/// **Allocation failure aborts.** The C's `push_back` throws `bad_alloc`
/// with nothing to catch it, so it reaches `std::terminate`. Returning
/// false instead would be worse than dying: every caller treats it as "no
/// conflict found", which silently yields a *different graph order* and a
/// gate that reports a porting bug.
fn addRange(mrs: *MemRanges, mr: MemRange) bool {
    mrs.ranges.append(allocator, mr) catch impl.abort("out of memory tracking a memory range");
    return true;
}

/// Ports `ggml_mem_range_from_tensor` (ggml-metal-common.cpp:66
/// @b29c606e2).
fn rangeFromTensor(tensor_in: *const Tensor, pt: RangeType) MemRange {
    // always use the base tensor
    const tensor = if (tensor_in.view_src != null)
        impl.one(Tensor, tensor_in.view_src)
    else
        tensor_in;

    impl.assert(tensor.view_src == null, "!tensor->view_src");

    if (tensor.buffer != null) {
        // When the tensor is allocated, use the actual memory address
        // range in the buffer. The allocated size can exceed the tensor's
        // own when the buffer type pads, so it is asked for rather than
        // computed — upstream PR 15966.
        const base = @intFromPtr(tensor.data);
        return .{
            .pb = @intFromPtr(tensor.buffer),
            .p0 = base,
            .p1 = base + backend.ggml_backend_buft_get_alloc_size(tensor.buffer.*.buft, tensor),
            .pt = pt,
        };
    }

    // Otherwise the tensor's own address is a unique id for whatever
    // ranges it will use once allocated, and `[0, 1024)` is a dummy.
    return .{
        .pb = @intFromPtr(tensor),
        .p0 = 0,
        .p1 = 1024,
        .pt = pt,
    };
}

/// Ports `ggml_mem_range_from_tensor_src` (ggml-metal-common.cpp:100
/// @b29c606e2).
fn rangeFromTensorSrc(tensor: *const Tensor) MemRange {
    return rangeFromTensor(tensor, .src);
}

/// Ports `ggml_mem_range_from_tensor_dst` (ggml-metal-common.cpp:104
/// @b29c606e2).
fn rangeFromTensorDst(tensor: *const Tensor) MemRange {
    return rangeFromTensor(tensor, .dst);
}

/// Ports `ggml_mem_ranges_add_src` (ggml-metal-common.cpp:108 @b29c606e2).
fn addSrc(mrs: *MemRanges, tensor: *const Tensor) bool {
    const mr = rangeFromTensorSrc(tensor);
    if (mrs.debug > 2) {
        impl.logDebug("%s: add src range buf=%lld, [%lld, %lld)\n", .{
            "ggml_mem_ranges_add_src", mr.pb, mr.p0, mr.p1,
        });
    }
    return addRange(mrs, mr);
}

/// Ports `ggml_mem_ranges_add_dst` (ggml-metal-common.cpp:120 @b29c606e2).
fn addDst(mrs: *MemRanges, tensor: *const Tensor) bool {
    const mr = rangeFromTensorDst(tensor);
    if (mrs.debug > 2) {
        impl.logDebug("%s: add dst range buf=%lld, [%lld, %lld)\n", .{
            "ggml_mem_ranges_add_dst", mr.pb, mr.p0, mr.p1,
        });
    }
    return addRange(mrs, mr);
}

/// Ports the tensor overload of `ggml_mem_ranges_add`
/// (ggml-metal-common.cpp:132 @b29c606e2): every source, then the
/// destination.
///
/// Parameters:
/// - `mrs`: the range set to extend.
/// - `tensor`: the node whose sources and destination are tracked.
///
/// Return: whether the destination range was added. The C ignores the
/// sources' results here, and so does this.
pub export fn ggml_mem_ranges_add(mrs: *MemRanges, tensor: *const Tensor) callconv(.c) bool {
    for (0..c.GGML_MAX_SRC) |i| {
        if (tensor.src[i]) |src| _ = addSrc(mrs, src);
    }
    return addDst(mrs, tensor);
}

/// Ports the `ggml_mem_range` overload of `ggml_mem_ranges_check`
/// (ggml-metal-common.cpp:142 @b29c606e2).
fn checkRange(mrs: *MemRanges, mr: MemRange) bool {
    for (mrs.ranges.items) |cmp| {
        // two memory ranges cannot intersect if they are in different buffers
        if (mr.pb != cmp.pb) continue;

        // intersecting source ranges are allowed
        if (mr.pt == .src and cmp.pt == .src) continue;

        // The C's test, asymmetric `<` and `>=`, verbatim.
        if (mr.p0 < cmp.p1 and mr.p1 >= cmp.p0) {
            if (mrs.debug > 2) {
                impl.logDebug(
                    "%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                    .{
                        "ggml_mem_ranges_check",
                        if (mr.pt == .src) "src" else "dst",
                        mr.pb,
                        mr.p0,
                        mr.p1,
                        if (cmp.pt == .src) "src" else "dst",
                        cmp.pb,
                        cmp.p0,
                        cmp.p1,
                    },
                );
            }
            return false;
        }
    }
    return true;
}

/// Ports `ggml_mem_ranges_check_src` (ggml-metal-common.cpp:173
/// @b29c606e2).
fn checkSrc(mrs: *MemRanges, tensor: *const Tensor) bool {
    return checkRange(mrs, rangeFromTensorSrc(tensor));
}

/// Ports `ggml_mem_ranges_check_dst` (ggml-metal-common.cpp:183
/// @b29c606e2).
fn checkDst(mrs: *MemRanges, tensor: *const Tensor) bool {
    return checkRange(mrs, rangeFromTensorDst(tensor));
}

/// Ports the tensor overload of `ggml_mem_ranges_check`
/// (ggml-metal-common.cpp:193 @b29c606e2).
///
/// Parameters:
/// - `mrs`: the ranges already in the concurrent set.
/// - `tensor`: the node being considered for it.
///
/// Return: false when a new source range overlaps an existing destination
/// range, or a new destination range overlaps any existing range.
pub export fn ggml_mem_ranges_check(mrs: *MemRanges, tensor: *const Tensor) callconv(.c) bool {
    for (0..c.GGML_MAX_SRC) |i| {
        if (tensor.src[i]) |src| {
            if (!checkSrc(mrs, src)) return false;
        }
    }
    return checkDst(mrs, tensor);
}

// -----------------------------------------------------------------------------
// The reordering pass

/// Ports `struct node_info` (ggml-metal-common.cpp:205 @b29c606e2): one
/// graph node plus the nodes fused onto it.
const NodeInfo = struct {
    node: *Tensor,
    fused: std.ArrayList(*Tensor) = .empty,

    fn op(self: NodeInfo) c.enum_ggml_op {
        return self.node.op;
    }

    /// The last fused tensor becomes the destination, because that is
    /// where a fused run actually writes.
    fn dst(self: NodeInfo) *const Tensor {
        return if (self.fused.items.len == 0) self.node else self.fused.items[self.fused.items.len - 1];
    }

    fn isEmpty(self: NodeInfo) bool {
        return impl.opIsEmpty(self.node.op);
    }
};

/// Ports the `h_add` lambda (ggml-metal-common.cpp:229 @b29c606e2): every
/// source of the node and of everything fused onto it, then the
/// destination.
fn hAdd(mrs: *MemRanges, node: NodeInfo) bool {
    for (0..c.GGML_MAX_SRC) |i| {
        if (node.node.src[i]) |src| {
            if (!addSrc(mrs, src)) return false;
        }
    }
    // keep track of the sources of the fused nodes as well
    for (node.fused.items) |fused| {
        for (0..c.GGML_MAX_SRC) |i| {
            if (fused.src[i]) |src| {
                if (!addSrc(mrs, src)) return false;
            }
        }
    }
    return addDst(mrs, node.dst());
}

/// Ports the `h_check` lambda (ggml-metal-common.cpp:253 @b29c606e2).
fn hCheck(mrs: *MemRanges, node: NodeInfo) bool {
    for (0..c.GGML_MAX_SRC) |i| {
        if (node.node.src[i]) |src| {
            if (!checkSrc(mrs, src)) return false;
        }
    }
    for (node.fused.items) |fused| {
        for (0..c.GGML_MAX_SRC) |i| {
            if (fused.src[i]) |src| {
                if (!checkSrc(mrs, src)) return false;
            }
        }
    }
    return checkDst(mrs, node.dst());
}

/// Ports the `h_safe` lambda (ggml-metal-common.cpp:277 @b29c606e2): the
/// ops a reorder may cross. The C's comment says it "can be expanded when
/// needed", so this list is a policy and not a property.
fn hSafe(op: c.enum_ggml_op) bool {
    return switch (op) {
        c.GGML_OP_MUL_MAT,
        c.GGML_OP_MUL_MAT_ID,
        c.GGML_OP_ROPE,
        c.GGML_OP_NORM,
        c.GGML_OP_RMS_NORM,
        c.GGML_OP_GROUP_NORM,
        c.GGML_OP_L2_NORM,
        c.GGML_OP_SUM_ROWS,
        c.GGML_OP_SSM_CONV,
        c.GGML_OP_SSM_SCAN,
        c.GGML_OP_CLAMP,
        c.GGML_OP_TRI,
        c.GGML_OP_DIAG,
        c.GGML_OP_MUL,
        c.GGML_OP_ADD,
        c.GGML_OP_SUB,
        c.GGML_OP_DIV,
        c.GGML_OP_GLU,
        c.GGML_OP_SCALE,
        c.GGML_OP_UNARY,
        c.GGML_OP_GET_ROWS,
        c.GGML_OP_SET_ROWS,
        c.GGML_OP_SET,
        c.GGML_OP_CPY,
        c.GGML_OP_CONT,
        c.GGML_OP_REPEAT,
        => true,
        else => impl.opIsEmpty(op),
    };
}

/// Ports `ggml_metal_graph_optimize_reorder` (ggml-metal-common.cpp:227
/// @b29c606e2).
///
/// Parameters:
/// - `arena`: owns the returned permutation and the scratch `used` set.
/// - `nodes`: the fused nodes, in graph order.
///
/// Return: the order to visit them in, as indices into `nodes`. Borrowed
/// from `arena`.
fn reorder(arena: std.mem.Allocator, nodes: []const NodeInfo) []const c_int {
    const n: c_int = @intCast(nodes.len);

    var res: std.ArrayList(c_int) = .empty;
    res.ensureTotalCapacity(arena, nodes.len) catch impl.abort("out of memory reordering the graph");

    const used = arena.alloc(bool, nodes.len) catch impl.abort("out of memory reordering the graph");
    @memset(used, false);

    // the memory ranges for the set of currently concurrent nodes
    const mrs0 = ggml_mem_ranges_init(0).?;
    defer ggml_mem_ranges_free(mrs0);

    // the memory ranges for the set of nodes that haven't been processed
    // yet, when looking forward for a node to reorder
    const mrs1 = ggml_mem_ranges_init(0).?;
    defer ggml_mem_ranges_free(mrs1);

    var j0: c_int = 0;
    while (j0 < n) : (j0 += 1) {
        if (used[@intCast(j0)]) continue;

        const node0 = nodes[@intCast(j0)];

        // The node is not concurrent with the existing set, so a barrier
        // has to go in — but first, look forward for nodes that can join
        // the set instead. Empty nodes always can: they read and write
        // nothing.
        if (!node0.isEmpty() and !hCheck(mrs0, node0)) {
            // the ranges of the nodes not yet processed; a node that is
            // not concurrent with these cannot be moved across them
            ggml_mem_ranges_reset(mrs1);
            _ = hAdd(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            const n_forward: c_int = 64;

            var j1: c_int = j0 + 1;
            while (j1 < j0 + n_forward and j1 < n) : (j1 += 1) {
                if (used[@intCast(j1)]) continue;

                const node1 = nodes[@intCast(j1)];

                // disallow reordering of certain ops
                if (!hSafe(node1.op())) break;

                const is_empty = node1.isEmpty();

                // To join the concurrent set a node has to be empty or
                // concurrent with everything in it, *and* concurrent with
                // everything before it that is still unprocessed.
                if ((is_empty or hCheck(mrs0, node1)) and hCheck(mrs1, node1)) {
                    _ = hAdd(mrs0, node1);
                    res.appendAssumeCapacity(j1);
                    used[@intCast(j1)] = true;
                } else {
                    _ = hAdd(mrs1, node1);
                }
            }

            // finalize the concurrent set and begin a new one
            ggml_mem_ranges_reset(mrs0);
        }

        // expand the concurrent set with the current node
        _ = hAdd(mrs0, node0);
        res.appendAssumeCapacity(j0);
    }

    return res.items;
}

/// Ports `ggml_graph_optimize` (ggml-metal-common.cpp:393 @b29c606e2).
///
/// Fuses what can be fused, reorders the fused nodes for concurrency, then
/// writes the nodes back out unfused. Fusing first is what keeps a reorder
/// from splitting a run the backend wants to issue as one kernel. Which
/// runs are packed is the fusion table's answer (`fusion.zig`), the same
/// table the op encoders consult.
///
/// Parameters:
/// - `gf`: the graph, reordered in place. `n_nodes` does not change.
pub export fn ggml_graph_optimize(gf: *CGraph) callconv(.c) void {
    const n = gf.n_nodes;

    // The C's `std::vector<node_info>` and its nested `fused` vectors, and
    // the permutation. All of it dies at the end of this call, so one
    // arena replaces four vectors' worth of individual frees.
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var nodes: std.ArrayList(NodeInfo) = .empty;
    nodes.ensureTotalCapacity(arena, @intCast(n)) catch impl.abort("out of memory optimizing the graph");

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table, so the packing here is driven
    // by the same patterns that the op encoders will later use
    var i: c_int = 0;
    while (i < n) : (i += 1) {
        var node: NodeInfo = .{ .node = gf.nodes[@intCast(i)].? };

        const f = fusion.ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        var k: c_int = 1;
        while (k < f) : (k += 1) {
            i += 1;

            // the .dst() becomes the last fused tensor
            node.fused.append(arena, gf.nodes[@intCast(i)].?) catch impl.abort("out of memory fusing nodes");
        }

        nodes.appendAssumeCapacity(node);
    }

    // reorder to improve concurrency
    const order = reorder(arena, nodes.items);

    // unfuse
    var j: usize = 0;
    for (order) |idx| {
        const node = nodes.items[@intCast(idx)];
        gf.nodes[j] = node.node;
        j += 1;
        for (node.fused.items) |fused| {
            gf.nodes[j] = fused;
            j += 1;
        }
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a range set starts empty and admits anything" {
    const mrs = ggml_mem_ranges_init(0).?;
    defer ggml_mem_ranges_free(mrs);
    try std.testing.expectEqual(@as(usize, 0), mrs.ranges.items.len);

    // Two unallocated tensors get distinct ids, so they never conflict.
    var a = std.mem.zeroes(Tensor);
    var b = std.mem.zeroes(Tensor);
    try std.testing.expect(ggml_mem_ranges_check(mrs, &a));
    try std.testing.expect(ggml_mem_ranges_add(mrs, &a));
    try std.testing.expect(ggml_mem_ranges_check(mrs, &b));
}

test "a destination conflicts with itself, two sources do not" {
    const mrs = ggml_mem_ranges_init(0).?;
    defer ggml_mem_ranges_free(mrs);

    var t = std.mem.zeroes(Tensor);
    var user = std.mem.zeroes(Tensor);
    user.src[0] = &t;

    // `t` as a destination, then `t` as a source: dst-vs-src overlaps.
    try std.testing.expect(addDst(mrs, &t));
    try std.testing.expect(!checkSrc(mrs, &t));

    // The same two source ranges are allowed to intersect.
    ggml_mem_ranges_reset(mrs);
    try std.testing.expect(addSrc(mrs, &t));
    try std.testing.expect(checkSrc(mrs, &t));
}

test "reset keeps the capacity the C reserved" {
    const mrs = ggml_mem_ranges_init(0).?;
    defer ggml_mem_ranges_free(mrs);
    const cap = mrs.ranges.capacity;
    try std.testing.expect(cap >= 256);

    var t = std.mem.zeroes(Tensor);
    _ = addDst(mrs, &t);
    ggml_mem_ranges_reset(mrs);
    try std.testing.expectEqual(@as(usize, 0), mrs.ranges.items.len);
    try std.testing.expectEqual(cap, mrs.ranges.capacity);
}
