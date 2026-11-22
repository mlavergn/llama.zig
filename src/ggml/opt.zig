//! The training API: datasets, the optimization context that turns a forward
//! graph into gradient and optimizer-step graphs, results, and the
//! high-level `ggml_opt_epoch` / `ggml_opt_fit` loops.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-opt.cpp` (v1.0.0, `b1b770d90`), 1,094
//! lines of C++, plus the three pieces of libc++ its one `std::shuffle`
//! reaches, which are not upstream code and are cited by header and template:
//! `__random/mersenne_twister_engine.h`, `__random/uniform_int_distribution.h`
//! and `__algorithm/shuffle.h`, from the libc++ 22.1.8 bundled with Zig 0.17
//! (`lib/libcxx/include/`), which is what `build/llamacpp.zig` compiles the
//! reference against. Each declaration below names the C++ it replaces and
//! the line it began at.
//!
//! # The contract is 37 symbols, not 9
//!
//! An early survey recorded 9; measured against `zig c++ -c` at the pinned
//! commit the unmangled export count is **37**. The mangled remainder is
//! `std::vector`, `std::map` and `<random>` template noise that no other
//! object references. The one caller outside ggml is `llama-context.cpp`'s
//! training entry points, which go through `ggml-opt.h` alone.
//!
//! # Exports
//!
//! The 37 functions are `pub fn … callconv(.c)`; `src/ggml/lib.zig`, the
//! one C-FFI root, exports them by name (`exported_names`). `make opt-diff`
//! compiles this file on its own beside the C++, so there it defines no
//! symbols and is reached only through Zig.
//!
//! # `std::shuffle` is reproduced, not merely "a shuffle"
//!
//! `ggml_opt_dataset_shuffle` draws from a `std::mt19937` through
//! `std::shuffle`, whose use of `uniform_int_distribution<ptrdiff_t>` is left
//! to the library. MT19937 itself is fully specified; the shuffle's loop
//! shape and the distribution's rejection sampling are libc++'s, and are
//! reproduced below so the permutation, and therefore every batch a training
//! run sees, matches the reference bit for bit. `make opt-diff` compares the
//! permutation directly, and the distribution over ranges up to 2^63.
//!
//! # Float contraction
//!
//! Clang contracts one expression in this translation unit, measured from the
//! reference object's disassembly: `sum_squared/nbatches - mean*mean` in
//! `ggml_opt_result_loss` (and its inlined copy in the progress bar) is one
//! `fmsub`. That site is named with `@mulAdd`; nothing else here fuses.
//!
//! # Deliberate differences
//!
//! - **`GGML_ASSERT` messages.** They go through `impl.assert`, which names
//!   this port rather than `ggml-opt.cpp:LINE`; the abort is the same.

const std = @import("std");
const mod = @import("module.zig");
const stdlib = @import("stdlib");

/// The heap the rest of ggml uses. A function entered through a C ABI has no
/// allocator to be handed one, the same reasoning as `backend.zig`.

// -----------------------------------------------------------------------------
// Export switch

/// The unmangled exports of `ggml-opt.cpp`, in source order.
pub const exported_names = [_][]const u8{
    "ggml_opt_dataset_init",
    "ggml_opt_dataset_free",
    "ggml_opt_dataset_ndata",
    "ggml_opt_dataset_data",
    "ggml_opt_dataset_labels",
    "ggml_opt_dataset_shuffle",
    "ggml_opt_dataset_get_batch",
    "ggml_opt_dataset_get_batch_host",
    "ggml_opt_get_default_optimizer_params",
    "ggml_opt_get_constant_optimizer_params",
    "ggml_opt_default_params",
    "ggml_opt_init",
    "ggml_opt_free",
    "ggml_opt_reset",
    "ggml_opt_static_graphs",
    "ggml_opt_inputs",
    "ggml_opt_outputs",
    "ggml_opt_labels",
    "ggml_opt_loss",
    "ggml_opt_pred",
    "ggml_opt_ncorrect",
    "ggml_opt_grad_acc",
    "ggml_opt_result_init",
    "ggml_opt_result_free",
    "ggml_opt_result_reset",
    "ggml_opt_result_ndata",
    "ggml_opt_result_loss",
    "ggml_opt_result_pred",
    "ggml_opt_result_accuracy",
    "ggml_opt_prepare_alloc",
    "ggml_opt_alloc",
    "ggml_opt_eval",
    "ggml_opt_epoch",
    "ggml_opt_epoch_callback_progress_bar",
    "ggml_opt_fit",
    "ggml_opt_context_optimizer_type",
    "ggml_opt_optimizer_name",
};

// -----------------------------------------------------------------------------
// The three opaque structs

/// Ports `struct ggml_opt_dataset` (ggml-opt.cpp:16 @b1b770d90).
///
/// `nbs_data` and `nbs_labels` are `size_t` initialised to `-1` in the C++,
/// so they start at the maximum.
const Dataset = struct {
    ctx: ?*mod.impl.ggml_c.ggml_context = null,
    buf: mod.impl.ggml_c.ggml_backend_buffer_t = null,
    data: [*c]mod.impl.ggml_c.ggml_tensor = null,
    labels: [*c]mod.impl.ggml_c.ggml_tensor = null,

    ndata: i64 = -1,
    ndata_shard: i64 = -1,
    nbs_data: usize = std.math.maxInt(usize),
    nbs_labels: usize = std.math.maxInt(usize),

    permutation: []i64 = &.{},
};

/// Ports `struct ggml_opt_context` (ggml-opt.cpp:30 @b1b770d90).
///
/// The three vectors of tensors are indexed by forward-graph node, with null
/// for nodes that need no accumulator or momentum.
const Context = struct {
    backend_sched: mod.impl.ggml_c.ggml_backend_sched_t = null,
    allocated_graph: ?*mod.impl.ggml_c.ggml_cgraph = null,
    allocated_graph_copy: ?*mod.impl.ggml_c.ggml_cgraph = null,
    ctx_static: ?*mod.impl.ggml_c.ggml_context = null,
    ctx_cpu: ?*mod.impl.ggml_c.ggml_context = null,
    ctx_compute: ?*mod.impl.ggml_c.ggml_context = null,
    ctx_copy: ?*mod.impl.ggml_c.ggml_context = null,
    buf_static: mod.impl.ggml_c.ggml_backend_buffer_t = null,
    buf_cpu: mod.impl.ggml_c.ggml_backend_buffer_t = null,
    rng: stdlib.libcxx.Mt19937 = undefined,
    loss_type: mod.impl.ggml_c.enum_ggml_opt_loss_type = undefined,
    build_type: mod.impl.ggml_c.enum_ggml_opt_build_type = undefined,
    build_type_alloc: mod.impl.ggml_c.enum_ggml_opt_build_type = undefined,

    inputs: [*c]mod.impl.ggml_c.ggml_tensor = null,
    outputs: [*c]mod.impl.ggml_c.ggml_tensor = null,
    labels: [*c]mod.impl.ggml_c.ggml_tensor = null,

    loss: [*c]mod.impl.ggml_c.ggml_tensor = null,
    pred: [*c]mod.impl.ggml_c.ggml_tensor = null,
    ncorrect: [*c]mod.impl.ggml_c.ggml_tensor = null,

    gf: ?*mod.impl.ggml_c.ggml_cgraph = null,
    gb_grad: ?*mod.impl.ggml_c.ggml_cgraph = null,
    gb_opt: ?*mod.impl.ggml_c.ggml_cgraph = null,
    static_graphs: bool = false,
    eval_ready: bool = false,
    grad_accs: std.ArrayList([*c]mod.impl.ggml_c.ggml_tensor) = .empty,
    grad_m: std.ArrayList([*c]mod.impl.ggml_c.ggml_tensor) = .empty,
    grad_v: std.ArrayList([*c]mod.impl.ggml_c.ggml_tensor) = .empty,

    iter: i64 = 1,
    opt_period: i32 = 1,
    opt_i: i32 = 0,
    loss_per_datapoint: bool = false,

    get_opt_pars: mod.impl.ggml_c.ggml_opt_get_optimizer_params = null,
    get_opt_pars_ud: ?*anyopaque = null,
    /// Stores output of get_opt_pars.
    opt_step_params: [*c]mod.impl.ggml_c.ggml_tensor = null,

    optimizer: mod.impl.ggml_c.enum_ggml_opt_optimizer_type = mod.impl.ggml_c.GGML_OPT_OPTIMIZER_TYPE_ADAMW,
};

/// Ports `struct ggml_opt_result` (ggml-opt.cpp:74 @b1b770d90).
const Result = struct {
    ndata: i64 = 0,
    loss: std.ArrayList(f32) = .empty,
    pred: std.ArrayList(i32) = .empty,
    ncorrect: i64 = 0,

    opt_period: i64 = -1,
    loss_per_datapoint: bool = false,
};

inline fn datasetOf(p: mod.impl.ggml_c.ggml_opt_dataset_t) *Dataset {
    return @ptrCast(@alignCast(p));
}

inline fn contextOf(p: mod.impl.ggml_c.ggml_opt_context_t) *Context {
    return @ptrCast(@alignCast(p));
}

inline fn resultOf(p: mod.impl.ggml_c.ggml_opt_result_t) *Result {
    return @ptrCast(@alignCast(p));
}

inline fn graphOf(g: ?*mod.impl.ggml_c.ggml_cgraph) *mod.impl.CGraph {
    return @ptrCast(@alignCast(g));
}

inline fn tensorOf(t: [*c]mod.impl.ggml_c.ggml_tensor) *mod.impl.ggml_c.ggml_tensor {
    return mod.impl.one(mod.impl.ggml_c.ggml_tensor, t);
}

/// `new T` where `T`'s constructor cannot fail: allocation failure is the one
/// thing the C++ lets escape as `std::bad_alloc`, and here it aborts, as every
/// other ported ggml allocation does.
fn create(comptime T: type) *T {
    const p = std.heap.c_allocator.create(T) catch mod.impl.abort("ggml-opt: out of memory");
    p.* = .{};
    return p;
}

fn oom() noreturn {
    mod.impl.abort("ggml-opt: out of memory");
}

// -----------------------------------------------------------------------------
// Dataset

/// Ports `ggml_opt_dataset_init` (ggml-opt.cpp:86 @b1b770d90).
///
/// Parameters:
/// - `type_data`: the type of the internal data tensor.
/// - `type_label`: the type of the internal labels tensor.
/// - `ne_datapoint`: elements per datapoint, positive.
/// - `ne_label`: elements per label; 0 for no labels.
/// - `ndata`: total datapoints, positive.
/// - `ndata_shard`: datapoints per shard, the unit of shuffling and copying.
///
/// Return: a dataset whose data lives in a CPU buffer; free it with
/// `ggml_opt_dataset_free`.
pub fn ggml_opt_dataset_init(
    type_data: mod.impl.ggml_c.enum_ggml_type,
    type_label: mod.impl.ggml_c.enum_ggml_type,
    ne_datapoint: i64,
    ne_label: i64,
    ndata: i64,
    ndata_shard: i64,
) callconv(.c) mod.impl.ggml_c.ggml_opt_dataset_t {
    mod.impl.assert(ne_datapoint > 0, "ne_datapoint >  0");
    mod.impl.assert(ne_label >= 0, "ne_label     >= 0");
    mod.impl.assert(ndata > 0, "ndata        >  0");
    mod.impl.assert(ndata_shard > 0, "ndata_shard  >  0");

    const result = create(Dataset);
    result.ndata = ndata;
    result.ndata_shard = ndata_shard;

    {
        const params: mod.impl.ggml_c.ggml_init_params = .{
            .mem_size = 2 * mod.impl.ggml_c.ggml_tensor_overhead(),
            .mem_buffer = null,
            .no_alloc = true,
        };
        result.ctx = mod.impl.ggml_c.ggml_init(params);
    }

    result.data = mod.impl.ggml_c.ggml_new_tensor_2d(result.ctx, type_data, ne_datapoint, ndata);
    result.nbs_data = mod.impl.ggml_c.ggml_nbytes(result.data) * @as(usize, @bitCast(ndata_shard)) / @as(usize, @bitCast(ndata));

    if (ne_label > 0) {
        result.labels = mod.impl.ggml_c.ggml_new_tensor_2d(result.ctx, type_label, ne_label, ndata);
        result.nbs_labels = mod.impl.ggml_c.ggml_nbytes(result.labels) * @as(usize, @bitCast(ndata_shard)) / @as(usize, @bitCast(ndata));
    } else {
        result.labels = null;
        result.nbs_labels = 0;
    }

    result.buf = mod.impl.ggml_c.ggml_backend_alloc_ctx_tensors_from_buft(result.ctx, mod.impl.ggml_c.ggml_backend_cpu_buffer_type());

    const nshards = @divTrunc(ndata, ndata_shard);
    result.permutation = std.heap.c_allocator.alloc(i64, @intCast(nshards)) catch oom();
    for (result.permutation, 0..) |*p, i| p.* = @intCast(i);
    return @ptrCast(result);
}

/// Ports `ggml_opt_dataset_free` (ggml-opt.cpp:132 @b1b770d90).
///
/// Parameters:
/// - `dataset`: the dataset; invalid afterwards.
///
/// Return: nothing.
pub fn ggml_opt_dataset_free(dataset: mod.impl.ggml_c.ggml_opt_dataset_t) callconv(.c) void {
    const ds = datasetOf(dataset);
    mod.impl.ggml_c.ggml_backend_buffer_free(ds.buf);
    mod.impl.ggml_c.ggml_free(ds.ctx);
    std.heap.c_allocator.free(ds.permutation);
    std.heap.c_allocator.destroy(ds);
}

/// Ports `ggml_opt_dataset_ndata` (ggml-opt.cpp:138 @b1b770d90).
///
/// Parameters:
/// - `dataset`: the dataset.
///
/// Return: the number of datapoints.
pub fn ggml_opt_dataset_ndata(dataset: mod.impl.ggml_c.ggml_opt_dataset_t) callconv(.c) i64 {
    return datasetOf(dataset).ndata;
}

/// Ports `ggml_opt_dataset_data` (ggml-opt.cpp:142 @b1b770d90).
///
/// Parameters:
/// - `dataset`: the dataset.
///
/// Return: the data tensor, `[ne_datapoint, ndata]`; owned by the dataset.
pub fn ggml_opt_dataset_data(dataset: mod.impl.ggml_c.ggml_opt_dataset_t) callconv(.c) [*c]mod.impl.ggml_c.ggml_tensor {
    return datasetOf(dataset).data;
}

/// Ports `ggml_opt_dataset_labels` (ggml-opt.cpp:146 @b1b770d90).
///
/// Parameters:
/// - `dataset`: the dataset.
///
/// Return: the labels tensor, `[ne_label, ndata]`, or null; owned by the
/// dataset.
pub fn ggml_opt_dataset_labels(dataset: mod.impl.ggml_c.ggml_opt_dataset_t) callconv(.c) [*c]mod.impl.ggml_c.ggml_tensor {
    return datasetOf(dataset).labels;
}

/// Ports `ggml_opt_dataset_shuffle` (ggml-opt.cpp:150 @b1b770d90).
///
/// Parameters:
/// - `opt_ctx`: the context whose engine supplies the randomness.
/// - `dataset`: the dataset whose shard permutation is shuffled.
/// - `idata`: shuffle the first `idata` datapoints, a multiple of the shard
///   size; negative shuffles all of them.
///
/// Return: nothing.
pub fn ggml_opt_dataset_shuffle(opt_ctx: mod.impl.ggml_c.ggml_opt_context_t, dataset: mod.impl.ggml_c.ggml_opt_dataset_t, idata: i64) callconv(.c) void {
    const ds = datasetOf(dataset);
    const ctx = contextOf(opt_ctx);
    mod.impl.assert(idata <= ds.ndata, "idata <= dataset->ndata");

    if (idata < 0) {
        stdlib.libcxx.shuffle(i64, ds.permutation, &ctx.rng);
        return;
    }

    mod.impl.assert(@rem(idata, ds.ndata_shard) == 0, "idata % dataset->ndata_shard == 0");
    const ishard_max = @divTrunc(idata, ds.ndata_shard);
    stdlib.libcxx.shuffle(i64, ds.permutation[0..@intCast(ishard_max)], &ctx.rng);
}

/// Ports `ggml_opt_dataset_get_batch` (ggml-opt.cpp:163 @b1b770d90).
///
/// Parameters:
/// - `dataset`: the dataset.
/// - `data_batch`: contiguous destination for the data, of the dataset's type.
/// - `labels_batch`: contiguous destination for the labels; null exactly when
///   the dataset has none.
/// - `ibatch`: which batch, in the current permutation.
///
/// Return: nothing.
pub fn ggml_opt_dataset_get_batch(
    dataset: mod.impl.ggml_c.ggml_opt_dataset_t,
    data_batch: [*c]mod.impl.ggml_c.ggml_tensor,
    labels_batch: [*c]mod.impl.ggml_c.ggml_tensor,
    ibatch: i64,
) callconv(.c) void {
    const ds = datasetOf(dataset);
    mod.impl.assert(data_batch != null and mod.impl.ggml_c.ggml_is_contiguous(data_batch), "   data_batch && ggml_is_contiguous(data_batch)");
    mod.impl.assert(labels_batch == null or mod.impl.ggml_c.ggml_is_contiguous(labels_batch), "!labels_batch || ggml_is_contiguous(labels_batch)");
    mod.impl.assert((labels_batch == null) == (ds.labels == null), "(labels_batch == nullptr) == (dataset->labels == nullptr)");
    mod.impl.assert(tensorOf(data_batch).type == tensorOf(ds.data).type, "                   data_batch->type == dataset->data->type");
    mod.impl.assert(labels_batch == null or tensorOf(labels_batch).type == tensorOf(ds.labels).type, "!labels_batch || labels_batch->type == dataset->labels->type");

    const nb_data_batch = mod.impl.ggml_c.ggml_nbytes(data_batch);
    mod.impl.assert(nb_data_batch % ds.nbs_data == 0, "nb_data_batch % dataset->nbs_data == 0");
    const shards_per_batch: i64 = @intCast(nb_data_batch / ds.nbs_data);

    if (labels_batch != null) {
        const nb_labels_batch = mod.impl.ggml_c.ggml_nbytes(labels_batch);
        mod.impl.assert(nb_labels_batch == @as(usize, @intCast(shards_per_batch)) *% ds.nbs_labels, "nb_labels_batch == shards_per_batch*dataset->nbs_labels");
    }

    mod.impl.assert((ibatch + 1) * shards_per_batch <= @as(i64, @intCast(ds.permutation.len)), "(ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size())");

    var ishard_batch: i64 = 0;
    while (ishard_batch < shards_per_batch) : (ishard_batch += 1) {
        const ishard = ds.permutation[@intCast(ibatch * shards_per_batch + ishard_batch)];

        const ptr_data: [*]const u8 = @as([*]const u8, @ptrCast(tensorOf(ds.data).data)) + @as(usize, @intCast(ishard)) * ds.nbs_data;
        mod.impl.ggml_c.ggml_backend_tensor_set(data_batch, ptr_data, @as(usize, @intCast(ishard_batch)) * ds.nbs_data, ds.nbs_data);

        if (labels_batch == null) continue;

        const ptr_labels: [*]const u8 = @as([*]const u8, @ptrCast(tensorOf(ds.labels).data)) + @as(usize, @intCast(ishard)) * ds.nbs_labels;
        mod.impl.ggml_c.ggml_backend_tensor_set(labels_batch, ptr_labels, @as(usize, @intCast(ishard_batch)) * ds.nbs_labels, ds.nbs_labels);
    }
}

/// Ports `ggml_opt_dataset_get_batch_host` (ggml-opt.cpp:196 @b1b770d90).
///
/// Parameters:
/// - `dataset`: the dataset.
/// - `data_batch`: host destination for the data.
/// - `nb_data_batch`: its size in bytes, a multiple of the shard size.
/// - `labels_batch`: host destination for the labels; null exactly when the
///   dataset has none.
/// - `ibatch`: which batch, in the current permutation.
///
/// Return: nothing.
pub fn ggml_opt_dataset_get_batch_host(
    dataset: mod.impl.ggml_c.ggml_opt_dataset_t,
    data_batch: ?*anyopaque,
    nb_data_batch: usize,
    labels_batch: ?*anyopaque,
    ibatch: i64,
) callconv(.c) void {
    const ds = datasetOf(dataset);
    mod.impl.assert((labels_batch == null) == (ds.labels == null), "(labels_batch == nullptr) == (dataset->labels == nullptr)");
    mod.impl.assert(nb_data_batch % ds.nbs_data == 0, "nb_data_batch % dataset->nbs_data == 0");

    const shards_per_batch: i64 = @intCast(nb_data_batch / ds.nbs_data);

    mod.impl.assert((ibatch + 1) * shards_per_batch <= @as(i64, @intCast(ds.permutation.len)), "(ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size())");

    var ishard_batch: i64 = 0;
    while (ishard_batch < shards_per_batch) : (ishard_batch += 1) {
        const ishard: usize = @intCast(ds.permutation[@intCast(ibatch * shards_per_batch + ishard_batch)]);
        const ib: usize = @intCast(ishard_batch);

        const ptr_data: [*]const u8 = @as([*]const u8, @ptrCast(tensorOf(ds.data).data)) + ishard * ds.nbs_data;
        const ptr_data_batch: [*]u8 = @as([*]u8, @ptrCast(data_batch)) + ib * ds.nbs_data;
        @memcpy(ptr_data_batch[0..ds.nbs_data], ptr_data[0..ds.nbs_data]);

        if (labels_batch == null) continue;

        const ptr_labels: [*]const u8 = @as([*]const u8, @ptrCast(tensorOf(ds.labels).data)) + ishard * ds.nbs_labels;
        const ptr_labels_batch: [*]u8 = @as([*]u8, @ptrCast(labels_batch)) + ib * ds.nbs_labels;
        @memcpy(ptr_labels_batch[0..ds.nbs_labels], ptr_labels[0..ds.nbs_labels]);
    }
}

// -----------------------------------------------------------------------------
// Model / Context

/// Ports `ggml_opt_get_default_optimizer_params` (ggml-opt.cpp:223 @b1b770d90).
///
/// Parameters:
/// - `userdata`: unused.
///
/// Return: AdamW at 1e-3 with betas 0.9 / 0.999 and eps 1e-8, SGD at 1e-3,
/// no weight decay for either.
pub fn ggml_opt_get_default_optimizer_params(userdata: ?*anyopaque) callconv(.c) mod.impl.ggml_c.struct_ggml_opt_optimizer_params {
    _ = userdata;

    var result: mod.impl.ggml_c.struct_ggml_opt_optimizer_params = undefined;

    result.adamw.alpha = 0.001;
    result.adamw.beta1 = 0.9;
    result.adamw.beta2 = 0.999;
    result.adamw.eps = 1e-8;
    result.adamw.wd = 0.0;

    result.sgd.alpha = 1e-3;
    result.sgd.wd = 0.0;

    return result;
}

/// Ports `ggml_opt_get_constant_optimizer_params` (ggml-opt.cpp:241 @b1b770d90).
///
/// Parameters:
/// - `userdata`: a `ggml_opt_optimizer_params *`.
///
/// Return: a copy of what `userdata` points at.
pub fn ggml_opt_get_constant_optimizer_params(userdata: ?*anyopaque) callconv(.c) mod.impl.ggml_c.struct_ggml_opt_optimizer_params {
    return @as(*const mod.impl.ggml_c.struct_ggml_opt_optimizer_params, @ptrCast(@alignCast(userdata))).*;
}

/// Ports `ggml_opt_default_params` (ggml-opt.cpp:245 @b1b770d90).
///
/// Parameters:
/// - `backend_sched`: the scheduler the graphs are built for.
/// - `loss_type`: the loss to minimize.
///
/// Return: dynamic graphs, a full optimizer build, no gradient accumulation,
/// AdamW with the default parameters.
pub fn ggml_opt_default_params(backend_sched: mod.impl.ggml_c.ggml_backend_sched_t, loss_type: mod.impl.ggml_c.enum_ggml_opt_loss_type) callconv(.c) mod.impl.ggml_c.struct_ggml_opt_params {
    return .{
        .backend_sched = backend_sched,
        .ctx_compute = null,
        .inputs = null,
        .outputs = null,
        .loss_type = loss_type,
        .build_type = mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_OPT,
        .opt_period = 1,
        .get_opt_pars = &ggml_opt_get_default_optimizer_params,
        .get_opt_pars_ud = null,
        .optimizer = mod.impl.ggml_c.GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    };
}

/// Pointer-keyed map, the role `std::map<ggml_tensor *, ggml_tensor *>`
/// plays in `dup_graph`. Its order is never observed — only `find` and
/// insertion — so a hash map is equivalent.
const TensorMap = std.AutoHashMapUnmanaged(*mod.impl.ggml_c.ggml_tensor, *mod.impl.ggml_c.ggml_tensor);

/// Ports `map_tensor` (ggml-opt.cpp:262 @b1b770d90).
///
/// Duplicates `tensor` into `ctx` field for field, sharing its data, and
/// recurses into its view source and sources first-come — the order decides
/// where in `ctx` each duplicate lands, so it is the C++'s exactly.
///
/// Parameters:
/// - `tensor_map`: duplicates made so far.
/// - `ctx`: the context the duplicates are allocated in.
/// - `tensor`: the tensor to duplicate, or null.
///
/// Return: the duplicate, or null for a null `tensor`.
fn mapTensor(tensor_map: *TensorMap, ctx: ?*mod.impl.ggml_c.ggml_context, tensor_c: [*c]mod.impl.ggml_c.ggml_tensor) [*c]mod.impl.ggml_c.ggml_tensor {
    if (tensor_c == null) return null;
    const tensor = tensorOf(tensor_c);

    if (tensor_map.get(tensor)) |t| return t;

    const new_tensor = tensorOf(mod.impl.ggml_c.ggml_dup_tensor(ctx, tensor));
    tensor_map.put(std.heap.c_allocator, tensor, new_tensor) catch oom();

    new_tensor.op = tensor.op;
    for (0..mod.impl.ggml_c.GGML_MAX_DIMS) |i| new_tensor.nb[i] = tensor.nb[i];
    new_tensor.flags = tensor.flags;
    @memcpy(&new_tensor.op_params, &tensor.op_params);
    // strcpy: the bytes up to and including the terminator, no further.
    const len = std.mem.indexOfScalar(u8, &tensor.name, 0) orelse tensor.name.len - 1;
    @memcpy(new_tensor.name[0 .. len + 1], tensor.name[0 .. len + 1]);
    new_tensor.data = tensor.data;
    new_tensor.buffer = tensor.buffer;
    new_tensor.extra = tensor.extra;
    new_tensor.view_offs = tensor.view_offs;
    new_tensor.view_src = mapTensor(tensor_map, ctx, tensor.view_src);
    for (0..mod.impl.ggml_c.GGML_MAX_SRC) |i| new_tensor.src[i] = mapTensor(tensor_map, ctx, tensor.src[i]);

    return new_tensor;
}

/// Ports `dup_graph` (ggml-opt.cpp:293 @b1b770d90).
///
/// Parameters:
/// - `ctx`: the context the copy and its tensors are allocated in.
/// - `src`: the graph to copy.
///
/// Return: a graph over fresh tensors sharing `src`'s data, with `src`'s
/// gradients and accumulators carried over node for node.
fn dupGraph(ctx: ?*mod.impl.ggml_c.ggml_context, src_c: ?*mod.impl.ggml_c.ggml_cgraph) ?*mod.impl.ggml_c.ggml_cgraph {
    var tensor_map: TensorMap = .empty;
    defer tensor_map.deinit(std.heap.c_allocator);

    const src = graphOf(src_c);
    const dst_c = mod.impl.ggml_c.ggml_new_graph_custom(ctx, @intCast(src.size), true);
    const dst = graphOf(dst_c);

    for (0..@intCast(src.n_leafs)) |i| {
        mod.impl.ggml_c.ggml_build_forward_expand(dst_c, mapTensor(&tensor_map, ctx, src.leafs[i]));
    }
    mod.impl.assert(dst.n_leafs == src.n_leafs, "dst->n_leafs == src->n_leafs");
    for (0..@intCast(src.n_nodes)) |i| {
        mod.impl.ggml_c.ggml_build_forward_expand(dst_c, mapTensor(&tensor_map, ctx, src.nodes[i]));
    }
    mod.impl.assert(dst.n_nodes == src.n_nodes, "dst->n_nodes == src->n_nodes");
    for (0..@intCast(src.n_nodes)) |i| {
        const igrad_src = mod.impl.hashFind(&src.visited_hash_set, src.nodes[i].?);
        const igrad_dst = mod.impl.hashFind(&dst.visited_hash_set, dst.nodes[i].?);

        mod.impl.assert(igrad_src != mod.impl.hashset_full, "igrad_src != GGML_HASHSET_FULL");
        mod.impl.assert(mod.impl.bitsetGet(src.visited_hash_set.used, igrad_src), "ggml_bitset_get(src->visited_hash_set.used, igrad_src)");
        mod.impl.assert(igrad_dst != mod.impl.hashset_full, "igrad_dst != GGML_HASHSET_FULL");
        mod.impl.assert(mod.impl.bitsetGet(dst.visited_hash_set.used, igrad_dst), "ggml_bitset_get(dst->visited_hash_set.used, igrad_dst)");

        dst.grads[igrad_dst] = src.grads[igrad_src];
        dst.grad_accs[igrad_dst] = src.grad_accs[igrad_src];
    }

    return dst_c;
}

/// Grows a node-indexed tensor list to `n` entries, as `std::vector::resize`
/// does; new entries are null.
fn resizeList(list: *std.ArrayList([*c]mod.impl.ggml_c.ggml_tensor), n: usize) void {
    const old = list.items.len;
    list.resize(std.heap.c_allocator, n) catch oom();
    if (n > old) @memset(list.items[old..], null);
}

/// Ports `ggml_opt_build` (ggml-opt.cpp:322 @b1b770d90).
///
/// Extends the forward graph with the loss, then — as far as the allocated
/// build type reaches — duplicates it into the gradient graph and that into
/// the optimizer graph, allocating accumulators and momenta statically the
/// first time through.
///
/// Parameters:
/// - `opt_ctx`: the context; its `gf`, `inputs` and `outputs` must be set.
///
/// Return: nothing.
fn optBuild(opt_ctx: *Context) void {
    mod.impl.assert(opt_ctx.ctx_compute != null, "opt_ctx->ctx_compute && \"no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc\"");
    mod.impl.assert(!opt_ctx.static_graphs or tensorOf(opt_ctx.inputs).data != null, "(!opt_ctx->static_graphs || opt_ctx->inputs->data) && \"when using static graphs the inputs must be allocated statically\"");

    const optimizer = opt_ctx.optimizer;

    const accumulate = opt_ctx.build_type_alloc >= mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_GRAD and
        !(opt_ctx.static_graphs and opt_ctx.build_type_alloc == mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_OPT and opt_ctx.opt_period == 1);

    const need_momenta = opt_ctx.build_type_alloc == mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_OPT and
        opt_ctx.optimizer == mod.impl.ggml_c.GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    mod.impl.ggml_c.ggml_set_input(opt_ctx.inputs);
    mod.impl.ggml_c.ggml_set_output(opt_ctx.outputs);

    const gf = graphOf(opt_ctx.gf);
    var n_param: usize = 0;
    for (0..@intCast(gf.n_nodes)) |i| {
        const node = gf.nodes[i].?;
        if ((node.flags & mod.impl.ggml_c.GGML_TENSOR_FLAG_PARAM) != 0) n_param += 1;
        mod.impl.assert((node.flags & mod.impl.ggml_c.GGML_TENSOR_FLAG_LOSS) == 0, "!(node->flags & GGML_TENSOR_FLAG_LOSS) && \"support for extra loss terms not implemented\"");
    }

    if (opt_ctx.ctx_static == null) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        const n_loss: usize = 1;
        const tensors_per_param: usize = @as(usize, if (accumulate) 1 else 0) + @as(usize, if (need_momenta) 2 else 0);
        const tensors_const: usize = if (opt_ctx.static_graphs) 9 else 0;
        const size_meta = (n_loss + tensors_per_param * n_param + tensors_const) * mod.impl.ggml_c.ggml_tensor_overhead();
        const params: mod.impl.ggml_c.ggml_init_params = .{
            .mem_size = size_meta,
            .mem_buffer = null,
            .no_alloc = true,
        };
        opt_ctx.ctx_static = mod.impl.ggml_c.ggml_init(params);
    }
    mod.impl.assert(opt_ctx.build_type <= opt_ctx.build_type_alloc, "opt_ctx->build_type <= opt_ctx->build_type_alloc");

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        const size_meta = 1 * mod.impl.ggml_c.ggml_tensor_overhead();
        const params: mod.impl.ggml_c.ggml_init_params = .{
            .mem_size = size_meta,
            .mem_buffer = null,
            .no_alloc = true,
        };
        mod.impl.ggml_c.ggml_free(opt_ctx.ctx_cpu);
        opt_ctx.ctx_cpu = mod.impl.ggml_c.ggml_init(params);

        mod.impl.ggml_c.ggml_backend_buffer_free(opt_ctx.buf_cpu);
        opt_ctx.buf_cpu = null;
    }

    const ctx_results = if (opt_ctx.static_graphs) opt_ctx.ctx_static else opt_ctx.ctx_compute;

    switch (opt_ctx.loss_type) {
        mod.impl.ggml_c.GGML_OPT_LOSS_TYPE_MEAN => {
            opt_ctx.loss = mod.impl.ggml_c.ggml_sum(ctx_results, opt_ctx.outputs);
            _ = mod.impl.ggml_c.ggml_set_name(opt_ctx.loss, "loss_sum");
            const scale: f32 = 1.0 / @as(f32, @floatFromInt(@as(i64, opt_ctx.opt_period) * mod.impl.ggml_c.ggml_nelements(opt_ctx.outputs)));
            opt_ctx.loss = mod.impl.ggml_c.ggml_scale(ctx_results, opt_ctx.loss, scale);
            _ = mod.impl.ggml_c.ggml_set_name(opt_ctx.loss, "loss_mean");
            opt_ctx.loss_per_datapoint = true;
        },
        mod.impl.ggml_c.GGML_OPT_LOSS_TYPE_SUM => {
            opt_ctx.loss = mod.impl.ggml_c.ggml_sum(ctx_results, opt_ctx.outputs);
            _ = mod.impl.ggml_c.ggml_set_name(opt_ctx.loss, "loss_sum");
            opt_ctx.loss_per_datapoint = false;
        },
        mod.impl.ggml_c.GGML_OPT_LOSS_TYPE_CROSS_ENTROPY => {
            opt_ctx.labels = mod.impl.ggml_c.ggml_dup_tensor(ctx_results, opt_ctx.outputs);
            mod.impl.ggml_c.ggml_set_input(opt_ctx.labels);
            _ = mod.impl.ggml_c.ggml_set_name(opt_ctx.labels, "labels");
            opt_ctx.loss = mod.impl.ggml_c.ggml_cross_entropy_loss(ctx_results, opt_ctx.outputs, opt_ctx.labels);
            _ = mod.impl.ggml_c.ggml_set_name(opt_ctx.loss, "loss_cross_entropy");
            if (opt_ctx.opt_period > 1) {
                opt_ctx.loss = mod.impl.ggml_c.ggml_scale(ctx_results, opt_ctx.loss, 1.0 / @as(f32, @floatFromInt(opt_ctx.opt_period)));
                _ = mod.impl.ggml_c.ggml_set_name(opt_ctx.loss, "loss_cross_entropy_scaled");
            }
            opt_ctx.loss_per_datapoint = true;
        },
        mod.impl.ggml_c.GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR => {
            opt_ctx.labels = mod.impl.ggml_c.ggml_dup_tensor(ctx_results, opt_ctx.outputs);
            mod.impl.ggml_c.ggml_set_input(opt_ctx.labels);
            _ = mod.impl.ggml_c.ggml_set_name(opt_ctx.labels, "labels");
            opt_ctx.loss = mod.impl.ggml_c.ggml_sub(ctx_results, opt_ctx.outputs, opt_ctx.labels);
            _ = mod.impl.ggml_c.ggml_set_name(opt_ctx.loss, "loss_error");
            opt_ctx.loss = mod.impl.ggml_c.ggml_sqr(ctx_results, opt_ctx.loss);
            _ = mod.impl.ggml_c.ggml_set_name(opt_ctx.loss, "loss_squared_error");
            opt_ctx.loss = mod.impl.ggml_c.ggml_sum(ctx_results, opt_ctx.loss);
            _ = mod.impl.ggml_c.ggml_set_name(opt_ctx.loss, "loss_sum_squared_error");
            const scale: f32 = 1.0 / @as(f32, @floatFromInt(@as(i64, opt_ctx.opt_period) * mod.impl.ggml_c.ggml_nelements(opt_ctx.outputs)));
            opt_ctx.loss = mod.impl.ggml_c.ggml_scale(ctx_results, opt_ctx.loss, scale);
            _ = mod.impl.ggml_c.ggml_set_name(opt_ctx.loss, "loss_mean_squared_error");
            opt_ctx.loss_per_datapoint = true;
        },
        else => {},
    }
    mod.impl.ggml_c.ggml_set_output(opt_ctx.loss);
    mod.impl.ggml_c.ggml_set_loss(opt_ctx.loss);
    mod.impl.ggml_c.ggml_build_forward_expand(opt_ctx.gf, opt_ctx.loss);

    if (opt_ctx.loss_type == mod.impl.ggml_c.GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx.pred = mod.impl.ggml_c.ggml_argmax(ctx_results, opt_ctx.outputs);
        _ = mod.impl.ggml_c.ggml_set_name(opt_ctx.pred, "pred");
        mod.impl.ggml_c.ggml_set_output(opt_ctx.pred);
        mod.impl.ggml_c.ggml_build_forward_expand(opt_ctx.gf, opt_ctx.pred);

        opt_ctx.ncorrect = mod.impl.ggml_c.ggml_count_equal(ctx_results, opt_ctx.pred, mod.impl.ggml_c.ggml_argmax(ctx_results, opt_ctx.labels));
        _ = mod.impl.ggml_c.ggml_set_name(opt_ctx.ncorrect, "ncorrect");
        mod.impl.ggml_c.ggml_set_output(opt_ctx.ncorrect);
        mod.impl.ggml_c.ggml_build_forward_expand(opt_ctx.gf, opt_ctx.ncorrect);
    }

    if (opt_ctx.buf_static != null) {
        if (opt_ctx.build_type == mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_FORWARD) return;
    } else if (opt_ctx.build_type_alloc == mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx.buf_static = mod.impl.ggml_c.ggml_backend_alloc_ctx_tensors(opt_ctx.ctx_static, mod.impl.ggml_c.ggml_backend_sched_get_backend(opt_ctx.backend_sched, 0));
        return;
    }

    if (opt_ctx.grad_accs.items.len == 0) {
        mod.impl.assert(opt_ctx.build_type_alloc >= mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_GRAD, "opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD");

        const n_nodes: usize = @intCast(gf.n_nodes);
        resizeList(&opt_ctx.grad_accs, n_nodes);
        for (0..n_nodes) |i| {
            const node = gf.nodes[i].?;
            if ((accumulate and (node.flags & mod.impl.ggml_c.GGML_TENSOR_FLAG_PARAM) != 0) or (node.flags & mod.impl.ggml_c.GGML_TENSOR_FLAG_LOSS) != 0) {
                opt_ctx.grad_accs.items[i] = mod.impl.ggml_c.ggml_new_tensor(opt_ctx.ctx_static, mod.impl.ggml_c.GGML_TYPE_F32, mod.impl.ggml_c.GGML_MAX_DIMS, &node.ne);
            } else {
                opt_ctx.grad_accs.items[i] = null;
            }
        }

        if (need_momenta and opt_ctx.build_type_alloc >= mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_OPT) {
            resizeList(&opt_ctx.grad_m, n_nodes);
            resizeList(&opt_ctx.grad_v, n_nodes);
            for (0..n_nodes) |i| {
                const node = gf.nodes[i].?;
                if ((node.flags & mod.impl.ggml_c.GGML_TENSOR_FLAG_PARAM) != 0) {
                    opt_ctx.grad_m.items[i] = mod.impl.ggml_c.ggml_new_tensor(opt_ctx.ctx_static, mod.impl.ggml_c.GGML_TYPE_F32, mod.impl.ggml_c.GGML_MAX_DIMS, &node.ne);
                    opt_ctx.grad_v.items[i] = mod.impl.ggml_c.ggml_new_tensor(opt_ctx.ctx_static, mod.impl.ggml_c.GGML_TYPE_F32, mod.impl.ggml_c.GGML_MAX_DIMS, &node.ne);
                } else {
                    opt_ctx.grad_m.items[i] = null;
                    opt_ctx.grad_v.items[i] = null;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx.gb_grad = mod.impl.ggml_c.ggml_graph_dup(opt_ctx.ctx_compute, opt_ctx.gf, true);
    mod.impl.ggml_c.ggml_build_backward_expand(opt_ctx.ctx_compute, opt_ctx.gb_grad, opt_ctx.grad_accs.items.ptr);

    if (opt_ctx.buf_static != null) {
        if (opt_ctx.build_type == mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_GRAD) return;
    } else if (opt_ctx.build_type_alloc == mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx.buf_static = mod.impl.ggml_c.ggml_backend_alloc_ctx_tensors(opt_ctx.ctx_static, mod.impl.ggml_c.ggml_backend_sched_get_backend(opt_ctx.backend_sched, 0));
        mod.impl.ggml_c.ggml_graph_reset(opt_ctx.gb_grad);
    }

    mod.impl.assert(opt_ctx.build_type_alloc == mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_OPT, "opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT");

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx.gb_opt = mod.impl.ggml_c.ggml_graph_dup(opt_ctx.ctx_compute, opt_ctx.gb_grad, true);

    opt_ctx.opt_step_params = mod.impl.ggml_c.ggml_new_tensor_1d(opt_ctx.ctx_cpu, mod.impl.ggml_c.GGML_TYPE_F32, if (need_momenta) 7 else 2);
    const adamw_params = opt_ctx.opt_step_params;
    mod.impl.ggml_c.ggml_set_input(adamw_params);
    const optimizer_name = ggml_opt_optimizer_name(opt_ctx.optimizer);
    _ = mod.impl.ggml_c.ggml_format_name(adamw_params, "%s_params", optimizer_name);
    const gb_opt = graphOf(opt_ctx.gb_opt);
    var i: isize = gf.n_nodes - 1;
    while (i >= 0) : (i -= 1) {
        const iu: usize = @intCast(i);
        const node = gb_opt.nodes[iu];
        const grad = mod.impl.ggml_c.ggml_graph_get_grad(opt_ctx.gb_opt, node);

        if (grad != null and (node.?.flags & mod.impl.ggml_c.GGML_TENSOR_FLAG_PARAM) != 0) {
            var m: [*c]mod.impl.ggml_c.ggml_tensor = null;
            var v: [*c]mod.impl.ggml_c.ggml_tensor = null;
            if (need_momenta) {
                m = opt_ctx.grad_m.items[iu];
                v = opt_ctx.grad_v.items[iu];
                _ = mod.impl.ggml_c.ggml_format_name(m, "AdamW m for %s", @as([*:0]const u8, @ptrCast(&node.?.name)));
                _ = mod.impl.ggml_c.ggml_format_name(v, "AdamW v for %s", @as([*:0]const u8, @ptrCast(&node.?.name)));
            }
            const opt_step: [*c]mod.impl.ggml_c.ggml_tensor = switch (optimizer) {
                mod.impl.ggml_c.GGML_OPT_OPTIMIZER_TYPE_ADAMW => mod.impl.ggml_c.ggml_opt_step_adamw(opt_ctx.ctx_compute, node, grad, m, v, adamw_params),
                mod.impl.ggml_c.GGML_OPT_OPTIMIZER_TYPE_SGD => mod.impl.ggml_c.ggml_opt_step_sgd(opt_ctx.ctx_compute, node, grad, adamw_params),
                else => mod.impl.abort("fatal error"),
            };
            _ = mod.impl.ggml_c.ggml_format_name(opt_step, "%s step for %s", optimizer_name, @as([*:0]const u8, @ptrCast(&node.?.name)));
            mod.impl.ggml_c.ggml_build_forward_expand(opt_ctx.gb_opt, opt_step);
        }
    }

    if (opt_ctx.buf_static == null) {
        opt_ctx.buf_static = mod.impl.ggml_c.ggml_backend_alloc_ctx_tensors(opt_ctx.ctx_static, mod.impl.ggml_c.ggml_backend_sched_get_backend(opt_ctx.backend_sched, 0));
        mod.impl.ggml_c.ggml_graph_reset(opt_ctx.gb_opt);
    }

    opt_ctx.buf_cpu = mod.impl.ggml_c.ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx.ctx_cpu, mod.impl.ggml_c.ggml_backend_cpu_buffer_type());
}

/// Ports `ggml_opt_init` (ggml-opt.cpp:549 @b1b770d90).
///
/// Parameters:
/// - `params`: the context's configuration. With `ctx_compute`, `inputs` and
///   `outputs` set the graphs are static and built here; otherwise all three
///   must be unset and each step supplies its graph through
///   `ggml_opt_prepare_alloc`.
///
/// Return: a new context; free it with `ggml_opt_free`.
pub fn ggml_opt_init(params: mod.impl.ggml_c.struct_ggml_opt_params) callconv(.c) mod.impl.ggml_c.ggml_opt_context_t {
    const result = create(Context);
    result.rng = stdlib.libcxx.Mt19937.init(stdlib.libcxx.Mt19937.default_seed);
    result.backend_sched = params.backend_sched;
    result.ctx_compute = params.ctx_compute;
    result.loss_type = params.loss_type;
    result.build_type = params.build_type;
    result.build_type_alloc = params.build_type;
    result.inputs = params.inputs;
    result.outputs = params.outputs;
    result.opt_period = params.opt_period;
    result.get_opt_pars = params.get_opt_pars;
    result.get_opt_pars_ud = params.get_opt_pars_ud;
    result.optimizer = params.optimizer;

    mod.impl.assert(result.opt_period >= 1, "result->opt_period >= 1");

    result.static_graphs = result.ctx_compute != null;

    if (!result.static_graphs) {
        mod.impl.assert(result.inputs == null, "!result->inputs");
        mod.impl.assert(result.outputs == null, "!result->outputs");
        return @ptrCast(result);
    }

    mod.impl.assert(result.inputs != null, "result->inputs");
    mod.impl.assert(result.outputs != null, "result->outputs");

    result.gf = mod.impl.ggml_c.ggml_new_graph_custom(result.ctx_compute, mod.impl.ggml_c.GGML_DEFAULT_GRAPH_SIZE, true); // Forward pass.
    mod.impl.ggml_c.ggml_build_forward_expand(result.gf, result.outputs);

    optBuild(result);

    return @ptrCast(result);
}

/// Ports `ggml_opt_free` (ggml-opt.cpp:584 @b1b770d90).
///
/// Parameters:
/// - `opt_ctx`: the context, or null; invalid afterwards.
///
/// Return: nothing.
pub fn ggml_opt_free(opt_ctx: mod.impl.ggml_c.ggml_opt_context_t) callconv(.c) void {
    if (opt_ctx == null) return;
    const ctx = contextOf(opt_ctx);
    mod.impl.ggml_c.ggml_backend_buffer_free(ctx.buf_static);
    mod.impl.ggml_c.ggml_backend_buffer_free(ctx.buf_cpu);
    mod.impl.ggml_c.ggml_free(ctx.ctx_static);
    mod.impl.ggml_c.ggml_free(ctx.ctx_cpu);
    mod.impl.ggml_c.ggml_free(ctx.ctx_copy);
    ctx.grad_accs.deinit(std.heap.c_allocator);
    ctx.grad_m.deinit(std.heap.c_allocator);
    ctx.grad_v.deinit(std.heap.c_allocator);
    std.heap.c_allocator.destroy(ctx);
}

/// Ports `ggml_opt_reset` (ggml-opt.cpp:596 @b1b770d90).
///
/// Parameters:
/// - `opt_ctx`: the context.
/// - `optimizer`: also zero the momenta and restart the iteration count.
///
/// Return: nothing.
pub fn ggml_opt_reset(opt_ctx: mod.impl.ggml_c.ggml_opt_context_t, optimizer: bool) callconv(.c) void {
    const ctx = contextOf(opt_ctx);
    if (optimizer) {
        mod.impl.ggml_c.ggml_graph_reset(ctx.gb_opt);
        ctx.iter = 1;
    } else {
        mod.impl.ggml_c.ggml_graph_reset(ctx.gb_grad);
    }
}

/// Ports `ggml_opt_static_graphs` (ggml-opt.cpp:605 @b1b770d90).
///
/// Parameters:
/// - `opt_ctx`: the context.
///
/// Return: whether the graphs are allocated statically.
pub fn ggml_opt_static_graphs(opt_ctx: mod.impl.ggml_c.ggml_opt_context_t) callconv(.c) bool {
    return contextOf(opt_ctx).static_graphs;
}

/// Ports `ggml_opt_inputs` (ggml-opt.cpp:609 @b1b770d90).
///
/// Parameters:
/// - `opt_ctx`: the context.
///
/// Return: the forward graph's input tensor; with dynamic graphs, valid
/// until the next `ggml_opt_alloc`.
pub fn ggml_opt_inputs(opt_ctx: mod.impl.ggml_c.ggml_opt_context_t) callconv(.c) [*c]mod.impl.ggml_c.ggml_tensor {
    return contextOf(opt_ctx).inputs;
}

/// Ports `ggml_opt_outputs` (ggml-opt.cpp:613 @b1b770d90).
///
/// Parameters:
/// - `opt_ctx`: the context.
///
/// Return: the forward graph's output tensor; with dynamic graphs, valid
/// until the next `ggml_opt_alloc`.
pub fn ggml_opt_outputs(opt_ctx: mod.impl.ggml_c.ggml_opt_context_t) callconv(.c) [*c]mod.impl.ggml_c.ggml_tensor {
    return contextOf(opt_ctx).outputs;
}

/// Ports `ggml_opt_labels` (ggml-opt.cpp:617 @b1b770d90).
///
/// Parameters:
/// - `opt_ctx`: the context.
///
/// Return: the labels the outputs are compared against, or null for a loss
/// that has none.
pub fn ggml_opt_labels(opt_ctx: mod.impl.ggml_c.ggml_opt_context_t) callconv(.c) [*c]mod.impl.ggml_c.ggml_tensor {
    return contextOf(opt_ctx).labels;
}

/// Ports `ggml_opt_loss` (ggml-opt.cpp:621 @b1b770d90).
///
/// Parameters:
/// - `opt_ctx`: the context.
///
/// Return: the scalar loss tensor.
pub fn ggml_opt_loss(opt_ctx: mod.impl.ggml_c.ggml_opt_context_t) callconv(.c) [*c]mod.impl.ggml_c.ggml_tensor {
    return contextOf(opt_ctx).loss;
}

/// Ports `ggml_opt_pred` (ggml-opt.cpp:625 @b1b770d90).
///
/// Parameters:
/// - `opt_ctx`: the context.
///
/// Return: the predictions, or null unless the loss is cross-entropy.
pub fn ggml_opt_pred(opt_ctx: mod.impl.ggml_c.ggml_opt_context_t) callconv(.c) [*c]mod.impl.ggml_c.ggml_tensor {
    return contextOf(opt_ctx).pred;
}

/// Ports `ggml_opt_ncorrect` (ggml-opt.cpp:629 @b1b770d90).
///
/// Parameters:
/// - `opt_ctx`: the context.
///
/// Return: the count of correct predictions, or null unless the loss is
/// cross-entropy.
pub fn ggml_opt_ncorrect(opt_ctx: mod.impl.ggml_c.ggml_opt_context_t) callconv(.c) [*c]mod.impl.ggml_c.ggml_tensor {
    return contextOf(opt_ctx).ncorrect;
}

/// Ports `ggml_opt_grad_acc` (ggml-opt.cpp:633 @b1b770d90).
///
/// Parameters:
/// - `opt_ctx`: the context.
/// - `node`: a node of the forward graph.
///
/// Return: its gradient accumulator in the optimizer graph, or null.
pub fn ggml_opt_grad_acc(opt_ctx: mod.impl.ggml_c.ggml_opt_context_t, node: [*c]mod.impl.ggml_c.ggml_tensor) callconv(.c) [*c]mod.impl.ggml_c.ggml_tensor {
    return mod.impl.ggml_c.ggml_graph_get_grad_acc(contextOf(opt_ctx).gb_opt, node);
}

// -----------------------------------------------------------------------------
// Optimization Result

/// Ports `ggml_opt_result_init` (ggml-opt.cpp:639 @b1b770d90).
///
/// Return: an empty result; free it with `ggml_opt_result_free`.
pub fn ggml_opt_result_init() callconv(.c) mod.impl.ggml_c.ggml_opt_result_t {
    return @ptrCast(create(Result));
}

/// Ports `ggml_opt_result_free` (ggml-opt.cpp:643 @b1b770d90).
///
/// Parameters:
/// - `result`: the result, or null; invalid afterwards.
///
/// Return: nothing.
pub fn ggml_opt_result_free(result: mod.impl.ggml_c.ggml_opt_result_t) callconv(.c) void {
    if (result == null) return; // `delete nullptr` is a no-op
    const r = resultOf(result);
    r.loss.deinit(std.heap.c_allocator);
    r.pred.deinit(std.heap.c_allocator);
    std.heap.c_allocator.destroy(r);
}

/// Ports `ggml_opt_result_reset` (ggml-opt.cpp:647 @b1b770d90).
///
/// `opt_period` and `loss_per_datapoint` are left as they were; the next
/// `ggml_opt_eval` overwrites them because `ndata` is 0.
///
/// Parameters:
/// - `result`: the result.
///
/// Return: nothing.
pub fn ggml_opt_result_reset(result: mod.impl.ggml_c.ggml_opt_result_t) callconv(.c) void {
    const r = resultOf(result);
    r.ndata = 0;
    r.loss.clearRetainingCapacity();
    r.pred.clearRetainingCapacity();
    r.ncorrect = 0;
}

/// Ports `ggml_opt_result_ndata` (ggml-opt.cpp:654 @b1b770d90).
///
/// Parameters:
/// - `result`: the result.
/// - `ndata`: receives the number of datapoints.
///
/// Return: nothing.
pub fn ggml_opt_result_ndata(result: mod.impl.ggml_c.ggml_opt_result_t, ndata: [*c]i64) callconv(.c) void {
    ndata.* = resultOf(result).ndata;
}

/// Ports `ggml_opt_result_loss` (ggml-opt.cpp:658 @b1b770d90).
///
/// `loss_scaled*loss_scaled` is a `float` product, rounded to `float` before
/// it is widened; `sum_squared/nbatches - mean*mean` is one fused
/// multiply-subtract in the reference object, measured.
///
/// Parameters:
/// - `result`: the result.
/// - `loss`: receives the mean loss per physical batch, or the sum.
/// - `unc`: receives its uncertainty, or null to skip.
///
/// Return: nothing.
pub fn ggml_opt_result_loss(result: mod.impl.ggml_c.ggml_opt_result_t, loss: [*c]f64, unc: [*c]f64) callconv(.c) void {
    const r = resultOf(result);
    const nbatches: i64 = @intCast(r.loss.items.len); // Number of physical batches.

    if (nbatches == 0) {
        loss.* = 0.0;
        unc.* = std.math.nan(f64);
        return;
    }

    var sum: f64 = 0.0;
    var sum_squared: f64 = 0.0;

    for (r.loss.items) |l| {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        const loss_scaled: f32 = if (r.loss_per_datapoint) l * @as(f32, @floatFromInt(r.opt_period)) else l;
        sum += loss_scaled;
        sum_squared += loss_scaled * loss_scaled;
    }

    const nb: f64 = @floatFromInt(nbatches);
    const mean = sum / nb;
    loss.* = if (r.loss_per_datapoint) mean else sum;

    if (unc == null) return;

    if (nbatches < 2) {
        unc.* = std.math.nan(f64);
        return;
    }

    // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    const var_sum = @mulAdd(f64, -mean, mean, sum_squared / nb);
    const nm1: f64 = @floatFromInt(nbatches - 1);
    unc.* = if (r.loss_per_datapoint) @sqrt(var_sum / nm1) else @sqrt(var_sum * nb / nm1);
}

/// Ports `ggml_opt_result_pred` (ggml-opt.cpp:693 @b1b770d90).
///
/// Parameters:
/// - `result`: the result.
/// - `pred`: receives one prediction per datapoint.
///
/// Return: nothing.
pub fn ggml_opt_result_pred(result: mod.impl.ggml_c.ggml_opt_result_t, pred: [*c]i32) callconv(.c) void {
    for (resultOf(result).pred.items, 0..) |p, i| pred[i] = p;
}

/// Ports `ggml_opt_result_accuracy` (ggml-opt.cpp:699 @b1b770d90).
///
/// Parameters:
/// - `result`: the result.
/// - `accuracy`: receives the fraction of correct predictions, or NaN.
/// - `unc`: receives its uncertainty, or null to skip.
///
/// Return: nothing.
pub fn ggml_opt_result_accuracy(result: mod.impl.ggml_c.ggml_opt_result_t, accuracy: [*c]f64, unc: [*c]f64) callconv(.c) void {
    const r = resultOf(result);
    accuracy.* = if (r.ncorrect >= 0) @as(f64, @floatFromInt(r.ncorrect)) / @as(f64, @floatFromInt(r.ndata)) else std.math.nan(f64);

    if (unc == null) return;

    unc.* = if (r.ncorrect >= 0 and r.ndata >= 2)
        @sqrt(accuracy.* * (1.0 - accuracy.*) / @as(f64, @floatFromInt(r.ndata - 1)))
    else
        std.math.nan(f64);
}

// -----------------------------------------------------------------------------
// Computation

/// Ports `ggml_opt_prepare_alloc` (ggml-opt.cpp:712 @b1b770d90).
///
/// Parameters:
/// - `opt_ctx`: a context with dynamic graphs.
/// - `ctx_compute`: the context the step's graphs are built in.
/// - `gf`: the forward graph.
/// - `inputs`: its input tensor.
/// - `outputs`: its output tensor.
///
/// Return: nothing.
pub fn ggml_opt_prepare_alloc(
    opt_ctx: mod.impl.ggml_c.ggml_opt_context_t,
    ctx_compute: ?*mod.impl.ggml_c.ggml_context,
    gf: ?*mod.impl.ggml_c.ggml_cgraph,
    inputs: [*c]mod.impl.ggml_c.ggml_tensor,
    outputs: [*c]mod.impl.ggml_c.ggml_tensor,
) callconv(.c) void {
    const ctx = contextOf(opt_ctx);
    mod.impl.assert(!ctx.static_graphs, "!opt_ctx->static_graphs");
    ctx.ctx_compute = ctx_compute;
    ctx.gf = gf;
    ctx.inputs = inputs;
    ctx.outputs = outputs;
}

/// Ports `ggml_opt_alloc` (ggml-opt.cpp:725 @b1b770d90).
///
/// Parameters:
/// - `opt_ctx`: the context.
/// - `backward`: allocate for a gradient (or optimizer) step rather than a
///   forward pass.
///
/// Return: nothing.
pub fn ggml_opt_alloc(opt_ctx: mod.impl.ggml_c.ggml_opt_context_t, backward: bool) callconv(.c) void {
    const ctx = contextOf(opt_ctx);
    mod.impl.assert(!ctx.eval_ready, "!opt_ctx->eval_ready");
    if (ctx.build_type == mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_OPT and ctx.opt_period > 1 and ctx.opt_i == 0) {
        mod.impl.ggml_c.ggml_graph_reset(ctx.gb_grad);
    }
    if (backward) {
        const opt_i_next = @mod(ctx.opt_i + 1, ctx.opt_period);
        ctx.build_type = if (opt_i_next == 0) mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_OPT else mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_GRAD;
    } else {
        ctx.build_type = mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_FORWARD;
    }

    if (!ctx.static_graphs) optBuild(ctx);

    const graph: ?*mod.impl.ggml_c.ggml_cgraph = switch (ctx.build_type) {
        mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_FORWARD => ctx.gf,
        mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_GRAD => ctx.gb_grad,
        mod.impl.ggml_c.GGML_OPT_BUILD_TYPE_OPT => ctx.gb_opt,
        else => null,
    };
    mod.impl.assert(graph != null, "graph");

    if (ctx.allocated_graph == graph) {
        ctx.eval_ready = true;
        return;
    }

    mod.impl.ggml_c.ggml_backend_sched_reset(ctx.backend_sched); // clear allocation of previous graph

    if (ctx.static_graphs) {
        const g = graphOf(graph);
        const size: usize = @intCast(g.size);
        const params: mod.impl.ggml_c.ggml_init_params = .{
            .mem_size = size * mod.impl.ggml_c.ggml_tensor_overhead() + mod.impl.ggml_c.ggml_graph_overhead_custom(size, g.grads != null),
            .mem_buffer = null,
            .no_alloc = true,
        };
        mod.impl.ggml_c.ggml_free(ctx.ctx_copy);
        ctx.ctx_copy = mod.impl.ggml_c.ggml_init(params);

        ctx.allocated_graph_copy = dupGraph(ctx.ctx_copy, graph);
    } else {
        ctx.allocated_graph_copy = graph;
    }

    _ = mod.impl.ggml_c.ggml_backend_sched_alloc_graph(ctx.backend_sched, ctx.allocated_graph_copy);
    ctx.allocated_graph = graph;

    ctx.eval_ready = true;
}

/// Ports `ggml_opt_eval` (ggml-opt.cpp:782 @b1b770d90).
///
/// `libm.powf(beta, iter)` converts the 64-bit iteration count to `float` and
/// calls libm's `powf`, as the reference object does — two calls, measured.
///
/// Parameters:
/// - `opt_ctx`: the context, after `ggml_opt_alloc`.
/// - `result`: accumulates this step's loss, predictions and correct count;
///   null to skip.
///
/// Return: nothing.
pub fn ggml_opt_eval(opt_ctx: mod.impl.ggml_c.ggml_opt_context_t, result: mod.impl.ggml_c.ggml_opt_result_t) callconv(.c) void {
    const ctx = contextOf(opt_ctx);
    mod.impl.assert(ctx.eval_ready, "opt_ctx->eval_ready");
    if (ctx.allocated_graph == ctx.gb_opt) {
        const opt_pars = ctx.get_opt_pars.?(ctx.get_opt_pars_ud);

        switch (ctx.optimizer) {
            mod.impl.ggml_c.GGML_OPT_OPTIMIZER_TYPE_ADAMW => {
                mod.impl.assert(opt_pars.adamw.alpha > 0.0, "opt_pars.adamw.alpha > 0.0f");
                mod.impl.assert(opt_pars.adamw.beta1 >= 0.0, "opt_pars.adamw.beta1 >= 0.0f");
                mod.impl.assert(opt_pars.adamw.beta1 <= 1.0, "opt_pars.adamw.beta1 <= 1.0f");
                mod.impl.assert(opt_pars.adamw.beta2 >= 0.0, "opt_pars.adamw.beta2 >= 0.0f");
                mod.impl.assert(opt_pars.adamw.beta2 <= 1.0, "opt_pars.adamw.beta2 <= 1.0f");
                mod.impl.assert(opt_pars.adamw.eps >= 0.0, "opt_pars.adamw.eps >= 0.0f");
                mod.impl.assert(opt_pars.adamw.wd >= 0.0, "opt_pars.adamw.wd >= 0.0f");
                mod.impl.assert(opt_pars.adamw.wd <= 1.0, "opt_pars.adamw.wd <= 1.0f");

                // beta1, beta2 after applying warmup
                const iter_f: f32 = @floatFromInt(ctx.iter);
                const beta1h: f32 = 1.0 / (1.0 - stdlib.libm.powf(opt_pars.adamw.beta1, iter_f));
                const beta2h: f32 = 1.0 / (1.0 - stdlib.libm.powf(opt_pars.adamw.beta2, iter_f));

                const adamw_par_data: [*]f32 = @ptrCast(mod.impl.ggml_c.ggml_get_data_f32(ctx.opt_step_params));
                adamw_par_data[0] = opt_pars.adamw.alpha;
                adamw_par_data[1] = opt_pars.adamw.beta1;
                adamw_par_data[2] = opt_pars.adamw.beta2;
                adamw_par_data[3] = opt_pars.adamw.eps;
                adamw_par_data[4] = opt_pars.adamw.wd;
                adamw_par_data[5] = beta1h;
                adamw_par_data[6] = beta2h;
            },
            mod.impl.ggml_c.GGML_OPT_OPTIMIZER_TYPE_SGD => {
                mod.impl.assert(opt_pars.sgd.alpha > 0.0, "opt_pars.sgd.alpha > 0.0f");
                mod.impl.assert(opt_pars.sgd.wd >= 0.0, "opt_pars.sgd.wd >= 0.0f");
                mod.impl.assert(opt_pars.sgd.wd <= 1.0, "opt_pars.sgd.wd <= 1.0f");
                const sgd: [*]f32 = @ptrCast(mod.impl.ggml_c.ggml_get_data_f32(ctx.opt_step_params));
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            },
            else => mod.impl.abort("fatal error"),
        }
    }

    _ = mod.impl.ggml_c.ggml_backend_sched_graph_compute(ctx.backend_sched, ctx.allocated_graph_copy);
    ctx.iter += @intFromBool(ctx.allocated_graph == ctx.gb_opt);
    ctx.opt_i = @mod(ctx.opt_i + 1, ctx.opt_period);

    if (!ctx.static_graphs) {
        ctx.gf = null;
        ctx.gb_grad = null;
        ctx.gb_opt = null;
        ctx.allocated_graph = null;
        ctx.allocated_graph_copy = null;
    }

    ctx.eval_ready = false;

    if (result == null) return;
    const r = resultOf(result);

    if (r.ndata == 0) {
        r.loss_per_datapoint = ctx.loss_per_datapoint;
        r.opt_period = ctx.opt_period;
    } else {
        mod.impl.assert(r.loss_per_datapoint == ctx.loss_per_datapoint, "result->loss_per_datapoint == opt_ctx->loss_per_datapoint");
        mod.impl.assert(r.opt_period == ctx.opt_period, "result->opt_period         == opt_ctx->opt_period");
    }

    const ndata = tensorOf(ctx.outputs).ne[1];
    mod.impl.assert(r.ndata == ndata *% @as(i64, @intCast(r.loss.items.len)), "result->ndata == ndata*int64_t(result->loss.size()) && \"varying batch size not supported\"");
    r.ndata += ndata;

    mod.impl.assert(mod.impl.ggml_c.ggml_is_scalar(ctx.loss), "ggml_is_scalar(opt_ctx->loss)");
    mod.impl.assert(tensorOf(ctx.loss).type == mod.impl.ggml_c.GGML_TYPE_F32, "opt_ctx->loss->type == GGML_TYPE_F32");
    var loss: f32 = undefined;
    mod.impl.ggml_c.ggml_backend_tensor_get(ctx.loss, &loss, 0, mod.impl.ggml_c.ggml_nbytes(ctx.loss));
    r.loss.append(std.heap.c_allocator, loss) catch oom();

    if (ctx.pred != null) {
        mod.impl.assert(tensorOf(ctx.pred).type == mod.impl.ggml_c.GGML_TYPE_I32, "opt_ctx->pred->type == GGML_TYPE_I32");
        const pred = std.heap.c_allocator.alloc(i32, @intCast(ndata)) catch oom();
        defer std.heap.c_allocator.free(pred);
        mod.impl.ggml_c.ggml_backend_tensor_get(ctx.pred, pred.ptr, 0, mod.impl.ggml_c.ggml_nbytes(ctx.pred));
        r.pred.appendSlice(std.heap.c_allocator, pred) catch oom();
    }

    if (ctx.ncorrect == null or r.ncorrect < 0) {
        r.ncorrect = -1;
        return;
    }

    mod.impl.assert(mod.impl.ggml_c.ggml_is_scalar(ctx.ncorrect), "ggml_is_scalar(opt_ctx->ncorrect)");
    mod.impl.assert(tensorOf(ctx.ncorrect).type == mod.impl.ggml_c.GGML_TYPE_I64, "opt_ctx->ncorrect->type == GGML_TYPE_I64");
    var ncorrect: i64 = undefined;
    mod.impl.ggml_c.ggml_backend_tensor_get(ctx.ncorrect, &ncorrect, 0, mod.impl.ggml_c.ggml_nbytes(ctx.ncorrect));
    r.ncorrect += ncorrect;
}

// -----------------------------------------------------------------------------
// High-Level Functions

/// Ports `ggml_opt_epoch` (ggml-opt.cpp:881 @b1b770d90).
///
/// Parameters:
/// - `opt_ctx`: a context with static graphs.
/// - `dataset`: the data, in its current permutation.
/// - `result_train`: accumulates the training batches; null to skip.
/// - `result_eval`: accumulates the evaluation batches; null to skip.
/// - `idata_split`: datapoints before this train, after it evaluate;
///   negative trains on all of them.
/// - `callback_train`: called after each training batch; may be null.
/// - `callback_eval`: called after each evaluation batch; may be null.
///
/// Return: nothing.
pub fn ggml_opt_epoch(
    opt_ctx: mod.impl.ggml_c.ggml_opt_context_t,
    dataset: mod.impl.ggml_c.ggml_opt_dataset_t,
    result_train: mod.impl.ggml_c.ggml_opt_result_t,
    result_eval: mod.impl.ggml_c.ggml_opt_result_t,
    idata_split_in: i64,
    callback_train: mod.impl.ggml_c.ggml_opt_epoch_callback,
    callback_eval: mod.impl.ggml_c.ggml_opt_epoch_callback,
) callconv(.c) void {
    mod.impl.assert(ggml_opt_static_graphs(opt_ctx), "ggml_opt_static_graphs(opt_ctx) && \"ggml_opt_epoch requires static graphs\"");
    const inputs = ggml_opt_inputs(opt_ctx);
    const labels = ggml_opt_labels(opt_ctx);
    const data = ggml_opt_dataset_data(dataset);
    mod.impl.assert(tensorOf(data).ne[0] == tensorOf(inputs).ne[0], "data->ne[0] == inputs->ne[0]");

    const ndata = tensorOf(data).ne[1];
    const ndata_batch = tensorOf(inputs).ne[1];

    mod.impl.assert(@rem(tensorOf(data).ne[1], tensorOf(inputs).ne[1]) == 0, "data->ne[1] % inputs->ne[1] == 0");
    const nbatches = @divTrunc(ndata, ndata_batch);

    const idata_split = if (idata_split_in < 0) ndata else idata_split_in;
    mod.impl.assert(@rem(idata_split, ndata_batch) == 0, "idata_split % ndata_batch == 0");
    const ibatch_split = @divTrunc(idata_split, ndata_batch);

    var ibatch: i64 = 0;
    var t_loop_start = mod.impl.ggml_c.ggml_time_us();
    while (ibatch < ibatch_split) : (ibatch += 1) {
        ggml_opt_alloc(opt_ctx, true);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_train);
        if (callback_train) |cb| {
            cb(true, opt_ctx, dataset, result_train, ibatch + 1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = mod.impl.ggml_c.ggml_time_us();
    while (ibatch < nbatches) : (ibatch += 1) {
        ggml_opt_alloc(opt_ctx, false);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_eval);
        if (callback_eval) |cb| {
            cb(false, opt_ctx, dataset, result_eval, ibatch + 1 - ibatch_split, nbatches - ibatch_split, t_loop_start);
        }
    }
}

/// Ports `ggml_opt_epoch_callback_progress_bar` (ggml-opt.cpp:926 @b1b770d90).
///
/// Parameters:
/// - `train`: whether the batch just evaluated was a training batch.
/// - `opt_ctx`: the context.
/// - `dataset`: unused.
/// - `result`: the result the batches accumulate into.
/// - `ibatch`: batches evaluated so far in this section.
/// - `ibatch_max`: batches in this section.
/// - `t_start_us`: when this section started.
///
/// Return: nothing; writes one progress line to stderr.
pub fn ggml_opt_epoch_callback_progress_bar(
    train: bool,
    opt_ctx: mod.impl.ggml_c.ggml_opt_context_t,
    dataset: mod.impl.ggml_c.ggml_opt_dataset_t,
    result: mod.impl.ggml_c.ggml_opt_result_t,
    ibatch: i64,
    ibatch_max: i64,
    t_start_us: i64,
) callconv(.c) void {
    _ = stdlib.libc.fprintf(stdlib.stdio.cStream(.stderr), "%s[", @as([*:0]const u8, if (train) "train: " else "val:   "));

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    const bar_length: i64 = 8;
    const ibatch8 = 8 * ibatch;
    var j: i64 = 0;
    while (j < bar_length) : (j += 1) {
        const block: [*:0]const u8 = if (@divTrunc(ibatch_max * (8 * j + 8), bar_length) < ibatch8)
            "\u{2588}" // full block
        else if (@divTrunc(ibatch_max * (8 * j + 7), bar_length) < ibatch8)
            "\u{2589}" // 7/8 filled
        else if (@divTrunc(ibatch_max * (8 * j + 6), bar_length) < ibatch8)
            "\u{258A}" // 6/8 filled
        else if (@divTrunc(ibatch_max * (8 * j + 5), bar_length) < ibatch8)
            "\u{258B}" // 5/8 filled
        else if (@divTrunc(ibatch_max * (8 * j + 4), bar_length) < ibatch8)
            "\u{258C}" // 4/8 filled
        else if (@divTrunc(ibatch_max * (8 * j + 3), bar_length) < ibatch8)
            "\u{258D}" // 3/8 filled
        else if (@divTrunc(ibatch_max * (8 * j + 2), bar_length) < ibatch8)
            "\u{258E}" // 2/8 filled
        else if (@divTrunc(ibatch_max * (8 * j + 1), bar_length) < ibatch8)
            "\u{258F}" // 1/8 filled
        else
            " ";
        _ = stdlib.libc.fprintf(stdlib.stdio.cStream(.stderr), block);
    }

    const batch_size = tensorOf(ggml_opt_inputs(opt_ctx)).ne[1];
    const idata = ibatch * batch_size;
    const idata_max = ibatch_max * batch_size;

    var loss: f64 = undefined;
    var loss_unc: f64 = undefined;
    ggml_opt_result_loss(result, &loss, &loss_unc);

    var accuracy: f64 = undefined;
    var accuracy_unc: f64 = undefined;
    ggml_opt_result_accuracy(result, &accuracy, &accuracy_unc);

    const t_ibatch_us = mod.impl.ggml_c.ggml_time_us() - t_start_us;
    var t_ibatch_s = @divTrunc(t_ibatch_us, 1000000);
    const t_ibatch_h = @divTrunc(t_ibatch_s, 3600);
    t_ibatch_s -= t_ibatch_h * 3600;
    const t_ibatch_m = @divTrunc(t_ibatch_s, 60);
    t_ibatch_s -= t_ibatch_m * 60;

    const t_eta_us = @divTrunc(t_ibatch_us * (ibatch_max - ibatch), ibatch);
    var t_eta_s = @divTrunc(t_eta_us, 1000000);
    const t_eta_h = @divTrunc(t_eta_s, 3600);
    t_eta_s -= t_eta_h * 3600;
    const t_eta_m = @divTrunc(t_eta_s, 60);
    t_eta_s -= t_eta_m * 60;

    _ = stdlib.libc.fprintf(
        stdlib.stdio.cStream(.stderr),
        "] data=%07lld/%07lld loss=%.5lf\u{b1}%.5lf acc=%.2lf\u{b1}%.2lf%% t=%02lld:%02lld:%02lld ETA=%02lld:%02lld:%02lld \r",
        idata,
        idata_max,
        loss,
        loss_unc,
        100.0 * accuracy,
        100.0 * accuracy_unc,
        t_ibatch_h,
        t_ibatch_m,
        t_ibatch_s,
        t_eta_h,
        t_eta_m,
        t_eta_s,
    );
    if (ibatch == ibatch_max) {
        _ = stdlib.libc.fprintf(stdlib.stdio.cStream(.stderr), "\n");
    }
    _ = stdlib.libc.fflush(stdlib.stdio.cStream(.stderr));

    _ = dataset;
}

/// Ports `ggml_opt_fit` (ggml-opt.cpp:999 @b1b770d90).
///
/// Parameters:
/// - `backend_sched`: the scheduler the graphs are built for.
/// - `ctx_compute`: the context holding the temporarily allocated tensors.
/// - `inputs`: input tensor, `[ne_datapoint, ndata_batch]`.
/// - `outputs`: output tensor, `[ne_label, ndata_batch]` if labels are used.
/// - `dataset`: data and optionally labels.
/// - `loss_type`: the loss to minimize.
/// - `optimizer`: AdamW or SGD.
/// - `get_opt_pars`: optimizer parameters; its userdata is the epoch, an
///   `int64_t *`.
/// - `nepoch`: passes over the dataset.
/// - `nbatch_logical`: datapoints per optimizer step, a multiple of the
///   physical batch.
/// - `val_split`: fraction held out for validation, in `[0, 1)`.
/// - `silent`: suppress the progress output on stderr.
///
/// Return: nothing.
pub fn ggml_opt_fit(
    backend_sched: mod.impl.ggml_c.ggml_backend_sched_t,
    ctx_compute: ?*mod.impl.ggml_c.ggml_context,
    inputs: [*c]mod.impl.ggml_c.ggml_tensor,
    outputs: [*c]mod.impl.ggml_c.ggml_tensor,
    dataset: mod.impl.ggml_c.ggml_opt_dataset_t,
    loss_type: mod.impl.ggml_c.enum_ggml_opt_loss_type,
    optimizer: mod.impl.ggml_c.enum_ggml_opt_optimizer_type,
    get_opt_pars: mod.impl.ggml_c.ggml_opt_get_optimizer_params,
    nepoch: i64,
    nbatch_logical: i64,
    val_split: f32,
    silent: bool,
) callconv(.c) void {
    mod.impl.ggml_c.ggml_time_init();
    const t_start_us = mod.impl.ggml_c.ggml_time_us();

    const ndata = tensorOf(ggml_opt_dataset_data(dataset)).ne[1];
    const nbatch_physical = tensorOf(inputs).ne[1];
    mod.impl.assert(@rem(ndata, nbatch_logical) == 0, "ndata          % nbatch_logical  == 0");
    mod.impl.assert(@rem(nbatch_logical, nbatch_physical) == 0, "nbatch_logical % nbatch_physical == 0");

    const opt_period = @divTrunc(nbatch_logical, nbatch_physical);
    const nbatches_logical = @divTrunc(ndata, nbatch_logical);

    mod.impl.assert(val_split >= 0.0, "val_split >= 0.0f");
    mod.impl.assert(val_split < 1.0, "val_split <  1.0f");
    const ibatch_split = @as(i64, @intFromFloat((1.0 - val_split) * @as(f32, @floatFromInt(nbatches_logical)))) * opt_period; // train <-> val split index (physical)
    const idata_split = ibatch_split * nbatch_physical;

    var epoch: i64 = 1;

    var params = ggml_opt_default_params(backend_sched, loss_type);
    params.ctx_compute = ctx_compute;
    params.inputs = inputs;
    params.outputs = outputs;
    params.opt_period = @truncate(opt_period);
    params.get_opt_pars = get_opt_pars;
    params.get_opt_pars_ud = &epoch;
    params.optimizer = optimizer;
    const opt_ctx = ggml_opt_init(params);

    // Shuffling the data is generally useful but there is only a point if not all data is used in a single batch.
    if (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    const result_train = ggml_opt_result_init();
    const result_val = ggml_opt_result_init();

    const epoch_callback: mod.impl.ggml_c.ggml_opt_epoch_callback = if (silent) null else &ggml_opt_epoch_callback_progress_bar;

    while (epoch <= nepoch) : (epoch += 1) {
        if (nbatch_logical < idata_split) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_val);

        if (!silent) {
            _ = stdlib.libc.fprintf(stdlib.stdio.cStream(.stderr), "%s: epoch %04lld/%04lld:\n", "ggml_opt_fit", epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val, idata_split, epoch_callback, epoch_callback);
        if (!silent) {
            _ = stdlib.libc.fprintf(stdlib.stdio.cStream(.stderr), "\n");
        }
    }

    if (!silent) {
        var t_total_s = @divTrunc(mod.impl.ggml_c.ggml_time_us() - t_start_us, 1000000);
        const t_total_h = @divTrunc(t_total_s, 3600);
        t_total_s -= t_total_h * 3600;
        const t_total_m = @divTrunc(t_total_s, 60);
        t_total_s -= t_total_m * 60;
        _ = stdlib.libc.fprintf(stdlib.stdio.cStream(.stderr), "%s: training took %02lld:%02lld:%02lld\n", "ggml_opt_fit", t_total_h, t_total_m, t_total_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_val);
}

/// Ports `ggml_opt_context_optimizer_type` (ggml-opt.cpp:1081 @b1b770d90).
///
/// Parameters:
/// - `opt_ctx`: the context.
///
/// Return: its optimizer.
pub fn ggml_opt_context_optimizer_type(opt_ctx: mod.impl.ggml_c.ggml_opt_context_t) callconv(.c) mod.impl.ggml_c.enum_ggml_opt_optimizer_type {
    return contextOf(opt_ctx).optimizer;
}

/// Ports `ggml_opt_optimizer_name` (ggml-opt.cpp:1085 @b1b770d90).
///
/// Parameters:
/// - `o`: an optimizer type.
///
/// Return: `"adamw"`, `"sgd"`, or `"undefined"`; static storage.
pub fn ggml_opt_optimizer_name(o: mod.impl.ggml_c.enum_ggml_opt_optimizer_type) callconv(.c) [*c]const u8 {
    return switch (o) {
        mod.impl.ggml_c.GGML_OPT_OPTIMIZER_TYPE_ADAMW => "adamw",
        mod.impl.ggml_c.GGML_OPT_OPTIMIZER_TYPE_SGD => "sgd",
        else => "undefined",
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the optimizer names are the C's" {
    try std.testing.expectEqualStrings("adamw", std.mem.span(ggml_opt_optimizer_name(mod.impl.ggml_c.GGML_OPT_OPTIMIZER_TYPE_ADAMW)));
    try std.testing.expectEqualStrings("sgd", std.mem.span(ggml_opt_optimizer_name(mod.impl.ggml_c.GGML_OPT_OPTIMIZER_TYPE_SGD)));
    try std.testing.expectEqualStrings("undefined", std.mem.span(ggml_opt_optimizer_name(mod.impl.ggml_c.GGML_OPT_OPTIMIZER_TYPE_COUNT)));
}
