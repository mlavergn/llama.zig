//! Diff `src/ggml/opt.zig` against the C++ `ggml-opt.cpp`, in one process.
//!
//! # Provenance
//!
//! **Not a port.** This is our own code. Its scenarios are taken from
//! `llama.cpp/tests/test-opt.cpp` (v0.3.0, `c1d0e7a00`) — the dataset,
//! gradient, forward/backward, epoch-versus-fit, idata-split,
//! gradient-accumulation and regression tests — and widened with the paths
//! that file never takes: cross-entropy with predictions and accuracy, a
//! two-layer network, dynamic graphs, a forward-only build, `ggml_opt_fit`
//! with shuffling, a validation split, gradient accumulation and its
//! progress output, and the assertions.
//!
//! # How both live in one process
//!
//! `scripts/opt-diff` compiles the reference `ggml-opt.cpp` with each of its
//! 37 unmangled exports renamed to `ref_*` on the command line, the method
//! `scripts/repack-diff` uses. The port is this file's `opt` module, and the
//! root declares `ggml_opt_exports = false` so it defines **no** symbols:
//! the port is reached only through Zig, the reference only through `ref_*`,
//! and the `ggml-opt.o` inside `libggml.a` — which still defines the real
//! names — is never pulled in, because nothing references them. One
//! definition of every name in the link, by construction.
//!
//! Both sides call the same `libggml.a` for everything else, so a difference
//! can only come from `ggml-opt`.
//!
//! # What is compared
//!
//! Every scenario runs once through each side's 37 functions, collected into
//! an `Api` whose field types are the header's own function types — a port
//! signature that disagrees with `ggml-opt.h` does not compile. Each run
//! appends records to a trace; the traces are compared line by line. The
//! records are: every parameter's bits after every step, the loss tensor,
//! the gradient accumulators, every `ggml_opt_result_*` answer (loss,
//! uncertainty, accuracy, predictions, counts) on bits, every batch a
//! dataset hands out, the names and shapes of the tensors the context
//! exposes, the progress bar's text with its timings masked, and — with the
//! scheduler's eval callback on — **every node of every graph the context
//! builds**: name, op, shape and a hash of its output. That last one sees
//! the graph `ggml_opt_build` constructs, the AdamW/SGD step nodes and their
//! names, and every intermediate value, not just the final weights.
//!
//! Separately, `opt.uniformInt` and `opt.shuffle` are compared with libc++'s
//! templates (`harness/opt_ref.cpp`) over ranges up to 2^63.

const std = @import("std");
const opt = @import("opt");
const c = @import("ggml_c");

/// Keeps `src/ggml/opt.zig` from defining its C symbols in this binary.
pub const ggml_opt_exports = false;

extern fn opt_ref_uniform(seed: u32, a: i64, b: i64, out: [*]i64, count: usize) u32;
extern fn opt_ref_shuffle(seed: u32, v: [*]i64, n: usize, reps: usize) u32;

extern "c" fn fork() c_int;
extern "c" fn waitpid(pid: c_int, status: *c_int, options: c_int) c_int;
extern "c" fn _exit(code: c_int) noreturn;
extern "c" fn dup(fd: c_int) c_int;
extern "c" fn dup2(a: c_int, b: c_int) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn ftruncate(fd: c_int, len: i64) c_int;
extern "c" fn mkstemp(template: [*:0]u8) c_int;
extern "c" fn unlink(path: [*:0]const u8) c_int;
extern "c" fn fflush(stream: ?*anyopaque) c_int;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern var __stderrp: *anyopaque;

// -----------------------------------------------------------------------------
// The two sides

/// One side's 37 entry points. Field `x` is `ggml_opt_x`; its type is the
/// header's.
const Api = struct {
    dataset_init: *const @TypeOf(c.ggml_opt_dataset_init),
    dataset_free: *const @TypeOf(c.ggml_opt_dataset_free),
    dataset_ndata: *const @TypeOf(c.ggml_opt_dataset_ndata),
    dataset_data: *const @TypeOf(c.ggml_opt_dataset_data),
    dataset_labels: *const @TypeOf(c.ggml_opt_dataset_labels),
    dataset_shuffle: *const @TypeOf(c.ggml_opt_dataset_shuffle),
    dataset_get_batch: *const @TypeOf(c.ggml_opt_dataset_get_batch),
    dataset_get_batch_host: *const @TypeOf(c.ggml_opt_dataset_get_batch_host),
    get_default_optimizer_params: *const @TypeOf(c.ggml_opt_get_default_optimizer_params),
    get_constant_optimizer_params: *const @TypeOf(c.ggml_opt_get_constant_optimizer_params),
    default_params: *const @TypeOf(c.ggml_opt_default_params),
    init: *const @TypeOf(c.ggml_opt_init),
    free: *const @TypeOf(c.ggml_opt_free),
    reset: *const @TypeOf(c.ggml_opt_reset),
    static_graphs: *const @TypeOf(c.ggml_opt_static_graphs),
    inputs: *const @TypeOf(c.ggml_opt_inputs),
    outputs: *const @TypeOf(c.ggml_opt_outputs),
    labels: *const @TypeOf(c.ggml_opt_labels),
    loss: *const @TypeOf(c.ggml_opt_loss),
    pred: *const @TypeOf(c.ggml_opt_pred),
    ncorrect: *const @TypeOf(c.ggml_opt_ncorrect),
    grad_acc: *const @TypeOf(c.ggml_opt_grad_acc),
    result_init: *const @TypeOf(c.ggml_opt_result_init),
    result_free: *const @TypeOf(c.ggml_opt_result_free),
    result_reset: *const @TypeOf(c.ggml_opt_result_reset),
    result_ndata: *const @TypeOf(c.ggml_opt_result_ndata),
    result_loss: *const @TypeOf(c.ggml_opt_result_loss),
    result_pred: *const @TypeOf(c.ggml_opt_result_pred),
    result_accuracy: *const @TypeOf(c.ggml_opt_result_accuracy),
    prepare_alloc: *const @TypeOf(c.ggml_opt_prepare_alloc),
    alloc: *const @TypeOf(c.ggml_opt_alloc),
    eval: *const @TypeOf(c.ggml_opt_eval),
    epoch: *const @TypeOf(c.ggml_opt_epoch),
    epoch_callback_progress_bar: *const @TypeOf(c.ggml_opt_epoch_callback_progress_bar),
    fit: *const @TypeOf(c.ggml_opt_fit),
    context_optimizer_type: *const @TypeOf(c.ggml_opt_context_optimizer_type),
    optimizer_name: *const @TypeOf(c.ggml_opt_optimizer_name),
};

comptime {
    const info = @typeInfo(Api).@"struct";
    if (info.field_names.len != opt.exported_names.len) @compileError("Api does not cover the contract");
    for (info.field_names, opt.exported_names) |f, name| {
        if (!std.mem.eql(u8, "ggml_opt_" ++ f, name)) @compileError("Api out of order at " ++ name);
    }
}

fn makeApi(comptime ref: bool) Api {
    var a: Api = undefined;
    const info = @typeInfo(Api).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        const sym = "ggml_opt_" ++ name;
        @field(a, name) = if (ref) @extern(T, .{ .name = "ref_" ++ sym }) else &@field(opt, sym);
    }
    return a;
}

const ref_api = makeApi(true);
const port_api = makeApi(false);

// -----------------------------------------------------------------------------
// Traces

const Trace = struct {
    arena: std.mem.Allocator,
    lines: std.ArrayList([]const u8) = .empty,

    fn add(self: *Trace, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.allocPrint(self.arena, fmt, args) catch @panic("oom");
        self.lines.append(self.arena, s) catch @panic("oom");
    }
};

/// What one scenario run sees: the side, its trace, and the backends.
const Run = struct {
    api: *const Api,
    tr: *Trace,
    /// The backend under test, which the scheduler prefers.
    backend: c.ggml_backend_t,
    /// The CPU backend, the scheduler's last.
    cpu: c.ggml_backend_t,
    /// Trace every node the scheduler computes.
    nodes: bool,
    /// The scheduler, made fresh for each scenario.
    sched: c.ggml_backend_sched_t = null,

    fn newSched(self: *Run) c.ggml_backend_sched_t {
        var backends: [2]c.ggml_backend_t = .{ self.backend, self.cpu };
        const n: c_int = if (self.backend == self.cpu) 1 else 2;
        const s = c.ggml_backend_sched_new(&backends, null, n, c.GGML_DEFAULT_GRAPH_SIZE, false, true);
        if (self.nodes) c.ggml_backend_sched_set_eval_callback(s, &nodeCallback, self.tr);
        self.sched = s;
        return s;
    }

    fn freeSched(self: *Run) void {
        c.ggml_backend_sched_free(self.sched);
        self.sched = null;
    }
};

/// The side currently running, for the C callbacks that have no user data
/// to carry it.
var g_run: *Run = undefined;

fn hashBytes(bytes: []const u8) u64 {
    return std.hash.Wyhash.hash(0, bytes);
}

fn nodeCallback(t: [*c]c.ggml_tensor, ask: bool, ud: ?*anyopaque) callconv(.c) bool {
    if (ask) return true;
    const tr: *Trace = @ptrCast(@alignCast(ud));
    const tt: *c.ggml_tensor = @ptrCast(t);
    const n = c.ggml_nbytes(t);
    const buf = std.heap.c_allocator.alloc(u8, n) catch @panic("oom");
    defer std.heap.c_allocator.free(buf);
    c.ggml_backend_tensor_get(t, buf.ptr, 0, n);
    tr.add("node {s} op={s} type={d} ne={d},{d},{d},{d} flags={x} h={x}", .{
        std.mem.sliceTo(&tt.name, 0), std.mem.span(c.ggml_op_desc(t)), tt.type,
        tt.ne[0],                     tt.ne[1],                        tt.ne[2],
        tt.ne[3],                     tt.flags,                        hashBytes(buf),
    });
    // The sources by name and kind: the optimizer's momenta and parameter
    // tensor are leaves, never computed, so this is the only place their
    // names and flags are seen. Swapping the names of `m` and `v` escaped
    // without it.
    for (tt.src, 0..) |s, i| {
        if (s == null) continue;
        const st: *c.ggml_tensor = @ptrCast(s);
        tr.add("  src{d} {s} op={s} type={d} ne={d},{d},{d},{d} flags={x}", .{
            i, std.mem.sliceTo(&st.name, 0), std.mem.span(c.ggml_op_desc(s)), st.type, st.ne[0], st.ne[1], st.ne[2], st.ne[3], st.flags,
        });
    }
    return true;
}

fn bits32(x: f32) u32 {
    return @bitCast(x);
}

fn bits64(x: f64) u64 {
    return @bitCast(x);
}

/// Records a tensor's identity and contents: every value when it is small,
/// a hash otherwise.
fn recTensor(r: *Run, tag: []const u8, t: [*c]c.ggml_tensor) void {
    if (t == null) {
        r.tr.add("{s}: null", .{tag});
        return;
    }
    const tt: *c.ggml_tensor = @ptrCast(t);
    const n = c.ggml_nbytes(t);
    var line: std.ArrayList(u8) = .empty;
    defer line.deinit(std.heap.c_allocator);
    line.print(std.heap.c_allocator, "{s}: '{s}' type={d} ne={d},{d},{d},{d} flags={x}", .{
        tag, std.mem.sliceTo(&tt.name, 0), tt.type, tt.ne[0], tt.ne[1], tt.ne[2], tt.ne[3], tt.flags,
    }) catch @panic("oom");
    if (tt.buffer != null or tt.data != null) {
        const buf = std.heap.c_allocator.alloc(u8, n) catch @panic("oom");
        defer std.heap.c_allocator.free(buf);
        c.ggml_backend_tensor_get(t, buf.ptr, 0, n);
        if (n <= 64) {
            line.appendSlice(std.heap.c_allocator, " =") catch @panic("oom");
            for (buf) |b| line.print(std.heap.c_allocator, "{x:0>2}", .{b}) catch @panic("oom");
        } else {
            line.print(std.heap.c_allocator, " h={x}", .{hashBytes(buf)}) catch @panic("oom");
        }
    } else {
        line.appendSlice(std.heap.c_allocator, " unallocated") catch @panic("oom");
    }
    r.tr.add("{s}", .{line.items});
}

/// Records every answer a result gives, on bits, with and without the
/// optional uncertainty pointers.
fn recResult(r: *Run, tag: []const u8, res: c.ggml_opt_result_t) void {
    const a = r.api;
    var nd: i64 = -99;
    a.result_ndata(res, &nd);
    var loss: f64 = 0;
    var unc: f64 = 0;
    a.result_loss(res, &loss, &unc);
    // With no batches the C writes NaN through `unc` before checking it for
    // null, so the null form is only asked once there is something to say.
    var loss2: f64 = 0;
    if (nd > 0) a.result_loss(res, &loss2, null);
    var acc: f64 = 0;
    var acc_unc: f64 = 0;
    a.result_accuracy(res, &acc, &acc_unc);
    var acc2: f64 = 0;
    a.result_accuracy(res, &acc2, null);
    r.tr.add("{s}: ndata={d} loss={x} unc={x} loss2={x} acc={x} acc_unc={x} acc2={x}", .{
        tag, nd, bits64(loss), bits64(unc), bits64(loss2), bits64(acc), bits64(acc_unc), bits64(acc2),
    });
    if (nd > 0 and nd < 100_000) {
        const pred = std.heap.c_allocator.alloc(i32, @intCast(nd)) catch @panic("oom");
        defer std.heap.c_allocator.free(pred);
        @memset(pred, -7);
        a.result_pred(res, pred.ptr);
        r.tr.add("{s}: pred h={x}", .{ tag, hashBytes(std.mem.sliceAsBytes(pred)) });
    }
}

/// Records what a context exposes: its tensors' names and shapes, whether
/// its graphs are static, and its optimizer.
fn recContext(r: *Run, tag: []const u8, ctx: c.ggml_opt_context_t) void {
    const a = r.api;
    r.tr.add("{s}: static={} optimizer={d}", .{ tag, a.static_graphs(ctx), a.context_optimizer_type(ctx) });
    const getters = .{ .{ "inputs", a.inputs }, .{ "outputs", a.outputs }, .{ "labels", a.labels }, .{ "loss", a.loss }, .{ "pred", a.pred }, .{ "ncorrect", a.ncorrect } };
    inline for (getters) |g| {
        const t = g[1](ctx);
        if (t == null) {
            r.tr.add("{s}.{s}: null", .{ tag, g[0] });
        } else {
            const tt: *c.ggml_tensor = @ptrCast(t);
            r.tr.add("{s}.{s}: '{s}' op={s} type={d} ne={d},{d},{d},{d} flags={x}", .{
                tag,      g[0],     std.mem.sliceTo(&tt.name, 0), std.mem.span(c.ggml_op_desc(t)), tt.type,
                tt.ne[0], tt.ne[1], tt.ne[2],                     tt.ne[3],                        tt.flags,
            });
        }
    }
}

// -----------------------------------------------------------------------------
// test-opt.cpp's fixture

const ne_datapoint: i64 = 2;
const ne_label: i64 = 1;
const ndata: i64 = 6;

/// Ports test-opt.cpp's `helper_ctx_data`.
const Helper = struct {
    datasets_supervised: [ndata]c.ggml_opt_dataset_t,
    data_batch: [ndata][*c]c.ggml_tensor,
    labels_batch: [ndata][*c]c.ggml_tensor,
    dataset_unsupervised: c.ggml_opt_dataset_t,
    ctx_static: ?*c.ggml_context,
    ctx_compute: ?*c.ggml_context,
    opt_params: c.struct_ggml_opt_params,
    opt_ctx: c.ggml_opt_context_t,
    inputs: [*c]c.ggml_tensor,
    weights: [*c]c.ggml_tensor,
    outputs: [*c]c.ggml_tensor,
    buf: c.ggml_backend_buffer_t,
    result: c.ggml_opt_result_t,
    result2: c.ggml_opt_result_t,
};

/// test-opt.cpp's `helper_get_test_opt_pars`, through the running side's
/// own default parameters.
fn testOptPars(userdata: ?*anyopaque) callconv(.c) c.struct_ggml_opt_optimizer_params {
    var result = g_run.api.get_default_optimizer_params(userdata);
    result.adamw.alpha = 1.0;
    result.adamw.beta1 = 0.0;
    result.adamw.beta2 = 0.0;
    result.adamw.eps = 0.0;
    result.adamw.wd = 0.0;
    result.sgd.wd = 0.0;
    result.sgd.alpha = 1.0;
    return result;
}

fn newCtx(size: usize) ?*c.ggml_context {
    return c.ggml_init(.{ .mem_size = size, .mem_buffer = null, .no_alloc = true });
}

fn computeCtxSize() usize {
    return @as(usize, c.GGML_DEFAULT_GRAPH_SIZE) * c.ggml_tensor_overhead() + 3 * c.ggml_graph_overhead();
}

/// Ports test-opt.cpp's `helper_get_ctx_data`.
fn helperInit(
    r: *Run,
    optim: c.enum_ggml_opt_optimizer_type,
    init_opt_ctx: bool,
    optimizer_defaults: bool,
    nbatch_logical: i64,
    nbatch_physical: i64,
    loss_type: c.enum_ggml_opt_loss_type,
) Helper {
    const a = r.api;
    var h: Helper = undefined;
    var ndata_shard: i64 = 1;
    while (ndata_shard <= ndata) : (ndata_shard += 1) {
        const dataset = a.dataset_init(c.GGML_TYPE_F32, c.GGML_TYPE_F32, ne_datapoint, ne_label, ndata, ndata_shard);
        const data: [*]f32 = @ptrCast(c.ggml_get_data_f32(a.dataset_data(dataset)));
        const labels: [*]f32 = @ptrCast(c.ggml_get_data_f32(a.dataset_labels(dataset)));
        for (0..ndata) |idata| {
            for (0..ne_datapoint) |id| data[idata * ne_datapoint + id] = @floatFromInt(16 * idata + id);
            for (0..ne_label) |il| labels[idata * ne_label + il] = @floatFromInt(16 * (16 * idata + il));
        }
        h.datasets_supervised[@intCast(ndata_shard - 1)] = dataset;
    }

    h.dataset_unsupervised = a.dataset_init(c.GGML_TYPE_F32, c.GGML_TYPE_F32, 1, 0, ndata, 1);
    const data: [*]f32 = @ptrCast(c.ggml_get_data_f32(a.dataset_data(h.dataset_unsupervised)));
    for (0..ndata) |idata| data[idata] = @floatFromInt(idata);

    h.ctx_static = newCtx(@as(usize, 2 * ndata + 2) * c.ggml_tensor_overhead());
    h.ctx_compute = newCtx(computeCtxSize());

    var nb: i64 = 1;
    while (nb <= ndata) : (nb += 1) {
        h.data_batch[@intCast(nb - 1)] = c.ggml_new_tensor_1d(h.ctx_static, c.GGML_TYPE_F32, nb * ne_datapoint);
        h.labels_batch[@intCast(nb - 1)] = c.ggml_new_tensor_1d(h.ctx_static, c.GGML_TYPE_F32, nb * ne_label);
    }

    h.inputs = c.ggml_new_tensor_1d(h.ctx_static, c.GGML_TYPE_F32, nbatch_physical);
    _ = c.ggml_set_name(h.inputs, "inputs");

    h.weights = c.ggml_new_tensor_1d(h.ctx_static, c.GGML_TYPE_F32, 1);
    _ = c.ggml_set_name(h.weights, "weights");
    c.ggml_set_param(h.weights);

    const intermediary = c.ggml_add(h.ctx_compute, h.inputs, h.weights);

    h.outputs = c.ggml_scale(h.ctx_compute, intermediary, 1.0);
    _ = c.ggml_set_name(h.outputs, "outputs");

    h.buf = c.ggml_backend_alloc_ctx_tensors(h.ctx_static, r.backend);
    const w0: f32 = @as(f32, @floatFromInt(ndata)) / 2;
    c.ggml_backend_tensor_set(h.weights, &w0, 0, @sizeOf(f32));

    const opt_period: i32 = @intCast(@divTrunc(nbatch_logical, nbatch_physical));

    h.opt_params = a.default_params(r.newSched(), loss_type);
    h.opt_params.ctx_compute = h.ctx_compute;
    h.opt_params.inputs = h.inputs;
    h.opt_params.outputs = h.outputs;
    h.opt_params.opt_period = opt_period;
    h.opt_params.optimizer = optim;
    if (!optimizer_defaults) h.opt_params.get_opt_pars = &testOptPars;
    h.opt_ctx = if (init_opt_ctx) a.init(h.opt_params) else null;

    h.result = a.result_init();
    h.result2 = a.result_init();
    return h;
}

/// Ports test-opt.cpp's `helper_free_ctx_data`.
fn helperFree(r: *Run, h: *Helper) void {
    const a = r.api;
    a.result_free(h.result);
    a.result_free(h.result2);
    a.free(h.opt_ctx);
    c.ggml_backend_buffer_free(h.buf);
    c.ggml_free(h.ctx_static);
    c.ggml_free(h.ctx_compute);
    for (h.datasets_supervised) |d| a.dataset_free(d);
    a.dataset_free(h.dataset_unsupervised);
    r.freeSched();
}

fn recWeights(r: *Run, tag: []const u8, w: [*c]c.ggml_tensor) void {
    var v: f32 = undefined;
    c.ggml_backend_tensor_get(w, &v, 0, @sizeOf(f32));
    r.tr.add("{s}: w={x}", .{ tag, bits32(v) });
}

fn recLoss(r: *Run, tag: []const u8, ctx: c.ggml_opt_context_t) void {
    var v: f32 = undefined;
    c.ggml_backend_tensor_get(r.api.loss(ctx), &v, 0, @sizeOf(f32));
    r.tr.add("{s}: loss={x}", .{ tag, bits32(v) });
}

// -----------------------------------------------------------------------------
// test-opt.cpp's scenarios

/// Ports test-opt.cpp's `test_dataset`, both through tensors and through
/// `get_batch_host`, recording every batch.
fn scenarioDataset(r: *Run, optim: c.enum_ggml_opt_optimizer_type, shuffle: bool) void {
    const a = r.api;
    var h = helperInit(r, optim, true, true, 1, 1, c.GGML_OPT_LOSS_TYPE_SUM);
    defer helperFree(r, &h);

    var ndata_shard: i64 = 1;
    while (ndata_shard <= ndata) : (ndata_shard += 1) {
        const dataset = h.datasets_supervised[@intCast(ndata_shard - 1)];
        if (shuffle) a.dataset_shuffle(h.opt_ctx, dataset, -1);
        r.tr.add("dataset shard={d} ndata={d}", .{ ndata_shard, a.dataset_ndata(dataset) });

        var ndata_batch: i64 = 1;
        while (ndata_batch <= ndata) : (ndata_batch += 1) {
            if (@rem(ndata_batch, ndata_shard) != 0) continue;
            const data_batch = h.data_batch[@intCast(ndata_batch - 1)];
            const labels_batch = h.labels_batch[@intCast(ndata_batch - 1)];
            const nbatches = @divTrunc(ndata, ndata_batch);
            var ibatch: i64 = 0;
            while (ibatch < nbatches) : (ibatch += 1) {
                a.dataset_get_batch(dataset, data_batch, labels_batch, ibatch);
                recTensor(r, "batch.data", data_batch);
                recTensor(r, "batch.labels", labels_batch);

                var host_data: [ndata * ne_datapoint]f32 = @splat(-1);
                var host_labels: [ndata * ne_label]f32 = @splat(-1);
                a.dataset_get_batch_host(dataset, &host_data, @intCast(ndata_batch * ne_datapoint * 4), &host_labels, ibatch);
                r.tr.add("host d={x} l={x}", .{ hashBytes(std.mem.sliceAsBytes(&host_data)), hashBytes(std.mem.sliceAsBytes(&host_labels)) });
            }
        }
    }
}

/// Ports test-opt.cpp's `test_grad`.
fn scenarioGrad(r: *Run, optim: c.enum_ggml_opt_optimizer_type) void {
    const a = r.api;
    var h = helperInit(r, optim, true, false, 999999, 1, c.GGML_OPT_LOSS_TYPE_SUM);
    defer helperFree(r, &h);
    for (0..ndata) |idata| {
        const idataf: f32 = @floatFromInt(idata);
        a.alloc(h.opt_ctx, true);
        c.ggml_backend_tensor_set(h.inputs, &idataf, 0, c.ggml_nbytes(h.inputs));
        a.eval(h.opt_ctx, h.result);
        recTensor(r, "grad_acc", a.grad_acc(h.opt_ctx, h.weights));
        recWeights(r, "grad", h.weights);
    }
    recResult(r, "grad.result", h.result);
}

/// Ports test-opt.cpp's `test_forward_backward`.
fn scenarioForwardBackward(r: *Run, optim: c.enum_ggml_opt_optimizer_type, high_level: bool, shuffle: bool) void {
    const a = r.api;
    var h = helperInit(r, optim, true, false, 1, 1, c.GGML_OPT_LOSS_TYPE_SUM);
    defer helperFree(r, &h);
    recContext(r, "fb.ctx", h.opt_ctx);
    recResult(r, "fb.initial", h.result);

    if (high_level) {
        const dataset = h.dataset_unsupervised;
        if (shuffle) a.dataset_shuffle(h.opt_ctx, dataset, -1);
        a.epoch(h.opt_ctx, dataset, null, h.result, 0, null, null);
    } else {
        for (0..ndata) |idata| {
            const idataf: f32 = @floatFromInt(idata);
            a.alloc(h.opt_ctx, false);
            c.ggml_backend_tensor_set(h.inputs, &idataf, 0, c.ggml_nbytes(h.inputs));
            a.eval(h.opt_ctx, h.result);
            recLoss(r, "fb.fwd", h.opt_ctx);
        }
    }
    recWeights(r, "fb.after_forward", h.weights);
    recResult(r, "fb.after_forward", h.result);

    var w0: f32 = undefined;
    c.ggml_backend_tensor_get(h.weights, &w0, 0, @sizeOf(f32));
    for (0..10) |_| {
        a.alloc(h.opt_ctx, true);
        a.eval(h.opt_ctx, h.result);
        recWeights(r, "fb.leak", h.weights);
    }
    c.ggml_backend_tensor_set(h.weights, &w0, 0, @sizeOf(f32));

    a.reset(h.opt_ctx, false);
    a.result_reset(h.result);
    recResult(r, "fb.reset", h.result);

    if (high_level) {
        const dataset = h.dataset_unsupervised;
        if (shuffle) a.dataset_shuffle(h.opt_ctx, dataset, -1);
        a.epoch(h.opt_ctx, dataset, h.result, null, ndata, null, null);
    } else {
        for (0..ndata) |idata| {
            const idataf: f32 = @floatFromInt(idata);
            a.alloc(h.opt_ctx, true);
            c.ggml_backend_tensor_set(h.inputs, &idataf, 0, c.ggml_nbytes(h.inputs));
            a.eval(h.opt_ctx, h.result);
            recLoss(r, "fb.bwd", h.opt_ctx);
            recWeights(r, "fb.bwd", h.weights);
        }
    }
    recWeights(r, "fb.after_backward", h.weights);
    recResult(r, "fb.after_backward", h.result);

    // Reset the optimizer too, which `test-opt.cpp` never does, and train on.
    a.reset(h.opt_ctx, true);
    for (0..3) |idata| {
        const idataf: f32 = @floatFromInt(idata);
        a.alloc(h.opt_ctx, true);
        c.ggml_backend_tensor_set(h.inputs, &idataf, 0, c.ggml_nbytes(h.inputs));
        a.eval(h.opt_ctx, h.result);
        recWeights(r, "fb.after_opt_reset", h.weights);
    }
}

/// Ports test-opt.cpp's `test_epoch_vs_fit`.
fn scenarioEpochVsFit(r: *Run, optim: c.enum_ggml_opt_optimizer_type) void {
    const a = r.api;
    {
        var h = helperInit(r, optim, true, true, 1, 1, c.GGML_OPT_LOSS_TYPE_SUM);
        defer helperFree(r, &h);
        a.dataset_shuffle(h.opt_ctx, h.dataset_unsupervised, -1);
        a.epoch(h.opt_ctx, h.dataset_unsupervised, h.result, null, ndata, null, null);
        recWeights(r, "evf.epoch", h.weights);
        recResult(r, "evf.epoch", h.result);
    }
    {
        var h = helperInit(r, optim, false, true, 1, 1, c.GGML_OPT_LOSS_TYPE_SUM);
        defer helperFree(r, &h);
        a.fit(r.sched, h.ctx_compute, h.inputs, h.outputs, h.dataset_unsupervised, c.GGML_OPT_LOSS_TYPE_SUM, optim, a.get_default_optimizer_params, 1, 1, 0.0, true);
        recWeights(r, "evf.fit", h.weights);
    }
}

/// Ports test-opt.cpp's `test_idata_split`, for both optimizers.
fn scenarioIdataSplit(r: *Run, optim: c.enum_ggml_opt_optimizer_type, high_level: bool) void {
    const a = r.api;
    var h = helperInit(r, optim, true, false, 1, 1, c.GGML_OPT_LOSS_TYPE_SUM);
    defer helperFree(r, &h);
    const idata_split = @divTrunc(ndata * 2, 3);
    for (1..5) |epoch| {
        if (high_level) {
            a.epoch(h.opt_ctx, h.dataset_unsupervised, h.result, h.result2, idata_split, null, null);
        } else {
            var idata: i64 = 0;
            while (idata < idata_split) : (idata += 1) {
                const idataf: f32 = @floatFromInt(idata);
                a.alloc(h.opt_ctx, true);
                c.ggml_backend_tensor_set(h.inputs, &idataf, 0, c.ggml_nbytes(h.inputs));
                a.eval(h.opt_ctx, h.result);
                recLoss(r, "split.train", h.opt_ctx);
            }
            while (idata < ndata) : (idata += 1) {
                const idataf: f32 = @floatFromInt(idata);
                a.alloc(h.opt_ctx, false);
                c.ggml_backend_tensor_set(h.inputs, &idataf, 0, c.ggml_nbytes(h.inputs));
                a.eval(h.opt_ctx, h.result2);
                recLoss(r, "split.eval", h.opt_ctx);
            }
        }
        r.tr.add("split epoch {d}", .{epoch});
        recWeights(r, "split", h.weights);
        recResult(r, "split.result", h.result);
        recResult(r, "split.result2", h.result2);
        a.result_reset(h.result);
        a.result_reset(h.result2);
    }
}

/// Ports test-opt.cpp's `test_gradient_accumulation`, for both optimizers
/// — the C++ test runs AdamW only.
fn scenarioGradAccum(r: *Run, optim: c.enum_ggml_opt_optimizer_type, nbatch_physical: i64, loss_type: c.enum_ggml_opt_loss_type) void {
    const a = r.api;
    var h = helperInit(r, optim, true, false, 6, nbatch_physical, loss_type);
    defer helperFree(r, &h);
    for (1..5) |epoch| {
        var idata: i64 = 0;
        while (idata < ndata) : (idata += nbatch_physical) {
            var idataf: [2]f32 = .{ @floatFromInt(idata), @floatFromInt(idata + 1) };
            a.alloc(h.opt_ctx, true);
            c.ggml_backend_tensor_set(h.inputs, &idataf, 0, @intCast(nbatch_physical * 4));
            a.eval(h.opt_ctx, h.result);
            recTensor(r, "accum.grad_acc", a.grad_acc(h.opt_ctx, h.weights));
            recWeights(r, "accum", h.weights);
            recLoss(r, "accum", h.opt_ctx);
        }
        r.tr.add("accum epoch {d}", .{epoch});
        recResult(r, "accum.result", h.result);
        a.result_reset(h.result);
    }
}

/// test-opt.cpp's `helper_get_regression_opt_pars`, recording each call
/// with the parameters as they stand before the step.
fn regressionOptPars(userdata: ?*anyopaque) callconv(.c) c.struct_ggml_opt_optimizer_params {
    const epoch: i64 = @as(*const i64, @ptrCast(@alignCast(userdata))).*;
    var result = g_run.api.get_default_optimizer_params(null);
    result.adamw.alpha = 0.1;
    result.sgd.alpha = @floatCast(1e-4 * std.math.pow(f64, 0.99, 1000 * @as(f64, @floatFromInt(epoch)) / 900));
    result.sgd.wd = 1e-10;
    g_run.tr.add("pars epoch={d}", .{epoch});
    for (g_params_watch[0..g_params_watch_n]) |t| recTensor(g_run, "pars.param", t);
    return result;
}

var g_params_watch: [8][*c]c.ggml_tensor = undefined;
var g_params_watch_n: usize = 0;

/// Ports test-opt.cpp's `test_regression`, shortened, then again with the
/// paths it never takes: physical batches smaller than logical ones
/// (gradient accumulation through `ggml_opt_fit`), a shuffled dataset, a
/// validation split, and the progress output.
fn scenarioRegression(r: *Run, optim: c.enum_ggml_opt_optimizer_type, variant: u8) void {
    const a = r.api;
    const ndata_regression: i64 = if (variant == 0) 201 else 200;
    const nbatch_physical: i64 = if (variant == 0) ndata_regression else 10;
    const nbatch_logical: i64 = if (variant == 0) ndata_regression else 20;
    const val_split: f32 = if (variant == 0) 0.0 else 0.25;
    const silent = variant != 2;
    const nepoch: i64 = if (variant == 0) 12 else 3;

    var prng = std.Random.DefaultPrng.init(12345);
    const rand = prng.random();

    const dataset = a.dataset_init(c.GGML_TYPE_F32, c.GGML_TYPE_F32, 1, 1, ndata_regression, if (variant == 0) ndata_regression else 2);
    defer a.dataset_free(dataset);
    const data: [*]f32 = @ptrCast(c.ggml_get_data_f32(a.dataset_data(dataset)));
    const labels: [*]f32 = @ptrCast(c.ggml_get_data_f32(a.dataset_labels(dataset)));
    for (0..@intCast(ndata_regression)) |idata| {
        const x: f32 = -100.0 + 200.0 * @as(f32, @floatFromInt(idata)) / @as(f32, @floatFromInt(ndata_regression - 1));
        data[idata] = x;
        labels[idata] = 1.2 * x + 3.4 + 0.1 * (rand.float(f32) - 0.5);
    }

    const ctx_static = newCtx(3 * c.ggml_tensor_overhead());
    defer c.ggml_free(ctx_static);
    const ctx_compute = newCtx(computeCtxSize());
    defer c.ggml_free(ctx_compute);

    const x = c.ggml_new_tensor_2d(ctx_static, c.GGML_TYPE_F32, 1, nbatch_physical);
    _ = c.ggml_set_name(x, "x");
    const pa = c.ggml_new_tensor_1d(ctx_static, c.GGML_TYPE_F32, 1);
    _ = c.ggml_set_name(pa, "a");
    c.ggml_set_param(pa);
    const pb = c.ggml_new_tensor_1d(ctx_static, c.GGML_TYPE_F32, 1);
    _ = c.ggml_set_name(pb, "b");
    c.ggml_set_param(pb);
    const f = c.ggml_add(ctx_compute, c.ggml_mul(ctx_compute, x, pa), pb);
    _ = c.ggml_set_name(f, "f");

    const buf = c.ggml_backend_alloc_ctx_tensors(ctx_static, r.backend);
    defer c.ggml_backend_buffer_free(buf);
    const a0: f32 = 1.0;
    const b0: f32 = 3.0;
    c.ggml_backend_tensor_set(pa, &a0, 0, 4);
    c.ggml_backend_tensor_set(pb, &b0, 0, 4);

    g_params_watch[0] = pa;
    g_params_watch[1] = pb;
    g_params_watch_n = 2;
    defer g_params_watch_n = 0;

    const sched = r.newSched();
    defer r.freeSched();
    const text = captureStderr(r, struct {
        fn call(rr: *Run, args: anytype) void {
            rr.api.fit(args[0], args[1], args[2], args[3], args[4], c.GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR, args[5], &regressionOptPars, args[6], args[7], args[8], args[9]);
        }
    }.call, .{ sched, ctx_compute, x, f, dataset, optim, nepoch, nbatch_logical, val_split, silent });
    recText(r, "regression.stderr", text);

    recTensor(r, "regression.a", pa);
    recTensor(r, "regression.b", pb);
    if (variant != 0) {
        var perm_data: [200]f32 = undefined;
        var perm_labels: [200]f32 = undefined;
        a.dataset_get_batch_host(dataset, &perm_data, 200 * 4, &perm_labels, 0);
        r.tr.add("regression.perm d={x} l={x}", .{ hashBytes(std.mem.sliceAsBytes(&perm_data)), hashBytes(std.mem.sliceAsBytes(&perm_labels)) });
    }
}

// -----------------------------------------------------------------------------
// Beyond test-opt.cpp: cross-entropy, a two-layer network, dynamic graphs

const n_in: i64 = 4;
const n_hidden: i64 = 16;
const n_out: i64 = 3;
const n_mlp_data: i64 = 64;
const mlp_batch: i64 = 8;

const Mlp = struct {
    ctx_static: ?*c.ggml_context,
    buf: c.ggml_backend_buffer_t,
    inputs: [*c]c.ggml_tensor,
    w1: [*c]c.ggml_tensor,
    b1: [*c]c.ggml_tensor,
    w2: [*c]c.ggml_tensor,
    b2: [*c]c.ggml_tensor,
    dataset: c.ggml_opt_dataset_t,

    fn params(self: *const Mlp) [4][*c]c.ggml_tensor {
        return .{ self.w1, self.b1, self.w2, self.b2 };
    }
};

/// Builds the network's parameters, its input and a separable dataset —
/// the class is which of three projections of the input is largest.
fn mlpInit(r: *Run, with_inputs: bool) Mlp {
    const a = r.api;
    var m: Mlp = undefined;
    m.ctx_static = newCtx(8 * c.ggml_tensor_overhead());
    m.inputs = if (with_inputs) c.ggml_new_tensor_2d(m.ctx_static, c.GGML_TYPE_F32, n_in, mlp_batch) else null;
    if (with_inputs) _ = c.ggml_set_name(m.inputs, "inputs");
    m.w1 = c.ggml_new_tensor_2d(m.ctx_static, c.GGML_TYPE_F32, n_in, n_hidden);
    m.b1 = c.ggml_new_tensor_1d(m.ctx_static, c.GGML_TYPE_F32, n_hidden);
    m.w2 = c.ggml_new_tensor_2d(m.ctx_static, c.GGML_TYPE_F32, n_hidden, n_out);
    m.b2 = c.ggml_new_tensor_1d(m.ctx_static, c.GGML_TYPE_F32, n_out);
    const names = [_][*:0]const u8{ "w1", "b1", "w2", "b2" };
    for (m.params(), names) |p, name| {
        _ = c.ggml_set_name(p, name);
        c.ggml_set_param(p);
    }
    m.buf = c.ggml_backend_alloc_ctx_tensors(m.ctx_static, r.backend);

    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rand = prng.random();
    for (m.params()) |p| {
        const n: usize = @intCast(c.ggml_nelements(p));
        const vals = std.heap.c_allocator.alloc(f32, n) catch @panic("oom");
        defer std.heap.c_allocator.free(vals);
        for (vals) |*v| v.* = (rand.float(f32) - 0.5) * 0.8;
        c.ggml_backend_tensor_set(p, vals.ptr, 0, n * 4);
    }

    m.dataset = a.dataset_init(c.GGML_TYPE_F32, c.GGML_TYPE_F32, n_in, n_out, n_mlp_data, 2);
    const data: [*]f32 = @ptrCast(c.ggml_get_data_f32(a.dataset_data(m.dataset)));
    const labels: [*]f32 = @ptrCast(c.ggml_get_data_f32(a.dataset_labels(m.dataset)));
    for (0..@intCast(n_mlp_data)) |i| {
        var x: [n_in]f32 = undefined;
        for (&x) |*v| v.* = rand.float(f32) * 2 - 1;
        const s = [_]f32{ x[0] + x[1], x[2] - x[0], x[3] - x[1] };
        var best: usize = 0;
        for (s, 0..) |v, k| if (v > s[best]) {
            best = k;
        };
        for (0..n_in) |k| data[i * n_in + k] = x[k];
        for (0..n_out) |k| labels[i * n_out + k] = if (k == best) 1.0 else 0.0;
    }
    return m;
}

fn mlpFree(r: *Run, m: *Mlp) void {
    r.api.dataset_free(m.dataset);
    c.ggml_backend_buffer_free(m.buf);
    c.ggml_free(m.ctx_static);
}

fn mlpForward(m: *const Mlp, ctx: ?*c.ggml_context, inputs: [*c]c.ggml_tensor) [*c]c.ggml_tensor {
    const h1 = c.ggml_relu(ctx, c.ggml_add(ctx, c.ggml_mul_mat(ctx, m.w1, inputs), m.b1));
    const out = c.ggml_add(ctx, c.ggml_mul_mat(ctx, m.w2, h1), m.b2);
    _ = c.ggml_set_name(out, "logits");
    return out;
}

fn recParams(r: *Run, tag: []const u8, m: *const Mlp) void {
    for (m.params()) |p| recTensor(r, tag, p);
}

/// Records each callback's arguments and the result it is handed.
fn recordingCallback(train: bool, ctx: c.ggml_opt_context_t, dataset: c.ggml_opt_dataset_t, result: c.ggml_opt_result_t, ibatch: i64, ibatch_max: i64, t_start_us: i64) callconv(.c) void {
    _ = ctx;
    _ = dataset;
    g_run.tr.add("cb train={} ibatch={d}/{d} t_start>0={}", .{ train, ibatch, ibatch_max, t_start_us > 0 });
    recResult(g_run, "cb.result", result);
}

/// A cross-entropy classifier on static graphs, trained with
/// `ggml_opt_epoch`: predictions, correct counts, accuracies, a validation
/// split, gradient accumulation when `opt_period > 1`, the progress bar.
fn scenarioMlpStatic(r: *Run, optim: c.enum_ggml_opt_optimizer_type, opt_period: i32, progress: bool) void {
    const loss_type = staticClassifierLoss(r);
    const a = r.api;
    var m = mlpInit(r, true);
    defer mlpFree(r, &m);
    const ctx_compute = newCtx(computeCtxSize());
    defer c.ggml_free(ctx_compute);
    const outputs = mlpForward(&m, ctx_compute, m.inputs);

    var p = a.default_params(r.newSched(), loss_type);
    defer r.freeSched();
    p.ctx_compute = ctx_compute;
    p.inputs = m.inputs;
    p.outputs = outputs;
    p.opt_period = opt_period;
    p.optimizer = optim;
    var pars = a.get_default_optimizer_params(null);
    pars.adamw.alpha = 0.05;
    pars.adamw.wd = 0.01;
    pars.sgd.alpha = 0.2;
    pars.sgd.wd = 0.001;
    p.get_opt_pars = a.get_constant_optimizer_params;
    p.get_opt_pars_ud = &pars;
    const ctx = a.init(p);
    defer a.free(ctx);
    recContext(r, "mlp.ctx", ctx);

    const train = a.result_init();
    defer a.result_free(train);
    const val = a.result_init();
    defer a.result_free(val);

    for (0..4) |epoch| {
        a.dataset_shuffle(ctx, m.dataset, 48);
        a.result_reset(train);
        a.result_reset(val);
        if (progress) {
            const text = captureStderr(r, struct {
                fn call(rr: *Run, args: anytype) void {
                    rr.api.epoch(args[0], args[1], args[2], args[3], 48, rr.api.epoch_callback_progress_bar, rr.api.epoch_callback_progress_bar);
                }
            }.call, .{ ctx, m.dataset, train, val });
            recText(r, "mlp.progress", text);
        } else {
            a.epoch(ctx, m.dataset, train, val, 48, &recordingCallback, &recordingCallback);
        }
        r.tr.add("mlp epoch {d}", .{epoch});
        recResult(r, "mlp.train", train);
        recResult(r, "mlp.val", val);
        recParams(r, "mlp.param", &m);
        for (m.params()) |prm| recTensor(r, "mlp.grad_acc", a.grad_acc(ctx, prm));
    }
}

/// The same network on dynamic graphs: a fresh compute context and forward
/// graph every step, `ggml_opt_prepare_alloc`, forward-only and backward
/// steps mixed, with gradient accumulation.
fn scenarioMlpDynamic(r: *Run, optim: c.enum_ggml_opt_optimizer_type, loss_type: c.enum_ggml_opt_loss_type, opt_period: i32) void {
    const a = r.api;
    var m = mlpInit(r, false);
    defer mlpFree(r, &m);

    var p = a.default_params(r.newSched(), loss_type);
    defer r.freeSched();
    p.opt_period = opt_period;
    p.optimizer = optim;
    p.get_opt_pars = &testOptParsSmall;
    const ctx = a.init(p);
    defer a.free(ctx);
    r.tr.add("dyn static={}", .{a.static_graphs(ctx)});

    const res = a.result_init();
    defer a.result_free(res);

    var host_data: [mlp_batch * n_in]f32 = undefined;
    var host_labels: [mlp_batch * n_out]f32 = undefined;
    for (0..10) |step| {
        const ctx_compute = newCtx(computeCtxSize());
        defer c.ggml_free(ctx_compute);
        const inputs = c.ggml_new_tensor_2d(ctx_compute, c.GGML_TYPE_F32, n_in, mlp_batch);
        _ = c.ggml_set_name(inputs, "inputs");
        const outputs = mlpForward(&m, ctx_compute, inputs);
        const gf = c.ggml_new_graph_custom(ctx_compute, c.GGML_DEFAULT_GRAPH_SIZE, true);
        c.ggml_build_forward_expand(gf, outputs);

        a.prepare_alloc(ctx, ctx_compute, gf, inputs, outputs);
        const backward = step % 5 != 4;
        a.alloc(ctx, backward);
        recContext(r, "dyn.ctx", ctx);
        a.dataset_get_batch_host(m.dataset, &host_data, @sizeOf(@TypeOf(host_data)), &host_labels, @intCast(step % 8));
        c.ggml_backend_tensor_set(inputs, &host_data, 0, @sizeOf(@TypeOf(host_data)));
        const labels = a.labels(ctx);
        if (labels != null) c.ggml_backend_tensor_set(labels, &host_labels, 0, @sizeOf(@TypeOf(host_labels)));
        a.eval(ctx, res);
        r.tr.add("dyn step {d} backward={}", .{ step, backward });
        recResult(r, "dyn.result", res);
        recParams(r, "dyn.param", &m);
    }
}

fn testOptParsSmall(userdata: ?*anyopaque) callconv(.c) c.struct_ggml_opt_optimizer_params {
    var result = g_run.api.get_default_optimizer_params(userdata);
    result.adamw.alpha = 0.02;
    result.adamw.beta1 = 0.8;
    result.adamw.beta2 = 0.95;
    result.adamw.eps = 1e-6;
    result.adamw.wd = 0.05;
    result.sgd.alpha = 0.1;
    result.sgd.wd = 0.01;
    return result;
}

/// The loss a static-graph classifier can use on the backend under test.
///
/// With static graphs the loss, predictions and correct count live in the
/// static buffer on the scheduler's first backend, so they cannot move:
/// Metal has no `CROSS_ENTROPY_LOSS`, `ARGMAX` or `COUNT_EQUAL` kernel, and
/// the reference aborts with "pre-allocated tensor in a buffer that cannot
/// run the operation". Measured. On Metal the static scenarios therefore use
/// mean squared error over the same one-hot labels; cross-entropy runs there
/// through dynamic graphs, where those nodes are not pre-allocated.
fn staticClassifierLoss(r: *const Run) c.enum_ggml_opt_loss_type {
    return if (r.backend == r.cpu) c.GGML_OPT_LOSS_TYPE_CROSS_ENTROPY else c.GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR;
}

/// A context allocated for the forward pass only.
fn scenarioForwardOnly(r: *Run, optim: c.enum_ggml_opt_optimizer_type, loss_type_in: c.enum_ggml_opt_loss_type) void {
    const loss_type = if (loss_type_in == c.GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) staticClassifierLoss(r) else loss_type_in;
    const a = r.api;
    var m = mlpInit(r, true);
    defer mlpFree(r, &m);
    const ctx_compute = newCtx(computeCtxSize());
    defer c.ggml_free(ctx_compute);
    const outputs = mlpForward(&m, ctx_compute, m.inputs);
    var p = a.default_params(r.newSched(), loss_type);
    defer r.freeSched();
    p.ctx_compute = ctx_compute;
    p.inputs = m.inputs;
    p.outputs = outputs;
    p.build_type = c.GGML_OPT_BUILD_TYPE_FORWARD;
    p.optimizer = optim;
    const ctx = a.init(p);
    defer a.free(ctx);
    recContext(r, "fwd.ctx", ctx);
    const res = a.result_init();
    defer a.result_free(res);
    a.epoch(ctx, m.dataset, null, res, 0, &recordingCallback, &recordingCallback);
    recResult(r, "fwd.result", res);
    recParams(r, "fwd.param", &m);
}

// -----------------------------------------------------------------------------
// Getters, defaults, empty results

fn scenarioMisc(r: *Run) void {
    const a = r.api;
    for ([_]c.enum_ggml_opt_optimizer_type{ 0, 1, 2, 3, 99, 0xffffffff }) |o| {
        r.tr.add("optimizer_name({d}) = {s}", .{ o, std.mem.span(a.optimizer_name(o)) });
    }
    const d = a.get_default_optimizer_params(null);
    r.tr.add("default pars bytes={x}", .{std.mem.asBytes(&d)});
    var k = d;
    k.adamw.alpha = 0.123;
    k.sgd.wd = 0.456;
    const kk = a.get_constant_optimizer_params(&k);
    r.tr.add("constant pars bytes={x}", .{std.mem.asBytes(&kk)});
    for ([_]c.enum_ggml_opt_loss_type{ 0, 1, 2, 3 }) |lt| {
        const p = a.default_params(null, lt);
        const got = p.get_opt_pars.?(null);
        r.tr.add("default_params lt={d} sched={} ctx={} in={} out={} loss={d} build={d} period={d} ud={} opt={d} pars={x}", .{
            lt, p.backend_sched == null, p.ctx_compute == null, p.inputs == null, p.outputs == null, p.loss_type, p.build_type, p.opt_period, p.get_opt_pars_ud == null, p.optimizer, std.mem.asBytes(&got),
        });
    }
    const res = a.result_init();
    defer a.result_free(res);
    recResult(r, "empty", res);
    a.result_reset(res);
    recResult(r, "empty.reset", res);
    a.result_free(null);
}

// -----------------------------------------------------------------------------
// The shuffle, through datasets and directly against libc++

fn scenarioShuffle(r: *Run) void {
    const a = r.api;
    // A context is only the engine's holder here; a forward-only one on a
    // one-node graph is the cheapest that `ggml_opt_init` accepts.
    const p = a.default_params(r.newSched(), c.GGML_OPT_LOSS_TYPE_SUM);
    defer r.freeSched();
    const ctx = a.init(p);
    defer a.free(ctx);

    const sizes = [_]i64{ 1, 2, 3, 7, 64, 1000, 4099, 70001 };
    for (sizes) |n| {
        const shard: i64 = if (@rem(n, 7) == 0) 7 else 1;
        const ds = a.dataset_init(c.GGML_TYPE_I32, c.GGML_TYPE_I32, 1, 0, n, shard);
        defer a.dataset_free(ds);
        const data: [*]i32 = @ptrCast(@alignCast(c.ggml_get_data(a.dataset_data(ds))));
        for (0..@intCast(n)) |i| data[i] = @intCast(i);
        const out = std.heap.c_allocator.alloc(i32, @intCast(n)) catch @panic("oom");
        defer std.heap.c_allocator.free(out);
        const partials = [_]i64{ -1, n, 0, shard, @divTrunc(n, 2) - @rem(@divTrunc(n, 2), shard), -1 };
        for (partials) |idata| {
            a.dataset_shuffle(ctx, ds, idata);
            a.dataset_get_batch_host(ds, out.ptr, @intCast(n * 4), null, 0);
            r.tr.add("shuffle n={d} shard={d} idata={d} h={x} first={d}", .{ n, shard, idata, hashBytes(std.mem.sliceAsBytes(out)), out[0] });
        }
    }
}

/// Compares `opt.uniformInt` and `opt.shuffle` with libc++'s templates.
/// Return: checks made; aborts the run on the first difference.
fn libcxxOracle() usize {
    var checks: usize = 0;
    const ranges = [_][2]i64{
        .{ 0, 1 },                    .{ 0, 2 },                     .{ 0, 3 },                .{ 0, 5 },
        .{ 0, 63 },                   .{ 0, 64 },                    .{ 0, 65 },               .{ -10, 10 },
        .{ 0, (1 << 16) - 1 },        .{ 0, 1 << 16 },               .{ 0, (1 << 31) - 1 },    .{ 0, 1 << 31 },
        .{ 0, (1 << 32) - 2 },        .{ 0, (1 << 32) - 1 },         .{ 0, 1 << 32 },          .{ 0, (1 << 33) + 12345 },
        .{ 0, (1 << 40) + 7 },        .{ 0, (1 << 62) + 1 },         .{ -(1 << 62), 1 << 62 }, .{ std.math.minInt(i64), std.math.maxInt(i64) },
        .{ 0, std.math.maxInt(i64) }, .{ std.math.minInt(i64), -1 }, .{ 7, 7 },                .{ -3, -3 },
    };
    var want: [512]i64 = undefined;
    var got: [512]i64 = undefined;
    for (ranges) |rg| {
        for ([_]u32{ 5489, 1, 0xdeadbeef }) |seed| {
            const tail_ref = opt_ref_uniform(seed, rg[0], rg[1], &want, want.len);
            var g = opt.Mt19937.init(seed);
            for (&got) |*v| v.* = opt.uniformInt(&g, rg[0], rg[1]);
            const tail_port = g.next();
            for (want, got, 0..) |w, p, i| {
                checks += 1;
                if (w != p) {
                    std.debug.print("FAIL uniform [{d}, {d}] seed {d} draw {d}: libc++ {d}, port {d}\n", .{ rg[0], rg[1], seed, i, w, p });
                    std.process.exit(1);
                }
            }
            checks += 1;
            if (tail_ref != tail_port) {
                std.debug.print("FAIL uniform [{d}, {d}] seed {d}: engine left at a different word\n", .{ rg[0], rg[1], seed });
                std.process.exit(1);
            }
        }
    }
    var n: usize = 0;
    while (n <= 300) : (n += if (n < 40) 1 else 37) {
        for ([_]usize{ 1, 3 }) |reps| {
            var v_ref: [300]i64 = undefined;
            var v_port: [300]i64 = undefined;
            for (0..n) |i| {
                v_ref[i] = @intCast(i);
                v_port[i] = @intCast(i);
            }
            const tail_ref = opt_ref_shuffle(5489, &v_ref, n, reps);
            var g = opt.Mt19937.init(5489);
            for (0..reps) |_| opt.shuffle(v_port[0..n], &g);
            const tail_port = g.next();
            checks += n + 1;
            if (!std.mem.eql(i64, v_ref[0..n], v_port[0..n]) or tail_ref != tail_port) {
                std.debug.print("FAIL shuffle n={d} reps={d}\n", .{ n, reps });
                std.process.exit(1);
            }
        }
    }
    return checks;
}

// -----------------------------------------------------------------------------
// Assertions

/// Runs `body` in a child and reports whether it died of SIGABRT.
fn aborts(body: *const fn (api: *const Api, sched: c.ggml_backend_sched_t) void, api: *const Api, sched: c.ggml_backend_sched_t) bool {
    const pid = fork();
    if (pid == 0) {
        const devnull = open("/dev/null", 1);
        _ = dup2(devnull, 2);
        body(api, sched);
        _exit(0);
    }
    var status: c_int = 0;
    _ = waitpid(pid, &status, 0);
    return (status & 0x7f) == 6;
}

const abort_cases = .{
    .{ "dataset_init ne_datapoint=0", struct {
        fn f(a: *const Api, _: c.ggml_backend_sched_t) void {
            _ = a.dataset_init(c.GGML_TYPE_F32, c.GGML_TYPE_F32, 0, 1, 4, 1);
        }
    }.f },
    .{ "dataset_init ne_label=-1", struct {
        fn f(a: *const Api, _: c.ggml_backend_sched_t) void {
            _ = a.dataset_init(c.GGML_TYPE_F32, c.GGML_TYPE_F32, 1, -1, 4, 1);
        }
    }.f },
    .{ "dataset_init ndata=0", struct {
        fn f(a: *const Api, _: c.ggml_backend_sched_t) void {
            _ = a.dataset_init(c.GGML_TYPE_F32, c.GGML_TYPE_F32, 1, 1, 0, 1);
        }
    }.f },
    .{ "dataset_init ndata_shard=0", struct {
        fn f(a: *const Api, _: c.ggml_backend_sched_t) void {
            _ = a.dataset_init(c.GGML_TYPE_F32, c.GGML_TYPE_F32, 1, 1, 4, 0);
        }
    }.f },
    .{ "dataset_init ok", struct {
        fn f(a: *const Api, _: c.ggml_backend_sched_t) void {
            a.dataset_free(a.dataset_init(c.GGML_TYPE_F32, c.GGML_TYPE_F32, 1, 0, 4, 1));
        }
    }.f },
    .{ "shuffle idata > ndata", struct {
        fn f(a: *const Api, s: c.ggml_backend_sched_t) void {
            const ctx = a.init(a.default_params(s, 0));
            a.dataset_shuffle(ctx, a.dataset_init(c.GGML_TYPE_F32, c.GGML_TYPE_F32, 1, 0, 4, 1), 5);
        }
    }.f },
    .{ "shuffle idata % shard", struct {
        fn f(a: *const Api, s: c.ggml_backend_sched_t) void {
            const ctx = a.init(a.default_params(s, 0));
            a.dataset_shuffle(ctx, a.dataset_init(c.GGML_TYPE_F32, c.GGML_TYPE_F32, 1, 0, 4, 2), 3);
        }
    }.f },
    .{ "get_batch_host labels mismatch", struct {
        fn f(a: *const Api, _: c.ggml_backend_sched_t) void {
            var buf: [16]f32 = undefined;
            a.dataset_get_batch_host(a.dataset_init(c.GGML_TYPE_F32, c.GGML_TYPE_F32, 1, 0, 4, 1), &buf, 16, &buf, 0);
        }
    }.f },
    .{ "get_batch_host past the end", struct {
        fn f(a: *const Api, _: c.ggml_backend_sched_t) void {
            var buf: [16]f32 = undefined;
            a.dataset_get_batch_host(a.dataset_init(c.GGML_TYPE_F32, c.GGML_TYPE_F32, 1, 0, 4, 1), &buf, 8, null, 2);
        }
    }.f },
    .{ "get_batch_host partial shard", struct {
        fn f(a: *const Api, _: c.ggml_backend_sched_t) void {
            var buf: [16]f32 = undefined;
            a.dataset_get_batch_host(a.dataset_init(c.GGML_TYPE_F32, c.GGML_TYPE_F32, 1, 0, 4, 2), &buf, 4, null, 0);
        }
    }.f },
    .{ "init opt_period=0", struct {
        fn f(a: *const Api, s: c.ggml_backend_sched_t) void {
            var p = a.default_params(s, 0);
            p.opt_period = 0;
            _ = a.init(p);
        }
    }.f },
    .{ "init dynamic with outputs", struct {
        fn f(a: *const Api, s: c.ggml_backend_sched_t) void {
            var t: c.ggml_tensor = std.mem.zeroes(c.ggml_tensor);
            var p = a.default_params(s, 0);
            p.outputs = &t;
            _ = a.init(p);
        }
    }.f },
    .{ "eval without alloc", struct {
        fn f(a: *const Api, s: c.ggml_backend_sched_t) void {
            a.eval(a.init(a.default_params(s, 0)), null);
        }
    }.f },
    .{ "epoch on dynamic graphs", struct {
        fn f(a: *const Api, s: c.ggml_backend_sched_t) void {
            a.epoch(a.init(a.default_params(s, 0)), null, null, null, -1, null, null);
        }
    }.f },
    .{ "alloc dynamic without prepare", struct {
        fn f(a: *const Api, s: c.ggml_backend_sched_t) void {
            a.alloc(a.init(a.default_params(s, 0)), true);
        }
    }.f },
    .{ "prepare_alloc on static graphs", struct {
        fn f(a: *const Api, s: c.ggml_backend_sched_t) void {
            const ctx_static = newCtx(4 * c.ggml_tensor_overhead());
            const in = c.ggml_new_tensor_1d(ctx_static, c.GGML_TYPE_F32, 1);
            const w = c.ggml_new_tensor_1d(ctx_static, c.GGML_TYPE_F32, 1);
            c.ggml_set_param(w);
            _ = c.ggml_backend_alloc_ctx_tensors(ctx_static, c.ggml_backend_sched_get_backend(s, 0));
            const cc = newCtx(computeCtxSize());
            var p = a.default_params(s, 1);
            p.ctx_compute = cc;
            p.inputs = in;
            p.outputs = c.ggml_add(cc, in, w);
            const ctx = a.init(p);
            a.prepare_alloc(ctx, cc, null, null, null);
        }
    }.f },
};

fn abortChecks(cpu: c.ggml_backend_t) usize {
    var backends: [1]c.ggml_backend_t = .{cpu};
    const sched = c.ggml_backend_sched_new(&backends, null, 1, c.GGML_DEFAULT_GRAPH_SIZE, false, true);
    defer c.ggml_backend_sched_free(sched);
    var checks: usize = 0;
    inline for (abort_cases) |case| {
        const ref = aborts(case[1], &ref_api, sched);
        const port = aborts(case[1], &port_api, sched);
        checks += 1;
        if (ref != port) {
            std.debug.print("FAIL abort '{s}': reference {s}, port {s}\n", .{ case[0], if (ref) "aborts" else "returns", if (port) "aborts" else "returns" });
            std.process.exit(1);
        }
        if (!ref and !std.mem.endsWith(u8, case[0], "ok")) {
            std.debug.print("FAIL abort '{s}': the reference did not abort, so the case tests nothing\n", .{case[0]});
            std.process.exit(1);
        }
    }
    return checks;
}

// -----------------------------------------------------------------------------
// stderr capture

/// Runs `body` with fd 2 pointed at a temporary file and returns what was
/// written, with the wall-clock fields masked — `t=`, `ETA=` and `took` are
/// the only text that depends on how long a step took.
///
/// `OPT_DIFF_NOCAPTURE=1` runs `body` without capturing, so an abort inside
/// it can say why.
fn captureStderr(r: *Run, comptime body: anytype, args: anytype) []const u8 {
    if (std.c.getenv("OPT_DIFF_NOCAPTURE") != null) {
        body(r, args);
        return "";
    }
    var name = "/tmp/opt_diff_XXXXXX".*;
    const fd = mkstemp(&name);
    if (fd < 0) @panic("mkstemp");
    _ = unlink(&name);
    _ = fflush(__stderrp);
    const saved = dup(2);
    _ = dup2(fd, 2);
    body(r, args);
    _ = fflush(__stderrp);
    _ = dup2(saved, 2);
    _ = close(saved);
    _ = lseek(fd, 0, 0);
    var out: std.ArrayList(u8) = .empty;
    var chunk: [4096]u8 = undefined;
    while (true) {
        const got = read(fd, &chunk, chunk.len);
        if (got <= 0) break;
        out.appendSlice(r.tr.arena, chunk[0..@intCast(got)]) catch @panic("oom");
    }
    _ = close(fd);
    maskTimes(out.items);
    return out.items;
}

fn maskTimes(text: []u8) void {
    for ([_][]const u8{ "t=", "ETA=", "took " }) |key| {
        var i: usize = 0;
        while (std.mem.findPos(u8, text, i, key)) |at| {
            var j = at + key.len;
            var seen: usize = 0;
            while (j < text.len and seen < 8 and (std.ascii.isDigit(text[j]) or text[j] == ':')) : (j += 1) {
                if (std.ascii.isDigit(text[j])) text[j] = '#';
                seen += 1;
            }
            i = at + key.len;
        }
    }
}

fn recText(r: *Run, tag: []const u8, text: []const u8) void {
    var it = std.mem.splitAny(u8, text, "\r\n");
    var n: usize = 0;
    while (it.next()) |line| : (n += 1) {
        // `ggml_opt_fit` prints its own `__func__`, which the `-D` rename
        // turns into `ref_ggml_opt_fit` on the reference side only.
        const fixed = if (std.mem.startsWith(u8, line, "ref_")) line[4..] else line;
        r.tr.add("{s}[{d}]: {s}", .{ tag, n, fixed });
    }
}

// -----------------------------------------------------------------------------
// Driver

const Backend = enum { cpu, metal };

/// With `OPT_DIFF_VERBOSE` set, names each scenario on stderr as it starts,
/// so an abort says where it happened.
fn mark(r: *Run, name: []const u8, optim: c.enum_ggml_opt_optimizer_type) void {
    if (std.c.getenv("OPT_DIFF_VERBOSE") != null) {
        std.debug.print("[{s}] {s} optim={d} nodes={}\n", .{ if (r.api == &ref_api) "ref" else "port", name, optim, r.nodes });
    }
}

fn runScenarios(r: *Run) void {
    const optims = [_]c.enum_ggml_opt_optimizer_type{ c.GGML_OPT_OPTIMIZER_TYPE_ADAMW, c.GGML_OPT_OPTIMIZER_TYPE_SGD };
    if (!r.nodes) {
        mark(r, "Misc", 0);
        scenarioMisc(r);
        mark(r, "Shuffle", 0);
        scenarioShuffle(r);
    }
    for (optims) |optim| {
        r.tr.add("== optimizer {d}", .{optim});
        if (!r.nodes) {
            mark(r, "Dataset", optim);
            scenarioDataset(r, optim, false);
            mark(r, "Dataset", optim);
            scenarioDataset(r, optim, true);
            mark(r, "EpochVsFit", optim);
            scenarioEpochVsFit(r, optim);
            mark(r, "IdataSplit", optim);
            scenarioIdataSplit(r, optim, false);
            mark(r, "IdataSplit", optim);
            scenarioIdataSplit(r, optim, true);
            for ([_]u8{ 0, 1, 2 }) |v| {
                mark(r, "Regression", optim);
                scenarioRegression(r, optim, v);
            }
            mark(r, "MlpStatic", optim);
            scenarioMlpStatic(r, optim, 1, true);
        }
        mark(r, "Grad", optim);
        scenarioGrad(r, optim);
        mark(r, "ForwardBackward", optim);
        scenarioForwardBackward(r, optim, false, false);
        mark(r, "ForwardBackward", optim);
        scenarioForwardBackward(r, optim, true, false);
        mark(r, "ForwardBackward", optim);
        scenarioForwardBackward(r, optim, true, true);
        for ([_]i64{ 2, 1 }) |nbp| {
            for ([_]c.enum_ggml_opt_loss_type{ c.GGML_OPT_LOSS_TYPE_SUM, c.GGML_OPT_LOSS_TYPE_MEAN }) |lt| {
                mark(r, "GradAccum", optim);
                scenarioGradAccum(r, optim, nbp, lt);
            }
        }
        mark(r, "MlpStatic", optim);
        scenarioMlpStatic(r, optim, 1, false);
        mark(r, "MlpStatic", optim);
        scenarioMlpStatic(r, optim, 2, false);
        mark(r, "MlpDynamic", optim);
        scenarioMlpDynamic(r, optim, c.GGML_OPT_LOSS_TYPE_CROSS_ENTROPY, 1);
        mark(r, "MlpDynamic", optim);
        scenarioMlpDynamic(r, optim, c.GGML_OPT_LOSS_TYPE_CROSS_ENTROPY, 3);
        mark(r, "MlpDynamic", optim);
        scenarioMlpDynamic(r, optim, c.GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR, 2);
        mark(r, "ForwardOnly", optim);
        scenarioForwardOnly(r, optim, c.GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
        mark(r, "ForwardOnly", optim);
        scenarioForwardOnly(r, optim, c.GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR);
    }
}

pub fn main() !void {
    _ = setenv("GGML_NO_BACKTRACE", "1", 1);
    c.ggml_log_set(&quietLog, null);

    const cpu_dev = c.ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_CPU);
    const gpu_dev = c.ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_GPU);
    const cpu = c.ggml_backend_dev_init(cpu_dev, null);
    c.ggml_backend_cpu_set_n_threads(cpu, 4);

    var total: usize = 0;

    const oracle = libcxxOracle();
    std.debug.print("  libc++ shuffle/uniform_int: {d} checks\n", .{oracle});
    total += oracle;

    // Before Metal is initialised: the children fork from this process.
    const aborted = abortChecks(cpu);
    std.debug.print("  assertions: {d} cases\n", .{aborted});
    total += aborted;

    const metal = if (gpu_dev != null) c.ggml_backend_dev_init(gpu_dev, null) else null;

    for ([_]Backend{ .cpu, .metal }) |which| {
        const backend = if (which == .cpu) cpu else metal;
        if (backend == null) {
            std.debug.print("FAIL: no Metal device\n", .{});
            std.process.exit(1);
        }
        for ([_]bool{ false, true }) |nodes| {
            var arenas: [2]std.heap.ArenaAllocator = .{ .init(std.heap.c_allocator), .init(std.heap.c_allocator) };
            defer for (&arenas) |*ar| ar.deinit();
            var traces: [2]Trace = .{ .{ .arena = arenas[0].allocator() }, .{ .arena = arenas[1].allocator() } };
            const apis = [_]*const Api{ &ref_api, &port_api };
            for (apis, &traces) |api, *tr| {
                var r: Run = .{ .api = api, .tr = tr, .backend = backend, .cpu = cpu, .nodes = nodes };
                g_run = &r;
                runScenarios(&r);
            }
            const want = traces[0].lines.items;
            const got = traces[1].lines.items;
            // `OPT_DIFF_DUMP=1` prints the port's trace, to see what is compared.
            if (std.c.getenv("OPT_DIFF_DUMP") != null) {
                for (got) |line| std.debug.print("{s} {} | {s}\n", .{ @tagName(which), nodes, line });
            }
            const n = @min(want.len, got.len);
            for (0..n) |i| {
                if (!std.mem.eql(u8, want[i], got[i])) {
                    std.debug.print("FAIL {s} nodes={}: record {d} differs\n", .{ @tagName(which), nodes, i });
                    const from = if (i >= 3) i - 3 else 0;
                    for (from..i) |k| std.debug.print("   ctx  {s}\n", .{want[k]});
                    std.debug.print("   ref  {s}\n   port {s}\n", .{ want[i], got[i] });
                    std.process.exit(1);
                }
            }
            if (want.len != got.len) {
                std.debug.print("FAIL {s} nodes={}: {d} records against {d}\n", .{ @tagName(which), nodes, want.len, got.len });
                std.process.exit(1);
            }
            std.debug.print("  {s}, node trace {s}: {d} records identical\n", .{ @tagName(which), if (nodes) "on " else "off", want.len });
            total += want.len;
        }
    }

    c.ggml_backend_free(metal);
    c.ggml_backend_free(cpu);
    std.debug.print("opt-diff: {d} checks, all identical\n", .{total});
}

/// Drops ggml's log output, except errors, which are printed: an abort
/// from inside a backend logs its reason rather than asserting.
fn quietLog(level: c.enum_ggml_log_level, text: [*c]const u8, ud: ?*anyopaque) callconv(.c) void {
    _ = ud;
    if (level == c.GGML_LOG_LEVEL_ERROR) std.debug.print("{s}", .{std.mem.span(text)});
}
