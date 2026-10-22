//! Compares `src/llama/loader.zig` and `src/llama/model.zig` with
//! llama.cpp's `llama-model-loader.cpp` and `llama-model.cpp`.
//!
//! **Not a port.** The harness behind `scripts/loader-diff`. It runs the
//! metadata half of `llama_model_load` twice per file in one process --
//! through the C++ in `libllama.a` (`harness/loader_ref.cpp`) and through
//! `model.loadMetadata` -- and compares:
//!
//! - **everything logged**: the loader's metadata dump and type counts, its
//!   `print_info`, override and warning lines, the model's `print_info` and
//!   the vocabulary's; or, for a file that must fail, the message;
//! - **every `llama_hparams` member**, the C++ object read through the
//!   port's layout-identical `Hparams` once every offset and size has been
//!   checked against `offsetof` and `sizeof`;
//! - the model's size class, architecture, name, description, element and
//!   byte counts, file type, classifier labels, and `gguf_kv` in its
//!   `std::unordered_map` order.
//!
//! An architecture the port has no `load_arch_hparams` for loads only as a
//! vocabulary; past that, the port's log must be a prefix of the C++'s and
//! nothing else is compared.
//!
//! Inputs are upstream's vocabulary files, any Qwen3.5 model present, and
//! files synthesized into the directory given: Qwen3.5's metadata without
//! its tensors and edited one key at a time, small files with real tensors
//! for the type guess and the bounds check, and split sets.
//!
//! Usage: loader_diff <tmpdir> <file.gguf>...

const std = @import("std");
const harness_root = @import("harness_root");

const c = harness_root.impl.c;
const Hparams = harness_root.hparams.Hparams;
const M = harness_root.model;

const gpa = std.heap.c_allocator;

extern fn ref_load_metadata(path: [*:0]const u8, overrides: ?[*]const c.llama_model_kv_override, vocab_only: c_int, what: [*]u8, n: usize) ?*c.llama_model;
extern fn ref_free(model: *c.llama_model) void;
extern fn ref_hparams(model: *const c.llama_model) *const anyopaque;
extern fn ref_type(model: *const c.llama_model) c_int;
extern fn ref_arch(model: *const c.llama_model) c_int;
extern fn ref_name(model: *const c.llama_model) [*:0]const u8;
extern fn ref_desc(model: *const c.llama_model, buf: [*]u8, n: usize) void;
extern fn ref_n_elements(model: *const c.llama_model) u64;
extern fn ref_n_bytes(model: *const c.llama_model) usize;
extern fn ref_ftype(model: *const c.llama_model) c_int;
extern fn ref_n_labels(model: *const c.llama_model) usize;
extern fn ref_label(model: *const c.llama_model, i: usize) [*:0]const u8;
extern fn ref_n_kv(model: *const c.llama_model) usize;
extern fn ref_kv(model: *const c.llama_model, i: usize, key: *[*:0]const u8, value: *[*:0]const u8) c_int;
extern fn ref_hparams_sizeof() usize;
extern fn ref_hparams_alignof() usize;
extern fn ref_hparams_nfields() usize;
extern fn ref_hparams_field(i: usize, offset: *usize, size: *usize) [*:0]const u8;
extern "c" fn truncate(path: [*:0]const u8, length: i64) c_int;
extern fn ref_layer_tensor(model: *const c.llama_model, il: c_int, field: [*:0]const u8) ?[*:0]const u8;
extern fn ref_model_tensor(model: *const c.llama_model, field: [*:0]const u8) ?[*:0]const u8;
extern fn ref_n_layers(model: *const c.llama_model) usize;
extern fn ref_n_tensors(model: *const c.llama_model) usize;
extern fn ref_tensor_at(model: *const c.llama_model, i: usize, name: *[*:0]const u8) *c.ggml_tensor;
extern fn ref_n_devices(model: *const c.llama_model) usize;
extern fn ref_device(model: *const c.llama_model, i: usize) c.ggml_backend_dev_t;
extern fn ref_dev_layer(model: *const c.llama_model, il: c_int) c.ggml_backend_dev_t;
extern fn ref_dev_output(model: *const c.llama_model) c.ggml_backend_dev_t;
extern fn ref_memory_breakdown(model: *const c.llama_model, i: usize, buft: *c.ggml_backend_buffer_type_t) usize;
extern fn ref_hash_tensor(t: *const c.ggml_tensor) u64;

var checks: usize = 0;
var failures: usize = 0;
var case_failures: usize = 0;

fn fail(comptime fmt: []const u8, args: anytype) void {
    failures += 1;
    case_failures += 1;
    if (case_failures <= 6) std.debug.print("FAIL " ++ fmt ++ "\n", args);
}

fn expectStr(what: []const u8, ref: []const u8, port: []const u8) void {
    checks += 1;
    if (!std.mem.eql(u8, ref, port)) fail("{s}: ref '{s}' port '{s}'", .{ what, ref, port });
}

fn expectInt(what: []const u8, ref: i128, port: i128) void {
    checks += 1;
    if (ref != port) fail("{s}: ref {d} port {d}", .{ what, ref, port });
}

// -----------------------------------------------------------------------------
// Logs

var log_target: ?*std.ArrayList(u8) = null;

fn logCb(_: c.enum_ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    const t = log_target orelse return;
    t.appendSlice(gpa, std.mem.span(@as([*:0]const u8, @ptrCast(text)))) catch unreachable;
}

/// The first line where two logs differ, for the report.
fn firstDiff(a: []const u8, b: []const u8) void {
    var ia = std.mem.splitScalar(u8, a, '\n');
    var ib = std.mem.splitScalar(u8, b, '\n');
    var line: usize = 1;
    while (true) : (line += 1) {
        const la = ia.next();
        const lb = ib.next();
        if (la == null and lb == null) return;
        if (la == null or lb == null or !std.mem.eql(u8, la.?, lb.?)) {
            fail("log line {d}:\n    ref  '{s}'\n    port '{s}'", .{ line, la orelse "<end>", lb orelse "<end>" });
            return;
        }
    }
}

// -----------------------------------------------------------------------------
// The layout

/// Every `llama_hparams` member's name, offset and size against the port's
/// `Hparams`, and the struct's size and alignment. The comparison of
/// values below reads the C++ object through `Hparams`, so it means nothing
/// unless this passes.
fn checkLayout() bool {
    @setEvalBranchQuota(20000);
    const before = failures;
    const names = @typeInfo(Hparams).@"struct".field_names;
    expectInt("sizeof(llama_hparams)", ref_hparams_sizeof(), @sizeOf(Hparams));
    expectInt("alignof(llama_hparams)", ref_hparams_alignof(), @alignOf(Hparams));
    expectInt("llama_hparams members", ref_hparams_nfields(), names.len);
    const n = @min(ref_hparams_nfields(), names.len);
    inline for (names, 0..) |name, i| {
        if (i < n) {
            var off: usize = 0;
            var size: usize = 0;
            const ref_name_ = std.mem.span(ref_hparams_field(i, &off, &size));
            expectStr("member name", ref_name_, name);
            expectInt(name ++ " offset", off, @offsetOf(Hparams, name));
            expectInt(name ++ " size", size, @sizeOf(@FieldType(Hparams, name)));
        }
    }
    return failures == before;
}

fn compareHparams(ref: *const Hparams, port: *const Hparams) void {
    @setEvalBranchQuota(20000);
    inline for (@typeInfo(Hparams).@"struct".field_names) |name| {
        checks += 1;
        const a = std.mem.asBytes(&@field(ref.*, name));
        const b = std.mem.asBytes(&@field(port.*, name));
        if (!std.mem.eql(u8, a, b)) {
            const at = std.mem.findDiff(u8, a, b) orelse 0;
            fail("hparams.{s}: first difference at byte {d} (ref 0x{x:0>2}, port 0x{x:0>2})", .{ name, at, a[at], b[at] });
        }
    }
}

// -----------------------------------------------------------------------------
// One case

const Case = struct {
    name: []const u8,
    path: [:0]const u8,
    vocab_only: bool,
    overrides: []const c.llama_model_kv_override = &.{},
};

const Tally = struct {
    loaded: usize = 0,
    thrown: usize = 0,
    unported: usize = 0,
};
var tally: Tally = .{};

fn runCase(cs: Case) void {
    case_failures = 0;
    const before = checks;

    var ovr: std.ArrayList(c.llama_model_kv_override) = .empty;
    defer ovr.deinit(gpa);
    if (cs.overrides.len != 0) {
        ovr.appendSlice(gpa, cs.overrides) catch unreachable;
        ovr.append(gpa, std.mem.zeroes(c.llama_model_kv_override)) catch unreachable;
    }
    const ovr_ptr: ?[*]const c.llama_model_kv_override = if (cs.overrides.len != 0) ovr.items.ptr else null;

    var ref_log: std.ArrayList(u8) = .empty;
    defer ref_log.deinit(gpa);
    var port_log: std.ArrayList(u8) = .empty;
    defer port_log.deinit(gpa);

    // Reference.
    var what: [1024]u8 = undefined;
    log_target = &ref_log;
    const ref_model = ref_load_metadata(cs.path.ptr, ovr_ptr, @intFromBool(cs.vocab_only), &what, what.len);
    defer if (ref_model) |m| ref_free(m);

    // Port.
    log_target = &port_log;
    var exc: harness_root.impl.Exception = .{};
    var splits: std.ArrayList([:0]const u8) = .empty;
    defer {
        for (splits.items) |s| gpa.free(s);
        splits.deinit(gpa);
    }
    var params = M.defaultParams();
    params.vocab_only = cs.vocab_only;
    var loaded = M.loadMetadata(.{ .path = cs.path }, &splits, params, ovr_ptr, &exc);
    defer if (loaded == .model) loaded.model.deinit();
    log_target = null;

    // An empty log is a prefix of anything and equal to another empty one,
    // so a model that loaded, or got as far as unported, must have logged.
    // A constructor that throws first legitimately logs nothing; its message
    // carries that comparison.
    if (loaded != .thrown) {
        checks += 1;
        if (ref_log.items.len == 0 or port_log.items.len == 0) {
            fail("{s}: a side logged nothing (ref {d} bytes, port {d})", .{ cs.name, ref_log.items.len, port_log.items.len });
        }
    }

    switch (loaded) {
        .unported => {
            tally.unported += 1;
            checks += 1;
            if (!std.mem.startsWith(u8, ref_log.items, port_log.items)) {
                fail("{s}: an unported architecture's log is not a prefix of the C++'s", .{cs.name});
                firstDiff(ref_log.items, port_log.items);
            }
        },
        .thrown => {
            tally.thrown += 1;
            checks += 1;
            if (ref_model != null) {
                fail("{s}: the port threw '{s}', the C++ loaded", .{ cs.name, exc.what() });
            } else {
                expectStr("error", std.mem.sliceTo(&what, 0), exc.what());
            }
            firstDiff(ref_log.items, port_log.items);
        },
        .model => |*m| {
            tally.loaded += 1;
            checks += 1;
            const rm = ref_model orelse {
                fail("{s}: the C++ threw '{s}', the port loaded", .{ cs.name, std.mem.sliceTo(&what, 0) });
                return;
            };
            firstDiff(ref_log.items, port_log.items);
            compareHparams(@ptrCast(@alignCast(ref_hparams(rm))), &m.hparams);
            expectInt("type", ref_type(rm), @backingInt(m.type));
            expectInt("arch", ref_arch(rm), @backingInt(m.arch));
            expectStr("name", std.mem.span(ref_name(rm)), m.name.items);
            var desc: [512]u8 = undefined;
            ref_desc(rm, &desc, desc.len);
            expectStr("desc", std.mem.sliceTo(&desc, 0), m.desc());
            expectInt("n_elements", ref_n_elements(rm), m.n_elements);
            expectInt("n_bytes", ref_n_bytes(rm), m.n_bytes);
            expectInt("ftype", ref_ftype(rm), m.ftype);
            expectInt("classifier labels", ref_n_labels(rm), m.classifier_labels.items.len);
            for (m.classifier_labels.items, 0..) |l, i| {
                if (i < ref_n_labels(rm)) expectStr("classifier label", std.mem.span(ref_label(rm, i)), l);
            }
            expectInt("gguf_kv size", ref_n_kv(rm), m.gguf_kv.len);
            var it = m.gguf_kv.iterator();
            var i: usize = 0;
            while (it.next()) |e| : (i += 1) {
                var k: [*:0]const u8 = undefined;
                var v: [*:0]const u8 = undefined;
                if (ref_kv(rm, i, &k, &v) == 0) break;
                expectStr("gguf_kv key", std.mem.span(k), e.key);
                expectStr("gguf_kv value", std.mem.span(v), e.value.*);
            }
        },
    }

    const outcome = switch (loaded) {
        .model => "loaded",
        .thrown => "threw",
        .unported => "unported",
    };
    std.debug.print("  {s:<44} {s:<8} {d:>8} checks\n", .{ cs.name, outcome, checks - before });
}

// -----------------------------------------------------------------------------
// Full loads: the tensor half

const FullCase = struct {
    name: []const u8,
    path: [:0]const u8,
    edit: *const fn (*c.llama_model_params) void,
};

/// A log line's free-memory figure moves with what the process has
/// allocated, and the two loads run one after the other.
fn normalizeLog(log: []const u8) std.ArrayList(u8) {
    var out: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, log, '\n');
    var first = true;
    while (lines.next()) |line| {
        if (!first) out.append(gpa, '\n') catch unreachable;
        first = false;
        if (std.mem.startsWith(u8, line, "ggml_metal_log_allocated_size:")) {
            // `( total / max)`: the process-wide total, which counts the
            // other side's model too.
            if (std.mem.findScalar(u8, line, '(')) |open| {
                out.appendSlice(gpa, line[0..open]) catch unreachable;
                out.appendSlice(gpa, "(N)") catch unreachable;
                continue;
            }
        }
        if (std.mem.find(u8, line, "available_memory_kb: ")) |at| {
            // CUDA on a unified-memory device reads the system's free
            // memory from /proc/meminfo, which moves between any two reads.
            // `context_diff.zig` masks it the same way.
            out.appendSlice(gpa, line[0 .. at + "available_memory_kb: ".len]) catch unreachable;
            out.appendSlice(gpa, "N") catch unreachable;
            continue;
        }
        if (std.mem.find(u8, line, " MiB free")) |end| {
            var start = end;
            while (start > 0 and std.ascii.isDigit(line[start - 1])) start -= 1;
            out.appendSlice(gpa, line[0..start]) catch unreachable;
            out.appendSlice(gpa, "N") catch unreachable;
            out.appendSlice(gpa, line[end..]) catch unreachable;
        } else {
            out.appendSlice(gpa, line) catch unreachable;
        }
    }
    return out;
}

fn cancelHalfway(progress: f32, _: ?*anyopaque) callconv(.c) bool {
    return progress < 0.5;
}

fn devName(dev: c.ggml_backend_dev_t) []const u8 {
    return if (dev == null) "(null)" else std.mem.span(c.ggml_backend_dev_name(dev));
}

fn compareTensor(what: []const u8, ref: *const c.ggml_tensor, port: *const c.ggml_tensor, hash: bool) void {
    expectStr(what, std.mem.sliceTo(&ref.name, 0), std.mem.sliceTo(&port.name, 0));
    expectInt("type", ref.type, port.type);
    for (0..4) |d| {
        expectInt("ne", ref.ne[d], port.ne[d]);
        expectInt("nb", ref.nb[d], port.nb[d]);
    }
    checks += 1;
    if ((ref.buffer == null) != (port.buffer == null)) {
        fail("{s}: buffer presence differs", .{std.mem.sliceTo(&ref.name, 0)});
        return;
    }
    if (ref.buffer == null) return;
    expectStr("buffer", std.mem.span(c.ggml_backend_buffer_name(ref.buffer)), std.mem.span(c.ggml_backend_buffer_name(port.buffer)));
    expectInt("usage", c.ggml_backend_buffer_get_usage(ref.buffer), c.ggml_backend_buffer_get_usage(port.buffer));
    const rb = c.ggml_backend_buffer_get_base(ref.buffer);
    const pb = c.ggml_backend_buffer_get_base(port.buffer);
    if (rb != null and pb != null and ref.data != null and port.data != null) {
        expectInt("offset in buffer", @as(i128, @intFromPtr(ref.data)) - @intFromPtr(rb), @as(i128, @intFromPtr(port.data)) - @intFromPtr(pb));
    }
    if (hash) expectInt("data hash", ref_hash_tensor(ref), ref_hash_tensor(port));
}

var warmed_up = false;

fn runFullCase(fc: FullCase) void {
    case_failures = 0;
    const before = checks;

    var params = c.llama_model_default_params();
    fc.edit(&params);
    var port_params = M.defaultParams();
    fc.edit(&port_params);

    // The first load in the process initializes the backends, which log.
    if (!warmed_up) {
        warmed_up = true;
        var warm = c.llama_model_default_params();
        warm.vocab_only = true;
        if (c.llama_model_load_from_file(fc.path.ptr, warm)) |m| c.llama_model_free(m);
    }

    var ref_log: std.ArrayList(u8) = .empty;
    defer ref_log.deinit(gpa);
    var port_log: std.ArrayList(u8) = .empty;
    defer port_log.deinit(gpa);

    log_target = &ref_log;
    const rm = c.llama_model_load_from_file(fc.path.ptr, params);
    defer if (rm) |m| c.llama_model_free(m);

    log_target = &port_log;
    var splits: std.ArrayList([:0]const u8) = .empty;
    defer {
        for (splits.items) |s| gpa.free(s);
        splits.deinit(gpa);
    }
    const pm = M.loadFromFileImpl(.{ .path = fc.path }, &splits, null, null, port_params);
    defer if (pm) |m| M.destroy(m);
    log_target = null;

    var rl = normalizeLog(ref_log.items);
    defer rl.deinit(gpa);
    var pl = normalizeLog(port_log.items);
    defer pl.deinit(gpa);
    checks += 1;
    if (rl.items.len == 0) fail("{s}: the C++ logged nothing", .{fc.name});
    firstDiff(rl.items, pl.items);

    checks += 1;
    const outcome: []const u8 = if (rm != null and pm != null) "loaded" else if (rm == null and pm == null) "failed" else blk: {
        fail("{s}: ref {s}, port {s}", .{ fc.name, if (rm != null) "loaded" else "failed", if (pm != null) "loaded" else "failed" });
        break :blk "differ";
    };

    if (rm != null and pm != null) {
        const r = rm.?;
        const m = pm.?;
        const hash = !port_params.no_alloc and !port_params.vocab_only;

        expectInt("n_tensors", ref_n_tensors(r), m.tensors_by_name.items.len);
        for (m.tensors_by_name.items, 0..) |t, i| {
            if (i >= ref_n_tensors(r)) break;
            var name: [*:0]const u8 = undefined;
            const rt = ref_tensor_at(r, i, &name);
            expectStr("tensors_by_name key", std.mem.span(name), std.mem.sliceTo(&t.name, 0));
            compareTensor("tensor", rt, t, hash);
        }

        // Every model-level tensor, and every member of every layer, the
        // nested ones by their path: the C++ tables are generated with the
        // port's structs, so a name the C++ does not know reads "?".
        {
            @setEvalBranchQuota(200_000);
            const minfo = @typeInfo(M.Model).@"struct";
            inline for (minfo.field_names, minfo.field_types) |f, F| {
                if (F != ?*c.ggml_tensor) continue;
                const rt = ref_model_tensor(r, f);
                const pt = @field(m.*, f);
                expectStr("model." ++ f, if (rt) |p| std.mem.span(p) else "(null)", if (pt) |p| std.mem.sliceTo(&p.name, 0) else "(null)");
            }
        }

        expectInt("n layers", ref_n_layers(r), m.layers.len);
        for (m.layers, 0..) |*layer, il| {
            if (il >= ref_n_layers(r)) break;
            compareLayer(M.Layer, "", r, @intCast(il), layer);
        }

        expectInt("n devices", ref_n_devices(r), m.devices.items.len);
        for (m.devices.items, 0..) |d, i| {
            if (i < ref_n_devices(r)) expectStr("device", devName(ref_device(r, i)), devName(d.dev));
        }
        if (!port_params.vocab_only) {
            for (0..m.hparams.n_layer_all) |il| expectStr("dev_layer", devName(ref_dev_layer(r, @intCast(il))), devName(m.devLayer(il)));
            expectStr("dev_output", devName(ref_dev_output(r)), devName(m.devOutput()));
        }

        const mb = m.memoryBreakdown();
        defer gpa.free(mb);
        for (mb, 0..) |e, i| {
            var buft: c.ggml_backend_buffer_type_t = null;
            const size = ref_memory_breakdown(r, i, &buft);
            expectStr("memory_breakdown buft", if (buft == null) "(none)" else std.mem.span(c.ggml_backend_buft_name(buft)), std.mem.span(c.ggml_backend_buft_name(e.buft)));
            expectInt("memory_breakdown size", size, e.size);
        }
        var extra: c.ggml_backend_buffer_type_t = null;
        _ = ref_memory_breakdown(r, mb.len, &extra);
        checks += 1;
        if (extra != null) fail("memory_breakdown: the C++ has more entries", .{});
    }
    std.debug.print("  {s:<44} {s:<8} {d:>8} checks\n", .{ fc.name, outcome, checks - before });
}

const Edit = struct {
    fn none(_: *c.llama_model_params) void {}
    fn cpuOnly(p: *c.llama_model_params) void {
        p.n_gpu_layers = 0;
    }
    fn partial(p: *c.llama_model_params) void {
        p.n_gpu_layers = 10;
    }
    fn noMmap(p: *c.llama_model_params) void {
        p.load_mode = c.LLAMA_LOAD_MODE_NONE;
    }
    fn noMmapCpu(p: *c.llama_model_params) void {
        p.load_mode = c.LLAMA_LOAD_MODE_NONE;
        p.n_gpu_layers = 0;
    }
    fn mmapExplicit(p: *c.llama_model_params) void {
        p.load_mode = c.LLAMA_LOAD_MODE_MMAP;
    }
    fn noExtraBufts(p: *c.llama_model_params) void {
        p.use_extra_bufts = false;
        p.n_gpu_layers = 0;
    }
    fn noHost(p: *c.llama_model_params) void {
        p.no_host = true;
        p.n_gpu_layers = 0;
    }
    fn checkTensors(p: *c.llama_model_params) void {
        p.check_tensors = true;
    }
    /// `no_alloc` with mmap trips `GGML_ASSERT(!ml.no_alloc)` in the C++
    /// itself (llama-model.cpp:1701), so it is only ever used without.
    fn noAlloc(p: *c.llama_model_params) void {
        p.no_alloc = true;
        p.load_mode = c.LLAMA_LOAD_MODE_NONE;
    }
    fn noAllocCpu(p: *c.llama_model_params) void {
        p.no_alloc = true;
        p.load_mode = c.LLAMA_LOAD_MODE_NONE;
        p.n_gpu_layers = 0;
    }
    fn vocabOnly(p: *c.llama_model_params) void {
        p.vocab_only = true;
    }
    fn loadMtp(p: *c.llama_model_params) void {
        p.load_mtp = true;
    }
    fn splitNoneNoGpu(p: *c.llama_model_params) void {
        p.split_mode = c.LLAMA_SPLIT_MODE_NONE;
        p.main_gpu = -1;
    }
    fn splitNone(p: *c.llama_model_params) void {
        p.split_mode = c.LLAMA_SPLIT_MODE_NONE;
    }
    fn badMainGpu(p: *c.llama_model_params) void {
        p.split_mode = c.LLAMA_SPLIT_MODE_NONE;
        p.main_gpu = 5;
    }
    fn tensorSplit(p: *c.llama_model_params) void {
        const S = struct {
            var split: [16]f32 = .{ 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        };
        p.tensor_split = &S.split;
    }
    fn cancel(p: *c.llama_model_params) void {
        p.progress_callback = cancelHalfway;
    }
    fn overrideFfnCpu(p: *c.llama_model_params) void {
        const S = struct {
            var o: [2]c.llama_model_tensor_buft_override = undefined;
        };
        S.o = .{ .{ .pattern = "ffn_(up|down)", .buft = c.ggml_backend_cpu_buffer_type() }, .{ .pattern = null, .buft = null } };
        p.tensor_buft_overrides = &S.o;
    }
    fn overrideBadRegex(p: *c.llama_model_params) void {
        const S = struct {
            var o: [2]c.llama_model_tensor_buft_override = undefined;
        };
        S.o = .{ .{ .pattern = "blk.(", .buft = c.ggml_backend_cpu_buffer_type() }, .{ .pattern = null, .buft = null } };
        p.tensor_buft_overrides = &S.o;
    }
    fn overrideEmpty(p: *c.llama_model_params) void {
        const S = struct {
            var o: [1]c.llama_model_tensor_buft_override = .{.{ .pattern = null, .buft = null }};
        };
        p.tensor_buft_overrides = &S.o;
    }
};

// -----------------------------------------------------------------------------
// Synthetic files

fn kvInt(key: []const u8, v: i64) c.llama_model_kv_override {
    var o = std.mem.zeroes(c.llama_model_kv_override);
    o.tag = c.LLAMA_KV_OVERRIDE_TYPE_INT;
    @memcpy(o.key[0..key.len], key);
    o.unnamed_0.val_i64 = v;
    return o;
}

fn kvFloat(key: []const u8, v: f64) c.llama_model_kv_override {
    var o = std.mem.zeroes(c.llama_model_kv_override);
    o.tag = c.LLAMA_KV_OVERRIDE_TYPE_FLOAT;
    @memcpy(o.key[0..key.len], key);
    o.unnamed_0.val_f64 = v;
    return o;
}

fn kvBool(key: []const u8, v: bool) c.llama_model_kv_override {
    var o = std.mem.zeroes(c.llama_model_kv_override);
    o.tag = c.LLAMA_KV_OVERRIDE_TYPE_BOOL;
    @memcpy(o.key[0..key.len], key);
    o.unnamed_0.val_bool = v;
    return o;
}

fn kvStr(key: []const u8, v: []const u8) c.llama_model_kv_override {
    var o = std.mem.zeroes(c.llama_model_kv_override);
    o.tag = c.LLAMA_KV_OVERRIDE_TYPE_STR;
    @memcpy(o.key[0..key.len], key);
    @memcpy(o.unnamed_0.val_str[0..v.len], v);
    return o;
}

/// A tensor-less copy of `base`'s metadata with `edit` applied, written to
/// `dir/name.gguf`.
fn derive(dir: []const u8, base: [:0]const u8, name: []const u8, comptime edit: fn (*c.gguf_context) void) [:0]const u8 {
    const src = c.gguf_init_from_file(base.ptr, .{ .no_alloc = true, .ctx = null }).?;
    defer c.gguf_free(src);
    const ctx = c.gguf_init_empty().?;
    defer c.gguf_free(ctx);
    c.gguf_set_kv(ctx, src);
    edit(ctx);
    const path = gpa.printSentinel("{s}/{s}.gguf", .{ dir, name }, 0) catch unreachable;
    if (!c.gguf_write_to_file(ctx, path.ptr, true)) @panic("cannot write a synthetic file");
    return path;
}

fn setI32s(ctx: *c.gguf_context, key: [:0]const u8, v: []const i32) void {
    c.gguf_set_arr_data(ctx, key.ptr, c.GGUF_TYPE_INT32, v.ptr, v.len);
}

fn setU32s(ctx: *c.gguf_context, key: [:0]const u8, v: []const u32) void {
    c.gguf_set_arr_data(ctx, key.ptr, c.GGUF_TYPE_UINT32, v.ptr, v.len);
}

/// `n` copies of `pattern`, cycled, as a `uint32` array.
fn setCycle(ctx: *c.gguf_context, key: [:0]const u8, pattern: []const u32, n: usize) void {
    var v: [512]u32 = undefined;
    for (v[0..n], 0..) |*x, i| x.* = pattern[i % pattern.len];
    setU32s(ctx, key, v[0..n]);
}

fn setStrs(ctx: *c.gguf_context, key: [:0]const u8, strs: []const [*:0]const u8) void {
    c.gguf_set_arr_str(ctx, key.ptr, @ptrCast(@constCast(strs.ptr)), strs.len);
}

/// Edits of Qwen3.5's metadata, one key or one family at a time.
const Qwen = struct {
    fn asIs(_: *c.gguf_context) void {}

    fn noContext(ctx: *c.gguf_context) void {
        _ = c.gguf_remove_key(ctx, "qwen35.context_length");
    }

    fn contextAsString(ctx: *c.gguf_context) void {
        c.gguf_set_val_str(ctx, "qwen35.context_length", "262144");
    }

    fn headsPerLayer(ctx: *c.gguf_context) void {
        setCycle(ctx, "qwen35.attention.head_count", &.{ 8, 16, 8, 4 }, 24);
        setCycle(ctx, "qwen35.attention.head_count_kv", &.{ 2, 4 }, 24);
        setCycle(ctx, "qwen35.feed_forward_length", &.{ 6144, 3072 }, 24);
    }

    fn headsWrongLength(ctx: *c.gguf_context) void {
        setCycle(ctx, "qwen35.attention.head_count", &.{8}, 23);
    }

    fn headsHuge(ctx: *c.gguf_context) void {
        // Past INT32_MAX: print_info's lambda returns int32_t.
        setCycle(ctx, "qwen35.attention.head_count_kv", &.{ 0x80000001, 2 }, 24);
    }

    fn sectionsScalar(ctx: *c.gguf_context) void {
        c.gguf_set_val_i32(ctx, "qwen35.rope.dimension_sections", 11);
    }

    fn sectionsShort(ctx: *c.gguf_context) void {
        setI32s(ctx, "qwen35.rope.dimension_sections", &.{ 11, 11, 10 });
    }

    fn sectionsU32(ctx: *c.gguf_context) void {
        setU32s(ctx, "qwen35.rope.dimension_sections", &.{ 7, 0, 3, 1 });
    }

    fn sectionsFloat(ctx: *c.gguf_context) void {
        const v = [_]f32{ 1, 2, 3, 4 };
        c.gguf_set_arr_data(ctx, "qwen35.rope.dimension_sections", c.GGUF_TYPE_FLOAT32, &v, v.len);
    }

    fn noSections(ctx: *c.gguf_context) void {
        _ = c.gguf_remove_key(ctx, "qwen35.rope.dimension_sections");
    }

    fn yarn(ctx: *c.gguf_context) void {
        c.gguf_set_val_str(ctx, "qwen35.rope.scaling.type", "yarn");
        c.gguf_set_val_f32(ctx, "qwen35.rope.scaling.factor", 4.0);
        c.gguf_set_val_u32(ctx, "qwen35.rope.scaling.original_context_length", 32768);
        c.gguf_set_val_bool(ctx, "qwen35.rope.scaling.finetuned", true);
        c.gguf_set_val_f32(ctx, "qwen35.rope.scaling.attn_factor", 0.5);
    }

    fn scaleLinear(ctx: *c.gguf_context) void {
        c.gguf_set_val_f32(ctx, "qwen35.rope.scale_linear", 3.0);
    }

    fn interval0(ctx: *c.gguf_context) void {
        c.gguf_set_val_u32(ctx, "qwen35.full_attention_interval", 0);
    }

    fn interval3(ctx: *c.gguf_context) void {
        c.gguf_set_val_u32(ctx, "qwen35.full_attention_interval", 3);
    }

    fn recurrentArray(ctx: *c.gguf_context) void {
        setCycle(ctx, "qwen35.attention.recurrent_layers", &.{ 1, 0, 0 }, 24);
    }

    fn recurrentBools(ctx: *c.gguf_context) void {
        var v: [24]bool = undefined;
        for (&v, 0..) |*x, i| x.* = i % 5 == 0;
        c.gguf_set_arr_data(ctx, "qwen35.attention.recurrent_layers", c.GGUF_TYPE_BOOL, &v, v.len);
    }

    fn nextn(ctx: *c.gguf_context) void {
        c.gguf_set_val_u32(ctx, "qwen35.nextn_predict_layers", 2);
    }

    fn size4b(ctx: *c.gguf_context) void {
        c.gguf_set_val_u32(ctx, "qwen35.block_count", 32);
        c.gguf_set_val_u32(ctx, "qwen35.embedding_length", 2560);
    }

    fn size9b(ctx: *c.gguf_context) void {
        c.gguf_set_val_u32(ctx, "qwen35.block_count", 32);
    }

    fn size27b(ctx: *c.gguf_context) void {
        c.gguf_set_val_u32(ctx, "qwen35.block_count", 64);
    }

    fn sizeOdd(ctx: *c.gguf_context) void {
        c.gguf_set_val_u32(ctx, "qwen35.block_count", 7);
    }

    fn size0_8b(ctx: *c.gguf_context) void {
        c.gguf_set_val_u32(ctx, "qwen35.embedding_length", 1024);
    }

    fn fileType(ctx: *c.gguf_context) void {
        c.gguf_set_val_u32(ctx, "general.file_type", c.LLAMA_FTYPE_MOSTLY_IQ3_M);
    }

    fn noFileType(ctx: *c.gguf_context) void {
        _ = c.gguf_remove_key(ctx, "general.file_type");
    }

    fn noName(ctx: *c.gguf_context) void {
        _ = c.gguf_remove_key(ctx, "general.name");
    }

    fn labels(ctx: *c.gguf_context) void {
        setStrs(ctx, "qwen35.classifier.output_labels", &.{ "negative", "neutral", "positive \"quoted\"" });
    }

    fn headSizes(ctx: *c.gguf_context) void {
        c.gguf_set_val_u32(ctx, "qwen35.attention.key_length", 96);
        c.gguf_set_val_u32(ctx, "qwen35.attention.value_length", 80);
        c.gguf_set_val_u32(ctx, "qwen35.rope.dimension_count", 48);
        c.gguf_set_val_u32(ctx, "qwen35.attention.key_length_swa", 32);
        c.gguf_set_val_u32(ctx, "qwen35.rope.dimension_count_swa", 16);
    }

    fn noHeads(ctx: *c.gguf_context) void {
        c.gguf_set_val_u32(ctx, "qwen35.attention.head_count", 0);
    }

    fn pooling(ctx: *c.gguf_context) void {
        c.gguf_set_val_u32(ctx, "qwen35.pooling_type", 2);
        c.gguf_set_val_bool(ctx, "qwen35.attention.causal", false);
        c.gguf_set_val_u32(ctx, "qwen35.embedding_length_out", 512);
    }

    fn moe(ctx: *c.gguf_context) void {
        c.gguf_set_val_u32(ctx, "qwen35.expert_count", 8);
        c.gguf_set_val_u32(ctx, "qwen35.expert_used_count", 2);
    }

    fn noSsm(ctx: *c.gguf_context) void {
        _ = c.gguf_remove_key(ctx, "qwen35.ssm.state_size");
    }

    fn rmsEpsAsDouble(ctx: *c.gguf_context) void {
        c.gguf_set_val_f64(ctx, "qwen35.attention.layer_norm_rms_epsilon", 1e-6);
    }

    fn freqBase(ctx: *c.gguf_context) void {
        c.gguf_set_val_f32(ctx, "qwen35.rope.freq_base", 1234567.875);
    }

    fn unknownArch(ctx: *c.gguf_context) void {
        c.gguf_set_val_str(ctx, "general.architecture", "no-such-arch");
    }

    fn noArch(ctx: *c.gguf_context) void {
        _ = c.gguf_remove_key(ctx, "general.architecture");
    }

    fn gptj(ctx: *c.gguf_context) void {
        c.gguf_set_val_str(ctx, "general.architecture", "gptj");
    }

    fn longValues(ctx: *c.gguf_context) void {
        c.gguf_set_val_str(ctx, "general.description", "line one\nline two, which is long enough to be cut off at forty bytes");
        c.gguf_set_val_str(ctx, "general.short", "a\nb");
        c.gguf_set_val_f64(ctx, "general.f64", 3.14159265358979);
        c.gguf_set_val_i8(ctx, "general.i8", -5);
        c.gguf_set_val_u64(ctx, "general.u64", 18446744073709551615);
        c.gguf_set_val_i64(ctx, "general.i64", -9223372036854775807);
        c.gguf_set_val_i16(ctx, "general.i16", -300);
        c.gguf_set_val_u16(ctx, "general.u16", 65535);
        const v = [_]f32{ 0.1, -2.5e10, 7 };
        c.gguf_set_arr_data(ctx, "general.f32s", c.GGUF_TYPE_FLOAT32, &v, v.len);
        const b = [_]bool{ true, false };
        c.gguf_set_arr_data(ctx, "general.bools", c.GGUF_TYPE_BOOL, &b, b.len);
    }
};

/// A small file with real tensors, for the file-type guess and the bounds
/// check. `blk.2.*` is Q6_K and `blk.10.*` Q4_K, two each: in
/// `weight_name_comparer` order Q6_K reaches two first and wins the tie,
/// where a plain name sort would put `blk.10` first and pick Q4_K.
fn tensorFile(dir: []const u8, base: [:0]const u8, name: []const u8, comptime edit: fn (*c.gguf_context, *c.ggml_context) void, truncate_by: usize) [:0]const u8 {
    const src = c.gguf_init_from_file(base.ptr, .{ .no_alloc = true, .ctx = null }).?;
    defer c.gguf_free(src);
    const gctx = c.gguf_init_empty().?;
    defer c.gguf_free(gctx);
    c.gguf_set_kv(gctx, src);

    const ctx = c.ggml_init(.{ .mem_size = 1 << 22, .mem_buffer = null, .no_alloc = false }).?;
    defer c.ggml_free(ctx);
    edit(gctx, ctx);

    const path = gpa.printSentinel("{s}/{s}.gguf", .{ dir, name }, 0) catch unreachable;
    if (!c.gguf_write_to_file(gctx, path.ptr, false)) @panic("cannot write a synthetic file");
    if (truncate_by != 0) {
        const f = c.fopen(path.ptr, "rb+").?;
        _ = c.fseeko(f, 0, c.SEEK_END);
        const size = c.ftello(f);
        _ = c.fclose(f);
        if (truncate(path.ptr, size - @as(i64, @intCast(truncate_by))) != 0) @panic("truncate");
    }
    return path;
}

fn addTensor(gctx: *c.gguf_context, ctx: *c.ggml_context, name: [:0]const u8, ty: c.enum_ggml_type, n: i64) void {
    const t = c.ggml_new_tensor_1d(ctx, ty, n);
    _ = c.ggml_set_name(t, name.ptr);
    @memset(@as([*]u8, @ptrCast(t.*.data))[0..c.ggml_nbytes(t)], 0);
    c.gguf_add_tensor(gctx, t);
}

const Tensors = struct {
    fn tie(g: *c.gguf_context, ctx: *c.ggml_context) void {
        addTensor(g, ctx, "output.weight", c.GGML_TYPE_F32, 64);
        addTensor(g, ctx, "blk.10.a.weight", c.GGML_TYPE_Q4_K, 256);
        addTensor(g, ctx, "blk.10.b.weight", c.GGML_TYPE_Q4_K, 256);
        addTensor(g, ctx, "blk.2.b.weight", c.GGML_TYPE_Q6_K, 256);
        addTensor(g, ctx, "blk.2.a.weight", c.GGML_TYPE_Q6_K, 256);
        _ = c.gguf_remove_key(g, "general.file_type");
    }

    fn integers(g: *c.gguf_context, ctx: *c.ggml_context) void {
        addTensor(g, ctx, "blk.0.x", c.GGML_TYPE_I32, 16);
        addTensor(g, ctx, "blk.0.y", c.GGML_TYPE_I32, 16);
        addTensor(g, ctx, "blk.-1.z", c.GGML_TYPE_F16, 16);
        _ = c.gguf_remove_key(g, "general.file_type");
    }

    fn many(g: *c.gguf_context, ctx: *c.ggml_context) void {
        const types = [_]c.enum_ggml_type{ c.GGML_TYPE_Q8_0, c.GGML_TYPE_BF16, c.GGML_TYPE_IQ4_NL, c.GGML_TYPE_Q5_1, c.GGML_TYPE_Q8_0 };
        for (types, 0..) |ty, i| {
            var buf: [32]u8 = undefined;
            const n = std.fmt.bufPrintSentinel(&buf, "blk.{d}.w", .{i % 3}, 0) catch unreachable;
            var nb: [40]u8 = undefined;
            const nm = std.fmt.bufPrintSentinel(&nb, "{s}{d}", .{ n, i }, 0) catch unreachable;
            addTensor(g, ctx, nm, ty, 1024);
        }
    }
};

/// A split set derived from `base`: the first file holds the metadata and
/// one tensor, the second another tensor; `kind` breaks it in one place.
fn splitSet(dir: []const u8, base: [:0]const u8, name: []const u8, kind: enum { ok, wrong_idx, first_not_zero, bad_count, missing_second, duplicate, bad_name }) [:0]const u8 {
    const src = c.gguf_init_from_file(base.ptr, .{ .no_alloc = true, .ctx = null }).?;
    defer c.gguf_free(src);

    var paths: [2][:0]const u8 = undefined;
    for (0..2) |i| {
        paths[i] = if (kind == .bad_name and i == 0)
            gpa.printSentinel("{s}/{s}-first.gguf", .{ dir, name }, 0) catch unreachable
        else
            gpa.printSentinel("{s}/{s}-{d:0>5}-of-00002.gguf", .{ dir, name, i + 1 }, 0) catch unreachable;
    }
    for (0..2) |i| {
        if (kind == .missing_second and i == 1) continue;
        const g = c.gguf_init_empty().?;
        defer c.gguf_free(g);
        if (i == 0) c.gguf_set_kv(g, src);
        _ = c.gguf_remove_key(g, "general.file_type");
        const no: u16 = switch (kind) {
            .wrong_idx => if (i == 1) 5 else 0,
            .first_not_zero => if (i == 0) 1 else 1,
            else => @intCast(i),
        };
        c.gguf_set_val_u16(g, "split.no", no);
        c.gguf_set_val_u16(g, "split.count", 2);
        c.gguf_set_val_i32(g, "split.tensors.count", if (kind == .bad_count) 3 else 2);
        const ctx = c.ggml_init(.{ .mem_size = 1 << 20, .mem_buffer = null, .no_alloc = false }).?;
        defer c.ggml_free(ctx);
        const tname: [:0]const u8 = if (kind == .duplicate) "blk.0.dup" else if (i == 0) "blk.0.a" else "blk.1.b";
        addTensor(g, ctx, tname, if (i == 0) c.GGML_TYPE_F16 else c.GGML_TYPE_Q8_0, 64);
        if (!c.gguf_write_to_file(g, paths[i].ptr, false)) @panic("cannot write a split");
    }
    return paths[0];
}

// -----------------------------------------------------------------------------

/// One layer's members against the C++'s, recursing into the nested
/// structs with their path as the prefix.
fn compareLayer(comptime T: type, comptime prefix: []const u8, r: *const c.llama_model, il: c_int, layer: *const T) void {
    @setEvalBranchQuota(200_000);
    const info = @typeInfo(T).@"struct";
    inline for (info.field_names, info.field_types) |f, F| {
        if (F == ?*c.ggml_tensor) {
            const rt = ref_layer_tensor(r, il, prefix ++ f);
            const pt = @field(layer.*, f);
            expectStr("layer." ++ prefix ++ f, if (rt) |p| std.mem.span(p) else "(null)", if (pt) |p| std.mem.sliceTo(&p.name, 0) else "(null)");
        } else {
            compareLayer(F, prefix ++ f ++ ".", r, il, &@field(layer.*, f));
        }
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: loader_diff <tmpdir> <file.gguf>...\n", .{});
        std.process.exit(2);
    }
    const dir = args[1];
    c.llama_log_set(logCb, null);
    harness_root.api.impl.llama_log_set(logCb, null);
    c.ggml_log_set(logCb, null);

    std.debug.print("  llama_hparams layout\n", .{});
    if (!checkLayout()) {
        std.debug.print("FAIL: the port's Hparams does not have llama_hparams's layout; nothing else is meaningful\n", .{});
        std.process.exit(1);
    }

    var qwen: ?[:0]const u8 = null;
    var gpt2: ?[:0]const u8 = null;
    for (args[2..]) |a| {
        const path = try gpa.dupeSentinel(u8, a, 0);
        const name = std.fs.path.basename(a);
        if (std.mem.startsWith(u8, name, "Qwen3.5")) qwen = path;
        if (std.mem.eql(u8, name, "ggml-vocab-gpt-2.gguf")) gpt2 = path;
        runCase(.{ .name = name, .path = path, .vocab_only = true });
        var nb: [128]u8 = undefined;
        runCase(.{ .name = try std.fmt.bufPrint(&nb, "{s} (full)", .{name}), .path = path, .vocab_only = false });
    }

    if (gpt2) |base| {
        runCase(.{ .name = "tensors: type tie", .path = tensorFile(dir, base, "tie", Tensors.tie, 0), .vocab_only = true });
        runCase(.{ .name = "tensors: integer types", .path = tensorFile(dir, base, "integers", Tensors.integers, 0), .vocab_only = true });
        runCase(.{ .name = "tensors: many types", .path = tensorFile(dir, base, "many", Tensors.many, 0), .vocab_only = true });
        runCase(.{ .name = "tensors: truncated", .path = tensorFile(dir, base, "truncated", Tensors.tie, 100), .vocab_only = true });
        runCase(.{ .name = "split: ok", .path = splitSet(dir, base, "s-ok", .ok), .vocab_only = true });
        runCase(.{ .name = "split: wrong index", .path = splitSet(dir, base, "s-idx", .wrong_idx), .vocab_only = true });
        runCase(.{ .name = "split: first not zero", .path = splitSet(dir, base, "s-first", .first_not_zero), .vocab_only = true });
        runCase(.{ .name = "split: tensor count", .path = splitSet(dir, base, "s-count", .bad_count), .vocab_only = true });
        runCase(.{ .name = "split: second missing", .path = splitSet(dir, base, "s-missing", .missing_second), .vocab_only = true });
        runCase(.{ .name = "split: duplicate tensor", .path = splitSet(dir, base, "s-dup", .duplicate), .vocab_only = true });
        runCase(.{ .name = "split: unrecognized name", .path = splitSet(dir, base, "s-name", .bad_name), .vocab_only = true });
        runCase(.{ .name = "missing file", .path = try gpa.printSentinel("{s}/no-such-file.gguf", .{dir}, 0), .vocab_only = true });
    }

    if (qwen) |base| {
        const edits = .{
            .{ "qwen35 metadata", Qwen.asIs },
            .{ "no context_length", Qwen.noContext },
            .{ "context_length as string", Qwen.contextAsString },
            .{ "per-layer heads", Qwen.headsPerLayer },
            .{ "heads, wrong length", Qwen.headsWrongLength },
            .{ "heads past INT32_MAX", Qwen.headsHuge },
            .{ "sections scalar", Qwen.sectionsScalar },
            .{ "sections short", Qwen.sectionsShort },
            .{ "sections uint32", Qwen.sectionsU32 },
            .{ "sections float", Qwen.sectionsFloat },
            .{ "no sections", Qwen.noSections },
            .{ "yarn", Qwen.yarn },
            .{ "scale_linear", Qwen.scaleLinear },
            .{ "full_attention_interval 0", Qwen.interval0 },
            .{ "full_attention_interval 3", Qwen.interval3 },
            .{ "recurrent_layers array", Qwen.recurrentArray },
            .{ "recurrent_layers bools", Qwen.recurrentBools },
            .{ "nextn", Qwen.nextn },
            .{ "size 4B", Qwen.size4b },
            .{ "size 9B", Qwen.size9b },
            .{ "size 27B", Qwen.size27b },
            .{ "size unknown", Qwen.sizeOdd },
            .{ "size 0.8B", Qwen.size0_8b },
            .{ "file_type", Qwen.fileType },
            .{ "no file_type", Qwen.noFileType },
            .{ "no name", Qwen.noName },
            .{ "classifier labels", Qwen.labels },
            .{ "head sizes", Qwen.headSizes },
            .{ "no heads", Qwen.noHeads },
            .{ "pooling", Qwen.pooling },
            .{ "moe", Qwen.moe },
            .{ "no ssm state", Qwen.noSsm },
            .{ "rms eps as f64", Qwen.rmsEpsAsDouble },
            .{ "freq_base", Qwen.freqBase },
            .{ "unknown arch", Qwen.unknownArch },
            .{ "no arch", Qwen.noArch },
            .{ "gptj", Qwen.gptj },
            .{ "long and odd values", Qwen.longValues },
        };
        inline for (edits, 0..) |e, i| {
            var nb: [32]u8 = undefined;
            const file = try std.fmt.bufPrint(&nb, "q{d}", .{i});
            const path = derive(dir, base, file, e[1]);
            runCase(.{ .name = e[0], .path = path, .vocab_only = false });
        }

        const meta = derive(dir, base, "qmeta", Qwen.asIs);
        runCase(.{ .name = "override: int", .path = meta, .vocab_only = false, .overrides = &.{
            kvInt("qwen35.context_length", 4096),
            kvInt("qwen35.block_count", 32),
            kvInt("qwen35.full_attention_interval", -1),
        } });
        runCase(.{ .name = "override: float", .path = meta, .vocab_only = false, .overrides = &.{
            kvFloat("qwen35.attention.layer_norm_rms_epsilon", 1.0e-5),
            kvFloat("qwen35.rope.freq_base", 1e300),
        } });
        runCase(.{ .name = "override: bool and str", .path = meta, .vocab_only = false, .overrides = &.{
            kvBool("qwen35.attention.causal", false),
            kvStr("qwen35.rope.scaling.type", "yarn"),
            kvStr("general.name", "overridden"),
            kvStr("general.architecture", "qwen35"),
        } });
        runCase(.{ .name = "override: mistyped", .path = meta, .vocab_only = false, .overrides = &.{
            kvStr("qwen35.context_length", "4096"),
            kvInt("qwen35.rope.freq_base", 5),
            kvFloat("qwen35.attention.causal", 1),
        } });
        runCase(.{ .name = "override: repeated key", .path = meta, .vocab_only = false, .overrides = &.{
            kvInt("qwen35.context_length", 111),
            kvInt("qwen35.context_length", 222),
        } });
        runCase(.{ .name = "override: arch to gptj", .path = meta, .vocab_only = false, .overrides = &.{
            kvStr("general.architecture", "gptj"),
        } });
    }

    // The tensor half: every Qwen3.5 model whole, and the first under
    // every parameter that changes where a tensor goes or how it loads.
    var first_qwen: ?[:0]const u8 = null;
    for (args[2..]) |a| {
        const name = std.fs.path.basename(a);
        if (!std.mem.startsWith(u8, name, "Qwen3.5")) continue;
        const path = try gpa.dupeSentinel(u8, a, 0);
        var nb: [128]u8 = undefined;
        runFullCase(.{ .name = try std.fmt.bufPrint(&nb, "{s} (tensors)", .{name}), .path = path, .edit = Edit.none });
        if (first_qwen == null) first_qwen = path;
    }
    if (first_qwen) |path| {
        const variants = .{
            .{ "tensors: cpu only", Edit.cpuOnly },
            .{ "tensors: 10 layers offloaded", Edit.partial },
            .{ "tensors: no mmap", Edit.noMmap },
            .{ "tensors: no mmap, cpu", Edit.noMmapCpu },
            .{ "tensors: mmap explicit", Edit.mmapExplicit },
            .{ "tensors: no extra buffer types", Edit.noExtraBufts },
            .{ "tensors: no host buffer", Edit.noHost },
            .{ "tensors: check tensors", Edit.checkTensors },
            .{ "tensors: no alloc", Edit.noAlloc },
            .{ "tensors: no alloc, cpu", Edit.noAllocCpu },
            .{ "tensors: vocab only", Edit.vocabOnly },
            .{ "tensors: load mtp", Edit.loadMtp },
            .{ "tensors: split none, no gpu", Edit.splitNoneNoGpu },
            .{ "tensors: split none", Edit.splitNone },
            .{ "tensors: bad main_gpu", Edit.badMainGpu },
            .{ "tensors: tensor split", Edit.tensorSplit },
            .{ "tensors: cancelled", Edit.cancel },
            .{ "tensors: ffn overridden to CPU", Edit.overrideFfnCpu },
            .{ "tensors: override, bad regex", Edit.overrideBadRegex },
            .{ "tensors: override list empty", Edit.overrideEmpty },
        };
        inline for (variants) |v| runFullCase(.{ .name = v[0], .path = path, .edit = v[1] });
        // Metadata alone: the first tensor is missing.
        runFullCase(.{ .name = "tensors: metadata only", .path = derive(dir, path, "qtensorless", Qwen.asIs), .edit = Edit.none });
    }

    std.debug.print("  {d} loaded, {d} threw, {d} unported\n", .{ tally.loaded, tally.thrown, tally.unported });
    if (failures != 0) {
        std.debug.print("FAIL: {d} of {d} loader results differ, Zig vs llama-model-loader.cpp\n", .{ failures, checks });
        std.process.exit(1);
    }
    std.debug.print("PASS: {d} loader results identical, Zig vs llama-model-loader.cpp\n", .{checks});
}
