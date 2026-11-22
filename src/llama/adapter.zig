//! Adapters applied on top of a model: control vectors and LoRA.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-adapter.h` -- `llama_adapter_cvec`,
//!   `llama_adapter_lora_weight`, `llama_adapter_lora` and
//!   `llama_adapter_loras`
//! - `llama.cpp/src/llama-adapter.cpp` -- all of it: the control vector's
//!   `tensor_for`, `apply_to`, `init` and `apply`, and the LoRA's
//!   `get_weight`, `llama_adapter_lora_init_impl`, `llama_adapter_lora_init`
//!   and `llama_adapter_lora_free`
//! - `llama.cpp/src/llama-model.cpp` -- the LoRA half of `~llama_model`
//!
//! All at v1.0.0 (`b1b770d90`). The C entry points are `api/adapter.zig`'s;
//! the context's setters are `context.zig`'s.
//!
//! # The order the C++ leaves to libc++
//!
//! Three containers here are iterated where it shows:
//!
//! - `std::map<ggml_backend_buffer_type_t, ggml_context *>` -- buffers are
//!   allocated, and their sizes logged, in buffer-type *address* order.
//!   `BuftCtxMap` keeps that order.
//! - `std::map<std::string, llama_adapter_lora_weight>` -- the LoRA's pairs
//!   are matched against the model, and the first bad one throws, in byte
//!   order of their names. `PairMap` keeps that order.
//! - `std::unordered_map<std::string, std::string>` for the metadata, read
//!   by index through the C API, and `std::unordered_map<llama_adapter_lora
//!   *, float>` for the active adapters, which the graph walks to add their
//!   deltas: both are `libcxx.StringMap`, the second keyed by the pointer's
//!   eight bytes, because libc++'s `hash<T *>` is `__hash_memory` over them.

const std = @import("std");
const mod = @import("module.zig");
const stdlib = @import("stdlib");

fn oom() noreturn {
    mod.impl.abort("llama-adapter.cpp", 0, "out of memory");
}

fn endsWith(s: []const u8, suffix: []const u8) bool {
    return std.mem.endsWith(u8, s, suffix);
}

/// `std::map<ggml_backend_buffer_type_t, ggml_context *>`, and the
/// `ctx_for_buft` lambda both `llama_adapter_cvec::init` and
/// `llama_adapter_lora_init_impl` build over it: one no-alloc context per
/// buffer type, iterated in address order as `std::less` on pointers does.
const BuftCtxMap = struct {
    entries: std.ArrayList(Entry) = .empty,

    const Entry = struct { buft: mod.impl.llama_c.ggml_backend_buffer_type_t, ctx: *mod.impl.llama_c.ggml_context };

    fn deinit(self: *BuftCtxMap) void {
        self.entries.deinit(std.heap.c_allocator);
    }

    /// Return: the context for `buft`, created on first use with room for
    /// `mem_size` bytes of tensor metadata and appended to `ctxs`; null
    /// when `ggml_init` fails.
    fn ctxForBuft(self: *BuftCtxMap, buft: mod.impl.llama_c.ggml_backend_buffer_type_t, mem_size: usize, ctxs: *std.ArrayList(*mod.impl.llama_c.ggml_context)) ?*mod.impl.llama_c.ggml_context {
        const key = @intFromPtr(buft);
        var at: usize = 0;
        while (at < self.entries.items.len) : (at += 1) {
            const k = @intFromPtr(self.entries.items[at].buft);
            if (k == key) return self.entries.items[at].ctx;
            if (k > key) break;
        }
        const ctx = mod.impl.llama_c.ggml_init(.{ .mem_size = mem_size, .mem_buffer = null, .no_alloc = true }) orelse return null;
        self.entries.insert(std.heap.c_allocator, at, .{ .buft = buft, .ctx = ctx }) catch oom();
        ctxs.append(std.heap.c_allocator, ctx) catch oom();
        return ctx;
    }
};

/// Mirrors `llama_adapter_cvec` (llama-adapter.h:17 @b1b770d90): one
/// direction per layer, added to that layer's output.
pub const Cvec = struct {
    layer_start: i32 = -1,
    layer_end: i32 = -1,
    ctxs: std.ArrayList(*mod.impl.llama_c.ggml_context) = .empty,
    bufs: std.ArrayList(mod.impl.llama_c.ggml_backend_buffer_t) = .empty,
    /// Per layer; null where the vector has no direction.
    tensors: std.ArrayList(?*mod.impl.llama_c.ggml_tensor) = .empty,

    /// Return: nothing.
    pub fn deinit(self: *Cvec) void {
        for (self.ctxs.items) |ctx| mod.impl.llama_c.ggml_free(ctx);
        self.ctxs.deinit(std.heap.c_allocator);
        for (self.bufs.items) |buf| mod.impl.llama_c.ggml_backend_buffer_free(buf);
        self.bufs.deinit(std.heap.c_allocator);
        self.tensors.deinit(std.heap.c_allocator);
    }

    /// Ports `llama_adapter_cvec`'s `tensor_for` (llama-adapter.cpp:16 @b1b770d90).
    ///
    /// Return: layer `il`'s direction, or null outside the active range.
    pub fn tensorFor(self: *const Cvec, il: c_int) ?*mod.impl.llama_c.ggml_tensor {
        if (il < 0 or il < self.layer_start or il > self.layer_end or @as(usize, @intCast(il)) >= self.tensors.items.len) {
            return null;
        }
        return self.tensors.items[@intCast(il)];
    }

    /// Ports `llama_adapter_cvec`'s `apply_to` (llama-adapter.cpp:24 @b1b770d90).
    ///
    /// Return: `cur` plus layer `il`'s direction, or `cur` when it has none.
    pub fn applyTo(self: *const Cvec, ctx: *mod.impl.llama_c.ggml_context, cur: *mod.impl.llama_c.ggml_tensor, il: c_int) *mod.impl.llama_c.ggml_tensor {
        if (self.tensorFor(il)) |layer_dir| return mod.impl.llama_c.ggml_add(ctx, cur, layer_dir);
        return cur;
    }

    /// Ports `llama_adapter_cvec`'s `init` (llama-adapter.cpp:33 @b1b770d90):
    /// one `n_embd` vector per layer but the first, each on the buffer type
    /// its layer's device can add it from, zeroed.
    ///
    /// Return: whether every context and buffer was allocated. A failure
    /// leaves what was made so far, as the C++ does.
    fn init(self: *Cvec, model: *const mod.model.Model) bool {
        const func = "init";
        const hparams = &model.hparams;

        mod.impl.assert(self.tensors.items.len == 0, "llama-adapter.cpp", 36, "tensors.empty()");
        mod.impl.assert(self.ctxs.items.len == 0, "llama-adapter.cpp", 37, "ctxs.empty()");
        mod.impl.assert(self.bufs.items.len == 0, "llama-adapter.cpp", 38, "bufs.empty()");

        var ctx_map: BuftCtxMap = .{};
        defer ctx_map.deinit();
        const n_layer = hparams.nLayer();
        const mem_size = @as(usize, n_layer) * mod.impl.llama_c.ggml_tensor_overhead();

        self.tensors.ensureTotalCapacity(std.heap.c_allocator, n_layer) catch oom();
        // There is never a tensor for layer 0.
        self.tensors.append(std.heap.c_allocator, null) catch oom();
        var il: usize = 1;
        while (il < n_layer) : (il += 1) {
            // `select_buft` throws only when no buffer type of the layer's
            // can add two vectors; nothing on this path catches it.
            var exc: mod.impl.Exception = .{};
            const buft = model.selectBuft(il, &exc) catch exc.terminate("llama-adapter.cpp", 69);
            const ctx = ctx_map.ctxForBuft(buft, mem_size, &self.ctxs) orelse {
                mod.impl.logError("%s: failed to allocate context for control vector\n", .{func});
                return false;
            };
            const tensor = mod.impl.llama_c.ggml_new_tensor_1d(ctx, mod.impl.llama_c.GGML_TYPE_F32, hparams.n_embd);
            self.tensors.append(std.heap.c_allocator, tensor) catch oom();
        }

        self.bufs.ensureTotalCapacity(std.heap.c_allocator, ctx_map.entries.items.len) catch oom();
        for (ctx_map.entries.items) |e| {
            const buf = mod.impl.llama_c.ggml_backend_alloc_ctx_tensors_from_buft(e.ctx, e.buft) orelse {
                mod.impl.logError("%s: failed to allocate buffer for control vector\n", .{func});
                return false;
            };
            mod.impl.llama_c.ggml_backend_buffer_clear(buf, 0);
            self.bufs.append(std.heap.c_allocator, buf) catch oom();
        }
        return true;
    }

    /// Ports `llama_adapter_cvec`'s `apply` (llama-adapter.cpp:96 @b1b770d90):
    /// set the vector from `data`, which holds layers 1 onward, `n_embd`
    /// floats each, and activate it over `[il_start, il_end]`. A null
    /// `data` deactivates it and keeps the tensors for later.
    ///
    /// Parameters:
    /// - `model`: the model the vector applies to.
    /// - `data`: the directions, or null; borrowed for the call.
    /// - `len`: how many floats `data` holds; layers past it keep their
    ///   previous direction.
    /// - `n_embd`: the width of one direction; must be the model's.
    /// - `il_start`, `il_end`: the active layers, inclusive.
    ///
    /// Return: whether the vector was applied.
    pub fn apply(self: *Cvec, model: *const mod.model.Model, data: ?[*]const f32, len: usize, n_embd: i32, il_start: i32, il_end: i32) bool {
        const func = "apply";
        const hparams = &model.hparams;

        const d = data orelse {
            self.layer_start = -1;
            self.layer_end = -1;
            return true;
        };

        if (n_embd != @as(c_int, @bitCast(hparams.n_embd))) {
            mod.impl.logError("%s: control vector n_embd does not match model\n", .{func});
            return false;
        }

        if (self.tensors.items.len == 0) {
            if (!self.init(model)) return false;
        }

        self.layer_start = il_start;
        self.layer_end = il_end;

        // `n_embd * (il - 1)`: the `int32_t` widens to `size_t`.
        const ne: usize = @as(usize, @bitCast(@as(isize, n_embd)));
        var il: usize = 1;
        while (il < hparams.nLayer()) : (il += 1) {
            // The buffer has no data for layer 0, which is never present.
            const off = ne *% (il - 1);
            if (off +% ne <= len) {
                const t = self.tensors.items[il].?;
                mod.impl.llama_c.ggml_backend_tensor_set(t, d + off, 0, ne *% mod.impl.llama_c.ggml_element_size(t));
            }
        }
        return true;
    }
};

/// Mirrors `llama_adapter_lora_weight` (llama-adapter.h:48 @b1b770d90).
pub const LoraWeight = struct {
    a: ?*mod.impl.llama_c.ggml_tensor = null,
    b: ?*mod.impl.llama_c.ggml_tensor = null,

    /// Ports `llama_adapter_lora_weight`'s `get_scale` (llama-adapter.h:53
    /// @b1b770d90): the adapter's scale, times `alpha / rank` when the
    /// adapter has an alpha.
    pub fn getScale(self: *const LoraWeight, alpha: f32, adapter_scale: f32) f32 {
        const rank: f32 = @floatFromInt(self.b.?.ne[0]);
        return if (alpha != 0) adapter_scale * alpha / rank else adapter_scale;
    }
};

/// Mirrors `llama_adapter_lora` (llama-adapter.h:63 @b1b770d90).
///
/// Behind a `llama_adapter_lora *`, so it is always heap-allocated through
/// `create` and released through `destroy`.
pub const Lora = struct {
    /// The model it was loaded against, which frees it with itself; null
    /// once `llama_adapter_lora_free` has detached it.
    model: ?*mod.model.Model = null,
    /// `ab_map`: a weight's name to its pair.
    ab_map: stdlib.libcxx.StringMap(LoraWeight),
    ctxs: std.ArrayList(*mod.impl.llama_c.ggml_context) = .empty,
    bufs: std.ArrayList(mod.impl.llama_c.ggml_backend_buffer_t) = .empty,
    alpha: f32 = 0,
    /// The file's scalar metadata, as text.
    gguf_kv: stdlib.libcxx.StringMap([:0]u8),
    alora_invocation_tokens: std.ArrayList(mod.impl.llama_c.llama_token) = .empty,

    /// Ports the constructor of `llama_adapter_lora` (llama-adapter.h:80
    /// @b1b770d90).
    ///
    /// Return: an empty adapter for `model`, the caller's to `destroy`.
    pub fn create(model: ?*mod.model.Model) *Lora {
        const self = std.heap.c_allocator.create(Lora) catch oom();
        self.* = .{
            .model = model,
            .ab_map = stdlib.libcxx.StringMap(LoraWeight).init(std.heap.c_allocator) catch oom(),
            .gguf_kv = stdlib.libcxx.StringMap([:0]u8).init(std.heap.c_allocator) catch oom(),
        };
        return self;
    }

    /// `delete adapter`: frees its contexts, buffers and metadata. It does
    /// not detach it from the model; `llama_adapter_lora_free` does.
    ///
    /// Return: nothing.
    pub fn destroy(self: *Lora) void {
        self.alora_invocation_tokens.deinit(std.heap.c_allocator);
        var it = self.gguf_kv.iterator();
        while (it.next()) |e| std.heap.c_allocator.free(e.value.*);
        self.gguf_kv.deinit();
        for (self.bufs.items) |buf| mod.impl.llama_c.ggml_backend_buffer_free(buf);
        self.bufs.deinit(std.heap.c_allocator);
        for (self.ctxs.items) |ctx| mod.impl.llama_c.ggml_free(ctx);
        self.ctxs.deinit(std.heap.c_allocator);
        self.ab_map.deinit();
        std.heap.c_allocator.destroy(self);
    }

    /// Ports `llama_adapter_lora`'s `get_weight` (llama-adapter.cpp:140 @b1b770d90).
    ///
    /// Return: the pair adapting `w`, borrowed from the adapter, or null.
    pub fn getWeight(self: *Lora, w: *const mod.impl.llama_c.ggml_tensor) ?*LoraWeight {
        return self.ab_map.get(std.mem.sliceTo(&w.name, 0));
    }

    /// Ports `llama_adapter_lora`'s `get_n_nodes` (llama-adapter.h:85 @b1b770d90).
    pub fn getNNodes(self: *const Lora) u32 {
        return @as(u32, @truncate(self.ab_map.count())) *% 6;
    }
};

/// `std::map<std::string, llama_adapter_lora_weight>`, the pairs as the file
/// lists them: in byte order of their names, as `std::string`'s `<` is.
const PairMap = struct {
    items: std.ArrayList(Pair) = .empty,

    const Pair = struct { name: []u8, w: LoraWeight };

    fn deinit(self: *PairMap) void {
        for (self.items.items) |p| std.heap.c_allocator.free(p.name);
        self.items.deinit(std.heap.c_allocator);
    }

    fn lowerBound(self: *const PairMap, name: []const u8) usize {
        var lo: usize = 0;
        var hi: usize = self.items.items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (std.mem.order(u8, self.items.items[mid].name, name) == .lt) lo = mid + 1 else hi = mid;
        }
        return lo;
    }

    fn find(self: *const PairMap, name: []const u8) ?*LoraWeight {
        const at = self.lowerBound(name);
        if (at < self.items.items.len and std.mem.eql(u8, self.items.items[at].name, name)) return &self.items.items[at].w;
        return null;
    }

    /// `operator[]`: the pair for `name`, inserted empty if absent.
    fn slot(self: *PairMap, name: []const u8) *LoraWeight {
        const at = self.lowerBound(name);
        if (at < self.items.items.len and std.mem.eql(u8, self.items.items[at].name, name)) return &self.items.items[at].w;
        self.items.insert(std.heap.c_allocator, at, .{ .name = std.heap.c_allocator.dupe(u8, name) catch oom(), .w = .{} }) catch oom();
        return &self.items.items[at].w;
    }
};

/// `get_kv_str` in `llama_adapter_lora_init_impl`: the key's string value,
/// or empty when the key is absent.
fn kvStr(ctx: *const mod.impl.llama_c.gguf_context, key: mod.arch_table.Kv) []const u8 {
    var name = mod.arch.kvName(.unknown, key, null);
    var z: [mod.arch.max_name + 1]u8 = undefined;
    @memcpy(z[0..name.len], name.slice());
    z[name.len] = 0;
    const id = mod.impl.llama_c.gguf_find_key(ctx, &z);
    return if (id < 0) "" else mod.gguf_meta.cString(mod.impl.llama_c.gguf_get_val_str(ctx, id));
}

/// `get_kv_f32` in `llama_adapter_lora_init_impl`: the key's value, or 0.
fn kvF32(ctx: *const mod.impl.llama_c.gguf_context, key: mod.arch_table.Kv) f32 {
    var name = mod.arch.kvName(.unknown, key, null);
    var z: [mod.arch.max_name + 1]u8 = undefined;
    @memcpy(z[0..name.len], name.slice());
    z[name.len] = 0;
    const id = mod.impl.llama_c.gguf_find_key(ctx, &z);
    return if (id < 0) 0.0 else mod.impl.llama_c.gguf_get_val_f32(ctx, id);
}

/// `count(name)` on one of `llama_hadamard`'s maps.
fn blocksHas(m: ?stdlib.libcxx.StringMap(u32), name: []const u8) bool {
    const map = m orelse return false;
    return map.get(name) != null;
}

/// Ports `llama_adapter_lora_init_impl` (llama-adapter.cpp:151 @b1b770d90):
/// read a LoRA file from an open `FILE *`'s current position, check it was
/// made for this model's architecture, pair its `lora_a` and `lora_b`
/// tensors, check each pair against the weight it adapts, put each pair on
/// that weight's buffer type -- the CPU's where that is a repacking type --
/// and load the data.
///
/// Return: nothing, or `error.Thrown` with what the C++ throws. On success
/// the adapter is registered with the model.
fn loraInitImpl(model: *mod.model.Model, file: *mod.impl.llama_c.FILE, adapter: *Lora, exc: *mod.impl.Exception) mod.impl.Thrown!void {
    const func = "llama_adapter_lora_init_impl";

    var ctx_init: ?*mod.impl.llama_c.ggml_context = null;
    const ctx_gguf = mod.impl.llama_c.gguf_init_from_file_ptr(file, .{ .no_alloc = true, .ctx = @ptrCast(&ctx_init) }) orelse
        return exc.throw(.runtime_error, "failed to load lora adapter from file", .{});
    defer mod.impl.llama_c.gguf_free(ctx_gguf);

    // Freed before the metadata, as the C++'s locals are destroyed.
    const ctx = ctx_init;
    defer if (ctx) |p| mod.impl.llama_c.ggml_free(p);

    // must come after gguf_init_from_file_ptr, the llama_file constructor moves the file position
    var gguf_file = try mod.mmap.File.borrow(file, exc);
    defer gguf_file.close();

    // check metadata
    {
        mod.impl.logInfo("%s: Dumping metadata keys/values.\n", .{func});

        var i: c_int = 0;
        while (i < mod.impl.llama_c.gguf_get_n_kv(ctx_gguf)) : (i += 1) {
            const ty = mod.impl.llama_c.gguf_get_kv_type(ctx_gguf, i);
            var type_name: [128]u8 = undefined;
            const tn: [*:0]const u8 = if (ty == mod.impl.llama_c.GGUF_TYPE_ARRAY) blk: {
                _ = mod.impl.llama_c.snprintf(&type_name, type_name.len, "%s[%s,%zu]", mod.impl.llama_c.gguf_type_name(ty), mod.impl.llama_c.gguf_type_name(mod.impl.llama_c.gguf_get_arr_type(ctx_gguf, i)), mod.impl.llama_c.gguf_get_arr_n(ctx_gguf, i));
                break :blk @ptrCast(&type_name);
            } else mod.impl.llama_c.gguf_type_name(ty);
            const name = mod.impl.llama_c.gguf_get_key(ctx_gguf, i);

            var value: std.ArrayList(u8) = .empty;
            defer value.deinit(std.heap.c_allocator);
            const max_value_len = 40;
            mod.loader.kvToStr(ctx_gguf, i, max_value_len, &value);

            if (ty != mod.impl.llama_c.GGUF_TYPE_ARRAY) {
                // `emplace`: the first of a repeated key is kept.
                const slot = adapter.gguf_kv.getOrPut(mod.gguf_meta.cString(name), undefined) catch oom();
                if (!slot.found) slot.value.* = std.heap.c_allocator.dupeSentinel(u8, value.items, 0) catch oom();
            }

            if (value.items.len > max_value_len) {
                value.shrinkRetainingCapacity(max_value_len - 3);
                value.appendSlice(std.heap.c_allocator, "...") catch oom();
            }
            mod.impl.replaceAll(&value, "\n", "\\n");

            mod.impl.logInfo("%s: - kv %3d: %42s %-16s = %.*s\n", .{
                func, i, name, tn, @as(c_int, @intCast(value.items.len)), value.items.ptr,
            });
        }

        const general_type = kvStr(ctx_gguf, .general_type);
        if (!std.mem.eql(u8, general_type, "adapter")) {
            return exc.throw(.runtime_error, "expect general.type to be 'adapter', but got: {s}", .{general_type});
        }

        const general_arch_str = kvStr(ctx_gguf, .general_architecture);
        const general_arch = mod.arch.archFromString(general_arch_str);
        if (general_arch != model.arch) {
            return exc.throw(.runtime_error, "model arch and LoRA arch mismatch", .{});
        }

        const adapter_type = kvStr(ctx_gguf, .adapter_type);
        if (!std.mem.eql(u8, adapter_type, "lora")) {
            return exc.throw(.runtime_error, "expect adapter.type to be 'lora', but got: {s}", .{adapter_type});
        }

        adapter.alpha = kvF32(ctx_gguf, .adapter_lora_alpha);

        // parse alora invocation sequence vector
        var key = mod.arch.kvName(.unknown, .adapter_alora_invocation_tokens, null);
        var key_z: [mod.arch.max_name + 1]u8 = undefined;
        @memcpy(key_z[0..key.len], key.slice());
        key_z[key.len] = 0;
        const kid = mod.impl.llama_c.gguf_find_key(ctx_gguf, &key_z);
        if (kid >= 0) {
            if (mod.impl.llama_c.gguf_get_kv_type(ctx_gguf, kid) != mod.impl.llama_c.GGUF_TYPE_ARRAY) {
                return exc.throw(.runtime_error, "invalid gguf type for {s}", .{key.slice()});
            }
            const arr_type = mod.impl.llama_c.gguf_get_arr_type(ctx_gguf, kid);
            if (arr_type != mod.impl.llama_c.GGUF_TYPE_UINT32) {
                return exc.throw(.runtime_error, "invalid gguf element type for {s}", .{key.slice()});
            }
            const seq_len = mod.impl.llama_c.gguf_get_arr_n(ctx_gguf, kid);
            const data: [*]const mod.impl.llama_c.llama_token = @ptrCast(@alignCast(mod.impl.llama_c.gguf_get_arr_data(ctx_gguf, kid)));
            adapter.alora_invocation_tokens.resize(std.heap.c_allocator, seq_len) catch oom();
            @memcpy(adapter.alora_invocation_tokens.items, data[0..seq_len]);
        }
    }

    // `int n_tensors`: the count narrows.
    const n_tensors: c_int = @truncate(mod.impl.llama_c.gguf_get_n_tensors(ctx_gguf));
    const mem_size = @as(usize, @bitCast(@as(isize, n_tensors))) *% mod.impl.llama_c.ggml_tensor_overhead();

    // contexts for each buffer type
    var ctx_map: BuftCtxMap = .{};
    defer ctx_map.deinit();

    // bundle lora_a and lora_b into pairs
    var ab_map: PairMap = .{};
    defer ab_map.deinit();

    var cur: ?*mod.impl.llama_c.ggml_tensor = mod.impl.llama_c.ggml_get_first_tensor(ctx);
    while (cur) |t| : (cur = mod.impl.llama_c.ggml_get_next_tensor(ctx, t)) {
        var name: std.ArrayList(u8) = .empty;
        defer name.deinit(std.heap.c_allocator);
        name.appendSlice(std.heap.c_allocator, std.mem.sliceTo(&t.name, 0)) catch oom();
        if (endsWith(name.items, ".lora_a")) {
            mod.impl.replaceAll(&name, ".lora_a", "");
            ab_map.slot(name.items).a = t;
        } else if (endsWith(name.items, ".lora_b")) {
            mod.impl.replaceAll(&name, ".lora_b", "");
            ab_map.slot(name.items).b = t;
        } else if (endsWith(name.items, "_norm.weight")) {
            // TODO: add support for norm vector
            // for now, we don't really care because most adapters still work fine without it
            continue;
        } else {
            return exc.throw(.runtime_error, "LoRA tensor '{s}' has unexpected suffix", .{name.items});
        }
    }

    // get extra buffer types of the CPU
    // TODO: a more general solution for non-CPU extra buft should be implemented in the future
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12593#pullrequestreview-2718659948
    var buft_extra: std.ArrayList(mod.impl.llama_c.ggml_backend_buffer_type_t) = .empty;
    defer buft_extra.deinit(std.heap.c_allocator);
    {
        const cpu_dev = mod.impl.llama_c.ggml_backend_dev_by_type(mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_CPU) orelse
            return exc.throw(.runtime_error, "{s}: no CPU backend found", .{func});
        const cpu_reg = mod.impl.llama_c.ggml_backend_dev_backend_reg(cpu_dev);

        const get_extra: ?*const fn (mod.impl.llama_c.ggml_backend_dev_t) callconv(.c) [*c]mod.impl.llama_c.ggml_backend_buffer_type_t =
            @ptrCast(@alignCast(mod.impl.llama_c.ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts")));
        if (get_extra) |f| {
            var extra_bufts = f(cpu_dev);
            while (extra_bufts != null and extra_bufts[0] != null) : (extra_bufts += 1) {
                buft_extra.append(std.heap.c_allocator, extra_bufts[0]) catch oom();
            }
        }
    }

    // add tensors
    for (ab_map.items.items) |*it| {
        const name = it.name;
        const w = it.w;
        const is_token_embd = endsWith(name, "token_embd.weight");

        if (w.a == null or w.b == null) {
            return exc.throw(.runtime_error, "LoRA tensor pair for '{s}' is missing one component", .{name});
        }
        const wa = w.a.?;
        const wb = w.b.?;

        // device buft and device ctx
        const model_tensor = model.getTensor(name) orelse
            return exc.throw(.runtime_error, "LoRA tensor '{s}' does not exist in base model (hint: maybe wrong base model?)", .{name});
        // build_lora_mm feeds a folded weight's LoRA delta the primal input,
        // so folded matmul weights are fine; a latent lookup table
        // (token_embd, also the head when the output is tied) stores rotated
        // rows, refuse it
        if (blocksHas(model.hdmd.inverse_blocks, name)) {
            return exc.throw(.runtime_error, "LoRA tensor '{s}' targets a prism.hadamard latent lookup table, which is not supported", .{name});
        }
        const mt_name: [*:0]const u8 = @ptrCast(&model_tensor.name);

        var buft = mod.impl.llama_c.ggml_backend_buffer_get_type(model_tensor.buffer);

        // do not load loras to extra buffer types (i.e. bufts for repacking) -> use the CPU in that case
        for (buft_extra.items) |ex| {
            if (ex == buft) {
                mod.impl.logWarn("%s: lora for '%s' cannot use buft '%s', fallback to CPU\n", .{ func, mt_name, mod.impl.llama_c.ggml_backend_buft_name(buft) });

                const cpu_dev = mod.impl.llama_c.ggml_backend_dev_by_type(mod.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_CPU) orelse
                    return exc.throw(.runtime_error, "{s}: no CPU backend found", .{func});
                buft = mod.impl.llama_c.ggml_backend_dev_buffer_type(cpu_dev);
                break;
            }
        }

        mod.impl.logDebug("%s: lora for '%s' -> '%s'\n", .{ func, mt_name, mod.impl.llama_c.ggml_backend_buft_name(buft) });

        const dev_ctx = ctx_map.ctxForBuft(buft, mem_size, &adapter.ctxs);
        // validate tensor shape
        if (is_token_embd) {
            // expect B to be non-transposed, A and B are flipped; see llm_build_inp_embd()
            if (model_tensor.ne[0] != wb.ne[1] or model_tensor.ne[1] != wa.ne[1]) {
                return exc.throw(.runtime_error, "tensor '{s}' has incorrect shape (hint: maybe wrong base model?)", .{name});
            }
        } else {
            if (model_tensor.ne[0] != wa.ne[0] or model_tensor.ne[1] != wb.ne[1]) {
                return exc.throw(.runtime_error, "tensor '{s}' has incorrect shape (hint: maybe wrong base model?)", .{name});
            }
            if (wa.ne[1] != wb.ne[0]) {
                return exc.throw(.runtime_error, "lora_a tensor is not transposed (hint: adapter from \"finetune\" example is no longer supported)", .{});
            }
        }

        // save tensor to adapter
        const tensor_a = mod.impl.llama_c.ggml_dup_tensor(dev_ctx, wa);
        const tensor_b = mod.impl.llama_c.ggml_dup_tensor(dev_ctx, wb);
        _ = mod.impl.llama_c.ggml_set_name(tensor_a, &wa.name);
        _ = mod.impl.llama_c.ggml_set_name(tensor_b, &wb.name);
        const slot = adapter.ab_map.getOrPut(name, .{}) catch oom();
        slot.value.* = .{ .a = tensor_a, .b = tensor_b };
    }

    // allocate tensors / buffers and zero
    {
        adapter.ctxs.ensureTotalCapacity(std.heap.c_allocator, ctx_map.entries.items.len) catch oom();
        adapter.bufs.ensureTotalCapacity(std.heap.c_allocator, ctx_map.entries.items.len) catch oom();
        for (ctx_map.entries.items) |e| {
            const buf = mod.impl.llama_c.ggml_backend_alloc_ctx_tensors_from_buft(e.ctx, e.buft) orelse
                return exc.throw(.runtime_error, "failed to allocate buffer for lora adapter\n", .{});
            mod.impl.logInfo("%s: %10s LoRA buffer size = %8.2f MiB\n", .{
                func, mod.impl.llama_c.ggml_backend_buffer_name(buf), @as(f64, @floatFromInt(mod.impl.llama_c.ggml_backend_buffer_get_size(buf))) / 1024.0 / 1024.0,
            });
            adapter.bufs.append(std.heap.c_allocator, buf) catch oom();
        }
    }

    // set tensor data
    {
        var read_buf: std.ArrayList(u8) = .empty;
        defer read_buf.deinit(std.heap.c_allocator);
        const SetTensor = struct {
            fn run(g: *const mod.impl.llama_c.gguf_context, gf: *mod.mmap.File, buf: *std.ArrayList(u8), orig: *mod.impl.llama_c.ggml_tensor, dev: *mod.impl.llama_c.ggml_tensor, e: *mod.impl.Exception) mod.impl.Thrown!void {
                const offs = mod.impl.llama_c.gguf_get_data_offset(g) +% mod.impl.llama_c.gguf_get_tensor_offset(g, mod.impl.llama_c.gguf_find_tensor(g, &orig.name));
                const size = mod.impl.llama_c.ggml_nbytes(orig);
                if (offs +% size < offs or offs +% size > gf.size) {
                    return e.throw(.runtime_error, "LoRA tensor '{s}' data is not within the file bounds, file is corrupted or incomplete", .{std.mem.sliceTo(&orig.name, 0)});
                }
                buf.resize(std.heap.c_allocator, size) catch oom();
                try gf.seek(offs, mod.impl.llama_c.SEEK_SET, e);
                try gf.readRaw(buf.items.ptr, size, e);
                mod.impl.llama_c.ggml_backend_tensor_set(dev, buf.items.ptr, 0, size);
            }
        };
        var it = adapter.ab_map.iterator();
        while (it.next()) |e| {
            const orig = ab_map.find(e.key).?.*;
            const dev = e.value.*;
            try SetTensor.run(ctx_gguf, &gguf_file, &read_buf, orig.a.?, dev.a.?, exc);
            try SetTensor.run(ctx_gguf, &gguf_file, &read_buf, orig.b.?, dev.b.?, exc);
        }
    }

    // register adapter with model
    model.loras.append(std.heap.c_allocator, adapter) catch oom();

    mod.impl.logInfo("%s: loaded %zu tensors from lora file\n", .{ func, adapter.ab_map.count() * 2 });
}

/// Ports `llama_adapter_lora_init` (llama-adapter.cpp:430 @b1b770d90):
/// open the file and read it through `loraInitFromFilePtr`.
///
/// Parameters:
/// - `model`: the model to adapt; it owns the adapter from here on.
/// - `path_lora`: the LoRA file.
///
/// Return: the adapter, or null with the reason logged.
pub fn loraInit(model: *mod.model.Model, path_lora: [*:0]const u8) ?*Lora {
    const func = "llama_adapter_lora_init";
    mod.impl.logInfo("%s: loading lora adapter from '%s' ...\n", .{ func, path_lora });

    const file = mod.impl.llama_c.ggml_fopen(path_lora, "rb") orelse {
        mod.impl.logError("%s: failed to open '%s': %s\n", .{ func, path_lora, stdlib.errno.strerror(std.c._errno().*) });
        return null;
    };

    const adapter = loraInitFromFilePtr(model, file);
    _ = mod.impl.llama_c.fclose(file);

    return adapter;
}

/// Ports `llama_adapter_lora_init_from_file_ptr` (llama-adapter.cpp:445
/// @b1b770d90): read a LoRA adapter from an open file, starting at its
/// current position.
///
/// Parameters:
/// - `model`: the model to adapt; it owns the adapter from here on.
/// - `file`: the open file; borrowed, never closed here. Null is refused.
///
/// Return: the adapter, or null with the reason logged.
pub fn loraInitFromFilePtr(model: *mod.model.Model, file: ?*mod.impl.llama_c.FILE) ?*Lora {
    const func = "llama_adapter_lora_init_from_file_ptr";
    const fp = file orelse {
        mod.impl.logError("%s: file is NULL\n", .{func});
        return null;
    };

    const adapter = Lora.create(model);
    var exc: mod.impl.Exception = .{};
    loraInitImpl(model, fp, adapter, &exc) catch {
        mod.impl.logError("%s: failed to apply lora adapter: %s\n", .{ func, exc.whatZ() });
        adapter.destroy();
        return null;
    };
    return adapter;
}

/// Ports `llama_adapter_lora_free` (llama-adapter.cpp:504 @b1b770d90):
/// detach the adapter from its model, then free it.
///
/// Return: nothing.
pub fn loraFree(adapter: ?*Lora) void {
    const a = adapter orelse return;
    if (a.model) |m| {
        // `std::unordered_set::erase`: whatever order the set keeps is
        // read only to sum node counts, so the list is not ordered.
        for (m.loras.items, 0..) |l, i| {
            if (l == a) {
                _ = m.loras.swapRemove(i);
                break;
            }
        }
        a.model = null;
    }
    a.destroy();
}

/// Ports the destructor of `llama_model` (llama-model.cpp:1451 @b1b770d90):
/// every adapter still registered with the model is freed with it.
///
/// Return: nothing.
pub fn freeModelLoras(model: *mod.model.Model) void {
    for (model.loras.items) |l| l.destroy();
    model.loras.clearRetainingCapacity();
}

/// `llama_adapter_loras` (llama-adapter.h:90 @b1b770d90): the active
/// adapters and their scales, in the order the C++'s
/// `std::unordered_map<llama_adapter_lora *, float>` iterates them, which
/// is the order the graph adds their deltas in.
///
/// `items` is that order, rebuilt after every insertion from `map`, which
/// lays the entries out as libc++ does: keyed by the pointer's eight
/// bytes, because libc++'s `hash<T *>` hashes them with `__hash_memory`.
pub const Loras = struct {
    items: std.ArrayList(Entry) = .empty,
    map: ?stdlib.libcxx.StringMap(f32) = null,

    pub const Entry = struct { lora: *Lora, scale: f32 };

    /// Return: nothing.
    pub fn deinit(self: *Loras) void {
        self.items.deinit(std.heap.c_allocator);
        if (self.map) |*m| m.deinit();
    }

    fn keyOf(lora: *const ?*const Lora) []const u8 {
        return std.mem.asBytes(lora);
    }

    /// `insert({lora, scale})`: a no-op when `lora` is already there.
    ///
    /// Return: nothing.
    pub fn insert(self: *Loras, lora: *Lora, scale: f32) void {
        if (self.map == null) self.map = stdlib.libcxx.StringMap(f32).init(std.heap.c_allocator) catch oom();
        const k: ?*const Lora = lora;
        _ = self.map.?.getOrPut(keyOf(&k), scale) catch oom();
        self.items.clearRetainingCapacity();
        var it = self.map.?.iterator();
        while (it.next()) |e| {
            const p: usize = std.mem.readInt(usize, e.key[0..@sizeOf(usize)], .little);
            self.items.append(std.heap.c_allocator, .{ .lora = @ptrFromInt(p), .scale = e.value.* }) catch oom();
        }
    }

    /// `find(lora)`.
    ///
    /// Return: the adapter's scale, or null when it is not active.
    pub fn find(self: *const Loras, lora: ?*const Lora) ?f32 {
        const m = &(self.map orelse return null);
        const k: ?*const Lora = lora;
        const v = m.get(keyOf(&k)) orelse return null;
        return v.*;
    }

    /// `size()`.
    pub fn count(self: *const Loras) usize {
        return self.items.items.len;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Cvec);
    std.testing.refAllDecls(LoraWeight);
    std.testing.refAllDecls(Lora);
    std.testing.refAllDecls(Loras);
}

test "PairMap keeps std::map's byte order" {
    var m: PairMap = .{};
    defer m.deinit();
    m.slot("blk.1.b").a = null;
    m.slot("blk.1").a = null;
    m.slot("blk.10").a = null;
    m.slot("blk.1.a").a = null;
    m.slot("blk.1").b = null;
    try std.testing.expectEqual(@as(usize, 4), m.items.items.len);
    try std.testing.expectEqualStrings("blk.1", m.items.items[0].name);
    try std.testing.expectEqualStrings("blk.1.a", m.items.items[1].name);
    try std.testing.expectEqualStrings("blk.1.b", m.items.items[2].name);
    try std.testing.expectEqualStrings("blk.10", m.items.items[3].name);
}

test "Loras inserts once and finds by pointer" {
    var l: Loras = .{};
    defer l.deinit();
    const a: *Lora = @ptrFromInt(0x1000);
    const b: *Lora = @ptrFromInt(0x2040);
    l.insert(a, 0.5);
    l.insert(b, 2.0);
    l.insert(a, 9.0);
    try std.testing.expectEqual(@as(usize, 2), l.count());
    try std.testing.expectEqual(@as(?f32, 0.5), l.find(a));
    try std.testing.expectEqual(@as(?f32, null), l.find(@ptrFromInt(0x3000)));
}
