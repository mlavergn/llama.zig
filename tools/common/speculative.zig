//! `common_speculative`: speculative decoding -- the drafters a target
//! model's guesses come from, chained in order of preference, and the
//! bookkeeping around them.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/speculative.cpp` and
//! `llama.cpp/common/speculative.h` (v0.7.0, `620d2225c`): the type names,
//! the parameters' helpers, the four n-gram drafters, the driver and its
//! statistics. The four drafters that run a second model are
//! `speculative_draft.zig`; the n-gram stores under them are
//! `ngram_cache.zig`, `ngram_map.zig` and `ngram_mod.zig`. Each declaration
//! names what it ports and the line it began at.
//!
//! # Exceptions
//!
//! Where the C++ throws `std::runtime_error` or `std::invalid_argument`
//! these return `error.Runtime` or `error.InvalidArgument`, the text in
//! `arg.what()`; a drafter built before the throw is freed, as the C++'s
//! `unique_ptr`s free it.
//!
//! # Drafts
//!
//! A `DraftParams.result` is appended to with `std.heap.c_allocator`; the
//! caller owns the list and frees it with the same allocator. So is a
//! `DraftParams.result_q`, the candidates of each drafted token, whose inner
//! lists the drafters own until they are freed with `freeCandidates`.
//!
//! # `llama-ext.h`
//!
//! The extraction and multi-token-prediction calls are C++-linkage in
//! upstream and not in `tools_h`; `ext` reaches the port's own functions of
//! those names through libllama's module.

const std = @import("std");
const c = @import("tools_h");
const llama = @import("llama");
const params_mod = @import("params.zig");
const common = @import("common.zig");
const log = @import("log.zig");
const arg = @import("arg.zig");
const ngram_cache = @import("ngram_cache.zig");
const ngram_map = @import("ngram_map.zig");
const ngram_mod = @import("ngram_mod.zig");
const draft_impls = @import("speculative_draft.zig");

const Type = params_mod.SpeculativeType;
const Params = params_mod.Params;
const SpecParams = params_mod.Speculative;

const gpa = std.heap.c_allocator;

fn assertC(ok: bool, comptime line: c_int, comptime expr: [*:0]const u8) void {
    if (!ok) _ = c.ggml_abort("speculative.cpp", line, "GGML_ASSERT(%s) failed", expr);
}

fn oom() noreturn {
    _ = c.ggml_abort("speculative.cpp", 0, "out of memory");
    unreachable;
}

/// The `llama-ext.h` entry points `common/speculative.cpp` calls, through
/// the port's libllama under its own `llama.h` translation.
pub const ext = struct {
    const api = llama.api;

    pub fn setEmbeddingsNextn(ctx: ?*c.llama_context, value: bool, masked: bool) void {
        api.context.llama_set_embeddings_nextn(@ptrCast(ctx), value, masked);
    }
    pub fn setNextnLayerOffset(ctx: ?*c.llama_context, offset: i32) void {
        api.context.llama_set_nextn_layer_offset(@ptrCast(ctx), offset);
    }
    pub fn getEmbeddingsNextn(ctx: ?*c.llama_context) ?[*]f32 {
        return api.context.llama_get_embeddings_nextn(@ptrCast(ctx));
    }
    pub fn getEmbeddingsNextnIth(ctx: ?*c.llama_context, i: i32) ?[*]f32 {
        return api.context.llama_get_embeddings_nextn_ith(@ptrCast(ctx), i);
    }
    pub fn setEmbeddingsLayerInp(ctx: ?*c.llama_context, lid: u32, value: bool) void {
        api.context.llama_set_embeddings_layer_inp(@ptrCast(ctx), lid, value);
    }
    pub fn getEmbeddingsLayerInp(ctx: ?*c.llama_context, lid: u32) ?[*]f32 {
        return api.context.llama_get_embeddings_layer_inp(@ptrCast(ctx), lid);
    }
    pub fn modelTargetLayerIds(model: ?*const c.llama_model) [*c]const i32 {
        return api.model.llama_model_target_layer_ids(@ptrCast(model));
    }
    pub fn modelTargetLayerIdsN(model: ?*const c.llama_model) u32 {
        return api.model.llama_model_target_layer_ids_n(@ptrCast(model));
    }
    pub fn modelDflashSelectorTopK(model: ?*const c.llama_model) i32 {
        return api.model.llama_model_dflash_selector_top_k(@ptrCast(model));
    }
    pub fn getCtxOther(ctx: ?*c.llama_context) ?*c.llama_context {
        return @ptrCast(api.context.llama_get_ctx_other(@ptrCast(ctx)));
    }
};

// Ports `SPC_DBG`, `SPC_TRC`, `SPC_INF`, `SPC_WRN`, `SPC_ERR`
// (speculative.cpp:23, 24, 25, 26, 27 @620d2225c): the `LOG_*` macros with
// the calling function's name, as `spec %12.*s: `.

pub fn spcDbg(comptime func: [*:0]const u8, comptime fmt: []const u8, args: anytype) void {
    log.dbg("spec %12.*s: " ++ fmt, .{ @as(c_int, 12), func } ++ args);
}
pub fn spcTrc(comptime func: [*:0]const u8, comptime fmt: []const u8, args: anytype) void {
    log.trc("spec %12.*s: " ++ fmt, .{ @as(c_int, 12), func } ++ args);
}
pub fn spcInf(comptime func: [*:0]const u8, comptime fmt: []const u8, args: anytype) void {
    log.inf("spec %12.*s: " ++ fmt, .{ @as(c_int, 12), func } ++ args);
}
pub fn spcWrn(comptime func: [*:0]const u8, comptime fmt: []const u8, args: anytype) void {
    log.wrn("spec %12.*s: " ++ fmt, .{ @as(c_int, 12), func } ++ args);
}
pub fn spcErr(comptime func: [*:0]const u8, comptime fmt: []const u8, args: anytype) void {
    log.err("spec %12.*s: " ++ fmt, .{ @as(c_int, 12), func } ++ args);
}

/// What the C++ throws out of `common_speculative_init` and its helpers.
pub const InitError = error{ Runtime, InvalidArgument, OutOfMemory };

/// `throw std::runtime_error(msg)`.
pub fn throwRuntime(msg: []const u8) InitError {
    switch (arg.throwStr(msg)) {
        error.Throw => {},
    }
    return error.Runtime;
}

/// `throw std::runtime_error(string_format(fmt, args))`.
pub fn throwRuntimeFmt(comptime fmt: [*:0]const u8, args: anytype) InitError {
    arg.setWhat(fmt, args);
    return error.Runtime;
}

/// `throw std::invalid_argument(string_format(fmt, args))`.
fn throwInvalid(comptime fmt: [*:0]const u8, args: anytype) InitError {
    arg.setWhat(fmt, args);
    return error.InvalidArgument;
}

/// Ports `common_speculative_type_from_name_map` (speculative.cpp:72
/// @620d2225c), a `std::map`: sorted by name.
const type_from_name_map = [_]struct { [:0]const u8, Type }{
    .{ "draft-dflash", .draft_dflash },
    .{ "draft-dspark", .draft_dspark },
    .{ "draft-eagle3", .draft_eagle3 },
    .{ "draft-mtp", .draft_mtp },
    .{ "draft-simple", .draft_simple },
    .{ "ngram-cache", .ngram_cache },
    .{ "ngram-map-k", .ngram_map_k },
    .{ "ngram-map-k4v", .ngram_map_k4v },
    .{ "ngram-mod", .ngram_mod },
    .{ "ngram-simple", .ngram_simple },
    .{ "none", .none },
};

fn findType(name: []const u8) ?Type {
    for (type_from_name_map) |e| {
        if (std.mem.eql(u8, e[0], name)) return e[1];
    }
    return null;
}

/// Ports `common_speculative_get_devices_str` (speculative.cpp:86
/// @620d2225c): the named devices, or `default`.
///
/// Return: the text, NUL-terminated, `c_allocator`'s.
pub fn devicesStr(devices: []const c.ggml_backend_dev_t) [:0]u8 {
    var result: std.ArrayList(u8) = .empty;
    for (devices) |d| {
        if (d == null) continue;
        if (result.items.len != 0) result.appendSlice(gpa, ", ") catch oom();
        result.appendSlice(gpa, std.mem.span(c.ggml_backend_dev_name(d))) catch oom();
    }
    if (result.items.len == 0) result.appendSlice(gpa, "default") catch oom();
    return result.toOwnedSliceSentinel(gpa, 0) catch oom();
}

/// Ports `common_speculative_are_compatible` (speculative.cpp:106
/// @620d2225c), exported from `speculative.h` since v0.6.0: the draft's vocabulary must be the target's, give or take
/// `SPEC_VOCAB_MAX_SIZE_DIFFERENCE` tokens at the end.
pub fn areCompatible(model_tgt: ?*const c.llama_model, model_dft: ?*const c.llama_model) bool {
    const func = "common_speculative_are_compatible";
    const max_size_difference = 128; // SPEC_VOCAB_MAX_SIZE_DIFFERENCE
    const check_start_token_id = 5; // SPEC_VOCAB_CHECK_START_TOKEN_ID

    const vocab_tgt = c.llama_model_get_vocab(model_tgt);
    const vocab_dft = c.llama_model_get_vocab(model_dft);

    const vocab_type_tgt = c.llama_vocab_type(vocab_tgt);
    spcDbg(func, "vocab_type tgt: %d\n", .{@as(c_int, @intCast(vocab_type_tgt))});

    const vocab_type_dft = c.llama_vocab_type(vocab_dft);
    spcDbg(func, "vocab_type dft: %d\n", .{@as(c_int, @intCast(vocab_type_dft))});

    if (vocab_type_tgt != vocab_type_dft) {
        spcWrn(func, "draft model vocab type must match target model to use speculation but " ++
            "vocab_type_dft = %d while vocab_type_tgt = %d\n", .{ @as(c_int, @intCast(vocab_type_dft)), @as(c_int, @intCast(vocab_type_tgt)) });
        return false;
    }

    if (c.llama_vocab_get_add_bos(vocab_tgt) != c.llama_vocab_get_add_bos(vocab_dft) or
        (c.llama_vocab_get_add_bos(vocab_tgt) and c.llama_vocab_bos(vocab_tgt) != c.llama_vocab_bos(vocab_dft)))
    {
        spcWrn(func, "draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n", .{
            @as(c_int, @intFromBool(c.llama_vocab_get_add_bos(vocab_tgt))), @as(c_int, @intFromBool(c.llama_vocab_get_add_bos(vocab_dft))),
            c.llama_vocab_bos(vocab_tgt),                                   c.llama_vocab_bos(vocab_dft),
        });
        return false;
    }

    if (c.llama_vocab_get_add_eos(vocab_tgt) != c.llama_vocab_get_add_eos(vocab_dft) or
        (c.llama_vocab_get_add_eos(vocab_tgt) and c.llama_vocab_eos(vocab_tgt) != c.llama_vocab_eos(vocab_dft)))
    {
        spcWrn(func, "draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n", .{
            @as(c_int, @intFromBool(c.llama_vocab_get_add_eos(vocab_tgt))), @as(c_int, @intFromBool(c.llama_vocab_get_add_eos(vocab_dft))),
            c.llama_vocab_eos(vocab_tgt),                                   c.llama_vocab_eos(vocab_dft),
        });
        return false;
    }

    {
        const n_vocab_tgt = c.llama_vocab_n_tokens(vocab_tgt);
        const n_vocab_dft = c.llama_vocab_n_tokens(vocab_dft);
        const vocab_diff = if (n_vocab_tgt > n_vocab_dft) n_vocab_tgt - n_vocab_dft else n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > max_size_difference) {
            spcDbg(func, "draft model vocab must closely match target model to use speculation but " ++
                "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n", .{ n_vocab_tgt, c.llama_vocab_n_tokens(vocab_dft), vocab_diff, @as(c_int, max_size_difference) });
            return false;
        }

        var i: i32 = check_start_token_id;
        while (i < @min(n_vocab_tgt, n_vocab_dft)) : (i += 1) {
            const token_text_tgt = c.llama_vocab_get_text(vocab_tgt, i);
            const token_text_dft = c.llama_vocab_get_text(vocab_dft, i);

            if (std.mem.orderZ(u8, token_text_tgt, token_text_dft) != .eq) {
                if (log.verbosity_thold >= log.level_debug) {
                    const pt = common.tokenToPiece(gpa, vocab_tgt, i, true) catch oom();
                    defer gpa.free(pt[0 .. pt.len + 1]);
                    const pd = common.tokenToPiece(gpa, vocab_dft, i, true) catch oom();
                    defer gpa.free(pd[0 .. pd.len + 1]);
                    spcDbg(func, "draft model vocab must match target model to use speculation but " ++
                        "token %d content differs - target '%s', draft '%s'\n", .{ i, pt.ptr, pd.ptr });
                }
                return false;
            }
        }
    }

    return true;
}

/// Candidate distributions, one per drafted token: the C++'s
/// `std::vector<std::vector<llama_token_data>>`, over `std.heap.c_allocator`.
pub const Candidates = std.ArrayList(std.ArrayList(c.llama_token_data));

/// Frees the inner lists of `q` past `n`, then shrinks it to `n` -- the
/// C++'s `resize` down, or `clear` at 0.
pub fn truncateCandidates(q: *Candidates, n: usize) void {
    if (n >= q.items.len) return;
    for (q.items[n..]) |*e| e.deinit(gpa);
    q.shrinkRetainingCapacity(n);
}

/// The C++'s `resize(n)`: shrinks as `truncateCandidates`, or grows with
/// empty lists.
fn resizeCandidates(q: *Candidates, n: usize) void {
    if (n <= q.items.len) return truncateCandidates(q, n);
    q.appendNTimes(gpa, .empty, n - q.items.len) catch oom();
}

/// Frees `q` and everything in it.
pub fn freeCandidates(q: *Candidates) void {
    truncateCandidates(q, 0);
    q.deinit(gpa);
}

/// Ports `common_speculative_draft_params` (speculative.h:56 @620d2225c):
/// what one sequence asks of a `draft` call.
pub const DraftParams = struct {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    drafting: bool = false,

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    n_max: i32 = -1,

    pos0: c.llama_pos = 0,
    id_last: c.llama_token = 0,

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    prompt: []const c.llama_token = &.{},

    // the generated draft from the last _draft() call
    result: ?*std.ArrayList(c.llama_token) = null,

    // candidate distribution per drafted token; set it to make draft-simple and draft-mtp sample
    result_q: ?*Candidates = null,

    // the target's temp and seed, read only when the drafter samples probabilistically
    temp: f32 = 1.0,
    seed: u32 = c.LLAMA_DEFAULT_SEED,
};

/// Ports `common_speculative_impl` (speculative.cpp:177 @620d2225c): what
/// every drafter shares -- its type, its limit and its statistics.
pub const Base = struct {
    type: Type,

    n_seq: u32,
    n_max: i32, // maximum draft length after implementation-specific limits

    n_call_begin: usize = 0, // number of times this implementation was called for refresh.
    n_call_draft: usize = 0, // number of times this implementation was called for generation.
    n_call_accept: usize = 0, // number of times this implementation was called for accumulation.

    n_gen_drafts: usize = 0, // number of times a draft or part was generated by this implementation.
    n_acc_drafts: usize = 0, // number of times a draft or part was accepted by the target model.
    n_gen_tokens: usize = 0, // number of tokens generated by this implementation.
    n_acc_tokens: usize = 0, // number of tokens accepted by the target model.

    n_acc_tokens_per_pos: std.ArrayList(usize) = .empty, // number of tokens accepted per draft position.

    gen_perf: bool = true, // whether to generate performance stats.

    t_begin_us: i64 = 0, // total time spent in refresh of this implementation in microseconds.
    t_draft_us: i64 = 0, // total time spent in generating drafts in this implementation in microseconds.
    t_accept_us: i64 = 0, // total time spent in accumulation of this implementation in microseconds.
};

/// `common_time_meas` (common/common.cpp:69 @620d2225c): the start, or -1
/// when disabled.
fn timeStart(disable: bool) i64 {
    return if (disable) -1 else c.ggml_time_us();
}

/// The `common_time_meas` destructor (common/common.cpp:71 @620d2225c).
fn timeEnd(t_start_us: i64, t_acc: *i64) void {
    if (t_start_us >= 0) t_acc.* += c.ggml_time_us() - t_start_us;
}

// -----------------------------------------------------------------------------
// the n-gram drafters

/// Ports `common_speculative_impl_ngram_simple` (speculative.cpp:1876
/// @620d2225c): self-speculation by looking the last n-gram up in the
/// history.
pub const NgramSimple = struct {
    params: params_mod.SpeculativeNgramMap,

    // shared across all sequences
    config: ngram_map.SimpleConfig,

    /// Ports `common_speculative_impl_ngram_simple` (speculative.cpp:1882
    /// @620d2225c), the constructor.
    fn init(params: *const SpecParams, config: ngram_map.SimpleConfig) NgramSimple {
        const func = "common_speculative_impl_ngram_simple";
        const p = params.ngram_simple;
        spcTrc(func, "%s", .{@as([*:0]const u8, "adding speculative implementation 'ngram-simple'\n")});
        spcTrc(func, "- size_n=%d, size_m=%d, min_hits=%d\n", .{ @as(c_int, p.size_n), @as(c_int, p.size_m), @as(c_int, p.min_hits) });
        return .{ .params = p, .config = config };
    }

    fn deinit(_: *NgramSimple) void {}

    /// Ports `begin` (speculative.cpp:1894 @620d2225c): nothing to do.
    fn begin(_: *NgramSimple, _: *Base, _: c.llama_seq_id, _: []const c.llama_token) void {}

    /// Ports `process` (speculative.cpp:1898 @620d2225c): nothing to do.
    fn process(_: *NgramSimple, _: *Base, _: *const common.Batch) bool {
        return true;
    }

    /// Ports `draft` (speculative.cpp:1903 @620d2225c).
    fn draft(self: *NgramSimple, base: *Base, dparams: []DraftParams) void {
        std.debug.assert(dparams.len == base.n_seq);
        for (dparams) |*dp| {
            if (!dp.drafting) continue;
            var r = ngram_map.simpleDraft(gpa, self.config, dp.prompt, dp.id_last) catch oom();
            defer r.deinit(gpa);
            dp.result.?.clearRetainingCapacity();
            dp.result.?.appendSlice(gpa, r.items) catch oom();
        }
    }

    /// Ports `accept` (speculative.cpp:1916 @620d2225c): nothing to do.
    fn accept(_: *NgramSimple, _: *Base, _: c.llama_seq_id, _: u16, _: bool) void {}
};

/// Ports `common_speculative_impl_ngram_map_k` (speculative.cpp:1921
/// @620d2225c): `ngram-map-k` and `ngram-map-k4v`, one map per sequence.
pub const NgramMapK = struct {
    // n_seq configs
    config: []ngram_map.Map,

    /// Ports `common_speculative_impl_ngram_map_k` (speculative.cpp:1925
    /// @620d2225c), the constructor: `n_seq` copies of `config`.
    fn init(base: *const Base, config: *const ngram_map.Map) NgramMapK {
        const func = "common_speculative_impl_ngram_map_k";
        const maps = gpa.alloc(ngram_map.Map, base.n_seq) catch oom();
        for (maps) |*m| m.* = config.clone(gpa) catch oom();

        spcTrc(func, "adding speculative implementation '%s'\n", .{typeToStr(base.type).ptr});
        spcTrc(func, "- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n", .{ @as(c_int, config.size_key), @as(c_int, config.size_value), @as(c_int, @intFromBool(config.key_only)), @as(c_int, config.min_hits) });
        return .{ .config = maps };
    }

    fn deinit(self: *NgramMapK) void {
        for (self.config) |*m| m.deinit();
        gpa.free(self.config);
    }

    /// Ports `begin` (speculative.cpp:1940 @620d2225c).
    fn begin(self: *NgramMapK, base: *Base, seq_id: c.llama_seq_id, prompt: []const c.llama_token) void {
        assertC(seq_id < @as(c.llama_seq_id, @intCast(base.n_seq)), 1896, "seq_id < (llama_seq_id) n_seq");

        ngram_map.begin(&self.config[@intCast(seq_id)], prompt);
    }

    /// Ports `process` (speculative.cpp:1946 @620d2225c): nothing to do.
    fn process(_: *NgramMapK, _: *Base, _: *const common.Batch) bool {
        return true;
    }

    /// Ports `draft` (speculative.cpp:1951 @620d2225c).
    fn draft(self: *NgramMapK, base: *Base, dparams: []DraftParams) void {
        std.debug.assert(dparams.len == base.n_seq);
        for (dparams, 0..) |*dp, s| {
            if (!dp.drafting) continue;
            ngram_map.draft(&self.config[s], dp.prompt, dp.id_last, dp.result.?) catch oom();
        }
    }

    /// Ports `accept` (speculative.cpp:1964 @620d2225c).
    fn accept(self: *NgramMapK, _: *Base, seq_id: c.llama_seq_id, n_accepted: u16, is_other: bool) void {
        assertC(seq_id < @as(c.llama_seq_id, @intCast(self.config.len)), 1920, "(seq_id < (llama_seq_id) config.size())");

        if (is_other) return;

        ngram_map.accept(&self.config[@intCast(seq_id)], n_accepted);
    }
};

/// Ports `common_speculative_impl_ngram_mod` (speculative.cpp:1975
/// @620d2225c): one hashed n-gram table shared by every sequence.
pub const NgramMod = struct {
    params: params_mod.SpeculativeNgramMod,

    // shared across all sequences
    mod: ngram_mod.Mod,

    // enable trace logging if LLAMA_TRACE is set
    verbose: bool,

    sinfos: []SeqInfo,

    /// Ports `seq_info` (speculative.cpp:1984 @620d2225c), a member of
    /// `common_speculative_impl_ngram_mod`.
    const SeqInfo = struct {
        // the last position in the prompt that was added to the ngram container
        i_last: usize = 0,

        // length of the last drafted n-gram (number of tokens returned by draft)
        n_draft_last: usize = 0,

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        n_low: c_int = 0,
    };

    /// Ports `common_speculative_impl_ngram_mod` (speculative.cpp:1997
    /// @620d2225c), the constructor.
    fn init(base: *const Base, params: *const SpecParams) NgramMod {
        const func = "common_speculative_impl_ngram_mod";
        const p = params.ngram_mod;
        const mod = ngram_mod.Mod.init(gpa, @truncate(@as(u32, @bitCast(p.n_match))), 4 * 1024 * 1024) catch oom();

        spcTrc(func, "%s", .{@as([*:0]const u8, "adding speculative implementation 'ngram-mod'\n")});
        spcTrc(func, "- n_match=%d, n_max=%d, n_min=%d\n", .{ p.n_match, p.n_max, p.n_min });
        const mb: f32 = @as(f32, @floatFromInt(mod.sizeBytes())) / 1024 / 1024;
        spcTrc(func, "- mod size=%zu (%.3f MB)\n", .{ mod.size(), @as(f64, mb) });

        if (p.n_match < 16) {
            spcWrn(func, "ngram_mod n_match=%d is too small - poor quality is possible, " ++
                "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", .{p.n_match});
        }

        const sinfos = gpa.alloc(SeqInfo, base.n_seq) catch oom();
        @memset(sinfos, .{});
        return .{ .params = p, .mod = mod, .verbose = c.getenv("LLAMA_TRACE") != null, .sinfos = sinfos };
    }

    fn deinit(self: *NgramMod) void {
        self.mod.deinit();
        gpa.free(self.sinfos);
    }

    /// Ports `begin` (speculative.cpp:2020 @620d2225c): the prompt's
    /// n-grams into the table, which resets past a quarter full.
    fn begin(self: *NgramMod, _: *Base, seq_id: c.llama_seq_id, prompt: []const c.llama_token) void {
        const sinfo = &self.sinfos[@intCast(seq_id)];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const n = self.mod.getN();
        if (prompt.len < n) return;

        var i: usize = 0;
        while (i < prompt.len - n) : (i += 1) self.mod.add(prompt.ptr + i);

        sinfo.i_last = prompt.len - n;

        const f = @as(f64, @floatFromInt(self.mod.getUsed())) / @as(f64, @floatFromInt(self.mod.size()));
        spcTrc("begin", "ngram_mod occupancy = %zu/%zu (%.2f)\n", .{ self.mod.getUsed(), self.mod.size(), f });

        const f_thold = 0.25;
        if (f > f_thold) {
            spcWrn("begin", "ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", .{ f, @as(f64, f_thold) });

            self.mod.reset();
        }
    }

    /// Ports `draft_one` (speculative.cpp:2048 @620d2225c), a member of
    /// `common_speculative_impl_ngram_mod`.
    fn draftOne(self: *NgramMod, seq_id: usize, dparams: *DraftParams) void {
        const sinfo = &self.sinfos[seq_id];
        const result = dparams.result.?;

        const prompt = dparams.prompt;

        sinfo.n_draft_last = 0;

        const cur_len = prompt.len;
        if (cur_len < self.mod.getN()) return;

        const n = self.mod.getN();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            var i = sinfo.i_last;
            while (i < cur_len - n) : (i += 1) self.mod.add(prompt.ptr + i);

            sinfo.i_last = cur_len - n;
        }

        const n_max: usize = @intCast(self.params.n_max);
        const old = result.items.len;
        result.resize(gpa, n + n_max) catch oom();
        if (result.items.len > old) @memset(result.items[old..], 0);
        for (0..n - 1) |i| result.items[i] = prompt[cur_len - n + 1 + i];
        result.items[n - 1] = dparams.id_last;

        var i: usize = 0;
        while (i < n_max) : (i += 1) {
            const token = self.mod.get(result.items.ptr + i);
            if (token == ngram_mod.Mod.empty_entry) {
                if (@as(i64, @intCast(i)) < self.params.n_min) {
                    result.clearRetainingCapacity();
                    return;
                }

                result.shrinkRetainingCapacity(n + i);
                break;
            }
            result.items[n + i] = token;
        }

        // only return the m tokens that were drafted
        i = 0;
        while (n + i < result.items.len) : (i += 1) result.items[i] = result.items[n + i];
        result.shrinkRetainingCapacity(result.items.len - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.items.len;
    }

    /// Ports `process` (speculative.cpp:2104 @620d2225c): nothing to do.
    fn process(_: *NgramMod, _: *Base, _: *const common.Batch) bool {
        return true;
    }

    /// Ports `draft` (speculative.cpp:2109 @620d2225c).
    fn draft(self: *NgramMod, base: *Base, dparams: []DraftParams) void {
        std.debug.assert(dparams.len == base.n_seq);
        for (dparams, 0..) |*dp, s| {
            if (!dp.drafting) continue;
            self.draftOne(s, dp);
        }
    }

    /// Ports `accept` (speculative.cpp:2122 @620d2225c): five rounds in a
    /// row under a quarter accepted reset the table.
    fn accept(self: *NgramMod, _: *Base, seq_id: c.llama_seq_id, n_accepted: u16, is_other: bool) void {
        if (is_other) return;

        const sinfo = &self.sinfos[@intCast(seq_id)];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const f_acc = @as(f64, @floatFromInt(n_accepted)) / @as(f64, @floatFromInt(sinfo.n_draft_last));
            if (f_acc < 0.25) {
                sinfo.n_low += 1;
                if (sinfo.n_low >= 5) {
                    if (self.verbose) {
                        spcTrc("accept", "low acceptance streak (%d) - resetting ngram_mod\n", .{sinfo.n_low});
                    }

                    self.mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
};

/// Ports `common_speculative_impl_ngram_cache` (speculative.cpp:2150
/// @620d2225c): the three-level n-gram cache -- context, dynamic, static.
pub const NgramCache = struct {
    params: params_mod.SpeculativeNgramCache,

    n_draft: u16,

    save_dynamic: bool,
    save_static: bool,

    sinfos: []SeqInfo,

    /// Ports `seq_info` (speculative.cpp:2158 @620d2225c), a member of
    /// `common_speculative_impl_ngram_cache`.
    const SeqInfo = struct {
        cache_size: usize = 0, // number of tokens in n-gram cache

        ngram_cache_context: ngram_cache.Cache,
        ngram_cache_dynamic: ngram_cache.Cache,
        ngram_cache_static: ngram_cache.Cache,
    };

    /// Ports `common_speculative_impl_ngram_cache` (speculative.cpp:2168
    /// @620d2225c), the constructor: each sequence gets a copy of the
    /// static and dynamic caches the paths name.
    fn init(base: *const Base, params: *const SpecParams, n_draft: u16, path_static: []const u8, path_dynamic: []const u8, save_dynamic: bool, save_static: bool) NgramCache {
        const func = "common_speculative_impl_ngram_cache";
        spcTrc(func, "%s", .{@as([*:0]const u8, "adding speculative implementation 'ngram-cache'\n")});
        const ps = gpa.dupeSentinel(u8, if (path_static.len == 0) "none" else path_static, 0) catch oom();
        defer gpa.free(ps);
        const pd = gpa.dupeSentinel(u8, if (path_dynamic.len == 0) "none" else path_dynamic, 0) catch oom();
        defer gpa.free(pd);
        spcTrc(func, "- n_draft=%d, cache_static=%s, cache_dynamic=%s\n", .{ @as(c_int, n_draft), ps.ptr, pd.ptr });

        const sinfos = gpa.alloc(SeqInfo, base.n_seq) catch oom();
        for (sinfos) |*si| si.* = .{
            .ngram_cache_context = ngram_cache.Cache.init(gpa) catch oom(),
            .ngram_cache_dynamic = ngram_cache.Cache.init(gpa) catch oom(),
            .ngram_cache_static = ngram_cache.Cache.init(gpa) catch oom(),
        };

        if (path_static.len != 0) {
            const path = gpa.dupeSentinel(u8, path_static, 0) catch oom();
            defer gpa.free(path);
            var loaded = ngram_cache.load(gpa, path) catch |e| switch (e) {
                error.OpenFailed => {
                    spcErr(func, "failed to open static lookup cache: %s", .{path.ptr});
                    _ = c.ggml_abort("speculative.cpp", 2199, "Couldn't read static lookup cache");
                    unreachable;
                },
                error.OutOfMemory => oom(),
            };
            defer ngram_cache.deinitCache(&loaded);
            for (sinfos) |*si| {
                ngram_cache.deinitCache(&si.ngram_cache_static);
                si.ngram_cache_static = ngram_cache.cloneCache(gpa, &loaded) catch oom();
            }
        }

        if (path_dynamic.len != 0) {
            const path = gpa.dupeSentinel(u8, path_dynamic, 0) catch oom();
            defer gpa.free(path);
            var loaded = ngram_cache.load(gpa, path) catch |e| switch (e) {
                error.OpenFailed => {
                    spcErr(func, "failed to open dynamic lookup cache: %s", .{path.ptr});
                    _ = c.ggml_abort("speculative.cpp", 2212, "Couldn't read dynamic lookup cache");
                    unreachable;
                },
                error.OutOfMemory => oom(),
            };
            defer ngram_cache.deinitCache(&loaded);
            for (sinfos) |*si| {
                ngram_cache.deinitCache(&si.ngram_cache_dynamic);
                si.ngram_cache_dynamic = ngram_cache.cloneCache(gpa, &loaded) catch oom();
            }
        }

        return .{ .params = params.ngram_cache, .n_draft = n_draft, .save_dynamic = save_dynamic, .save_static = save_static, .sinfos = sinfos };
    }

    fn deinit(self: *NgramCache) void {
        for (self.sinfos) |*si| {
            ngram_cache.deinitCache(&si.ngram_cache_context);
            ngram_cache.deinitCache(&si.ngram_cache_dynamic);
            ngram_cache.deinitCache(&si.ngram_cache_static);
        }
        gpa.free(self.sinfos);
    }

    /// Ports `begin` (speculative.cpp:2217 @620d2225c): nothing to do.
    fn begin(_: *NgramCache, _: *Base, _: c.llama_seq_id, _: []const c.llama_token) void {}

    /// Ports `draft_one` (speculative.cpp:2221 @620d2225c), a member of
    /// `common_speculative_impl_ngram_cache`.
    fn draftOne(self: *NgramCache, seq_id: usize, dparams: *DraftParams) void {
        const sinfo = &self.sinfos[seq_id];
        const result = dparams.result.?;

        const prompt = dparams.prompt;

        if (sinfo.cache_size < prompt.len + 1) {
            var tokens_new: std.ArrayList(c.llama_token) = .empty;
            defer tokens_new.deinit(gpa);
            tokens_new.ensureTotalCapacity(gpa, prompt.len + 1 -% sinfo.cache_size) catch oom();
            var j = sinfo.cache_size;
            while (j < prompt.len) : (j += 1) tokens_new.append(gpa, prompt[j]) catch oom();
            tokens_new.append(gpa, dparams.id_last) catch oom(); // add the last token

            // Update context ngram cache with new dparams.prompt:
            ngram_cache.update(&sinfo.ngram_cache_context, ngram_cache.ngram_min, ngram_cache.ngram_max, tokens_new.items, @intCast(tokens_new.items.len), false) catch oom();
            sinfo.cache_size = prompt.len + 1;
        }

        var inp: std.ArrayList(c.llama_token) = .empty;
        defer inp.deinit(gpa);
        inp.ensureTotalCapacity(gpa, prompt.len + 1) catch oom();
        inp.appendSliceAssumeCapacity(prompt);
        inp.appendAssumeCapacity(dparams.id_last);

        result.append(gpa, dparams.id_last) catch oom();

        ngram_cache.draft(inp.items, result, gpa, self.n_draft, ngram_cache.ngram_min, ngram_cache.ngram_max, &sinfo.ngram_cache_context, &sinfo.ngram_cache_dynamic, &sinfo.ngram_cache_static) catch oom();

        if (result.items.len > 0) {
            // delete first token in result (which is the id_last token)
            _ = result.orderedRemove(0);
        }
    }

    /// Ports `process` (speculative.cpp:2266 @620d2225c): nothing to do.
    fn process(_: *NgramCache, _: *Base, _: *const common.Batch) bool {
        return true;
    }

    /// Ports `draft` (speculative.cpp:2271 @620d2225c).
    fn draft(self: *NgramCache, base: *Base, dparams: []DraftParams) void {
        std.debug.assert(dparams.len == base.n_seq);
        for (dparams, 0..) |*dp, s| {
            if (!dp.drafting) continue;
            self.draftOne(s, dp);
        }
    }

    /// Ports `accept` (speculative.cpp:2284 @620d2225c): nothing to do.
    fn accept(_: *NgramCache, _: *Base, _: c.llama_seq_id, _: u16, _: bool) void {}
};

// -----------------------------------------------------------------------------
// the driver

/// One drafter: the C++'s `common_speculative_impl` subclasses as a tagged
/// union, its virtual calls as `switch`es.
pub const Impl = struct {
    base: Base,
    kind: Kind,

    pub const Kind = union(enum) {
        draft_simple: draft_impls.DraftSimple,
        eagle3: draft_impls.Eagle3,
        dflash: draft_impls.DFlash,
        mtp: draft_impls.Mtp,
        ngram_simple: NgramSimple,
        ngram_map_k: NgramMapK,
        ngram_mod: NgramMod,
        ngram_cache: NgramCache,
    };

    fn destroy(self: *Impl) void {
        switch (self.kind) {
            inline else => |*k| k.deinit(),
        }
        self.base.n_acc_tokens_per_pos.deinit(gpa);
        gpa.destroy(self);
    }

    fn begin(self: *Impl, seq_id: c.llama_seq_id, prompt: []const c.llama_token) void {
        switch (self.kind) {
            inline else => |*k| k.begin(&self.base, seq_id, prompt),
        }
    }

    fn process(self: *Impl, batch: *const common.Batch) bool {
        return switch (self.kind) {
            inline else => |*k| k.process(&self.base, batch),
        };
    }

    fn draft(self: *Impl, dparams: []DraftParams) void {
        switch (self.kind) {
            inline else => |*k| k.draft(&self.base, dparams),
        }
    }

    fn accept(self: *Impl, seq_id: c.llama_seq_id, n_accepted: u16, is_other: bool) void {
        switch (self.kind) {
            inline else => |*k| k.accept(&self.base, seq_id, n_accepted, is_other),
        }
    }

    /// `get_state` (speculative.cpp:214 @620d2225c): only EAGLE3 has one.
    fn getState(self: *Impl, seq_id: c.llama_seq_id, data: *std.ArrayList(u8)) bool {
        return switch (self.kind) {
            .eagle3 => |*k| k.getState(&self.base, seq_id, data),
            else => false,
        };
    }

    /// `set_state` (speculative.cpp:215 @620d2225c).
    fn setState(self: *Impl, seq_id: c.llama_seq_id, data: []const u8) void {
        switch (self.kind) {
            .eagle3 => |*k| k.setState(&self.base, seq_id, data),
            else => {},
        }
    }
};

/// Ports `common_speculative` (speculative.cpp:2289 @620d2225c).
pub const Speculative = struct {
    dparams: []DraftParams,

    // list of implementations to use and their states
    impls: []*Impl,

    // which implementaion was used for a given seq_id
    impl_last: []?*Impl,

    synth_probs: std.ArrayList(f64) = .empty,
};

/// Ports `get_common_ngram_map` (speculative.cpp:2301 @620d2225c).
fn getCommonNgramMap(t: Type, config: params_mod.SpeculativeNgramMap) ngram_map.Map {
    const size_key = config.size_n;
    const size_value = config.size_m;
    const key_only = t == .ngram_map_k;
    const min_hits = config.min_hits;

    return ngram_map.Map.init(gpa, size_key, size_value, key_only, min_hits) catch oom();
}

/// Ports `create_state_ngram_cache` (speculative.cpp:2312 @620d2225c).
fn createStateNgramCache(base: *const Base, params: *const SpecParams, path_static: []const u8, path_dynamic: []const u8) NgramCache {
    const n_draft: u16 = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    const save_static = false;
    const save_dynamic = false;

    return NgramCache.init(base, params, n_draft, path_static, path_dynamic, save_static, save_dynamic);
}

/// Ports `common_speculative_type_name_str` (speculative.cpp:2328
/// @620d2225c): the types, comma-separated.
///
/// Return: the text, the allocator's.
pub fn typeNameStr(a: std.mem.Allocator, types: []const Type) ![:0]u8 {
    var result: std.ArrayList(u8) = .empty;
    for (types, 0..) |t, i| {
        if (i > 0) try result.append(a, ',');
        try result.appendSlice(a, typeToStr(t));
    }
    return result.toOwnedSliceSentinel(a, 0);
}

/// Ports `common_speculative_all_types_str` (speculative.cpp:2340
/// @620d2225c): every type, `none` included, comma-separated.
pub fn allTypesStr() [:0]const u8 {
    return comptime blk: {
        var s: [:0]const u8 = "";
        const info = @typeInfo(Type).@"enum";
        for (info.field_names, info.field_values, 0..) |name, value, i| {
            if (std.mem.eql(u8, name, "count")) continue;
            s = s ++ (if (i > 0) "," else "") ++ typeToStr(@fromBackingInt(@intCast(value)));
        }
        break :blk s;
    };
}

/// Ports `common_speculative_type_to_str` (speculative.cpp:2352 @620d2225c).
pub fn typeToStr(t: Type) [:0]const u8 {
    return switch (t) {
        .none => "none",
        .draft_simple => "draft-simple",
        .draft_eagle3 => "draft-eagle3",
        .draft_mtp => "draft-mtp",
        .draft_dflash => "draft-dflash",
        .draft_dspark => "draft-dspark",
        .ngram_simple => "ngram-simple",
        .ngram_map_k => "ngram-map-k",
        .ngram_map_k4v => "ngram-map-k4v",
        .ngram_mod => "ngram-mod",
        .ngram_cache => "ngram-cache",
        .count => "unknown",
    };
}

/// Ports `common_speculative_types_from_names` (speculative.cpp:2369
/// @620d2225c): `none` anywhere means `{ none }`; an unknown name throws
/// `std::invalid_argument`.
///
/// Return: the types, the allocator's.
pub fn typesFromNames(a: std.mem.Allocator, names: []const []const u8) ![]Type {
    var types: std.ArrayList(Type) = .empty;
    errdefer types.deinit(a);
    try types.ensureTotalCapacity(a, names.len);

    for (names) |nm| {
        if (findType(nm)) |t| {
            if (t == .none) {
                types.deinit(a);
                return a.dupe(Type, &.{.none});
            }
            types.appendAssumeCapacity(t);
            continue;
        }
        arg.setWhat("unknown speculative type: %.*s", .{ @as(c_int, @intCast(nm.len)), nm.ptr });
        return error.InvalidArgument;
    }

    return types.toOwnedSlice(a);
}

/// Ports `common_speculative_type_from_name` (speculative.cpp:2388
/// @620d2225c).
///
/// Return: the type, or `.count` for an unknown name.
pub fn typeFromName(name: []const u8) Type {
    return findType(name) orelse .count;
}

/// Ports `common_speculative_types_from_gguf` (speculative.cpp:2396
/// @620d2225c): the speculative type a draft model's metadata implies --
/// MTP for a model with next-token prediction layers, DFlash or DSpark for
/// a `dflash` model -- or none.
///
/// Return: the types, the allocator's; empty when unknown.
pub fn typesFromGguf(a: std.mem.Allocator, path: [:0]const u8) ![]Type {
    const gguf_ctx = c.gguf_init_from_file(path.ptr, .{ .no_alloc = true, .ctx = null }) orelse return a.alloc(Type, 0);
    defer c.gguf_free(gguf_ctx);

    const arch_id = c.gguf_find_key(gguf_ctx, "general.architecture");
    if (arch_id < 0 or c.gguf_get_kv_type(gguf_ctx, arch_id) != c.GGUF_TYPE_STRING) return a.alloc(Type, 0);

    const arch = std.mem.span(c.gguf_get_val_str(gguf_ctx, arch_id));
    if (!std.mem.eql(u8, arch, "dflash")) {
        const key = try std.fmt.allocPrintSentinel(a, "{s}.block_count", .{arch}, 0);
        defer a.free(key);
        const block_count = c.gguf_get_val_u32(gguf_ctx, c.gguf_find_key(gguf_ctx, key.ptr));

        const name = try std.fmt.allocPrintSentinel(a, "blk.{d}.nextn.eh_proj.weight", .{block_count -% 1}, 0);
        defer a.free(name);
        if (c.gguf_find_tensor(gguf_ctx, name.ptr) >= 0) return a.dupe(Type, &.{.draft_mtp});

        return a.alloc(Type, 0);
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    const t: Type = if (c.gguf_find_tensor(gguf_ctx, "markov_w1.weight") >= 0) .draft_dspark else .draft_dflash;

    spcInf("common_speculative_types_from_gguf", "auto-detected speculative type '%s' from the draft model metadata\n", .{typeToStr(t).ptr});

    return a.dupe(Type, &.{t});
}

/// Ports `common_get_enabled_speculative_configs` (speculative.cpp:2433
/// @620d2225c): the types as a bit set.
fn getEnabledConfigs(configs: []const Type) u32 {
    var result: u32 = 0;
    for (configs) |t| result |= @as(u32, 1) << @intCast(@backingInt(t));
    return result;
}

/// Ports `common_speculative_n_max` (speculative.cpp:2441 @620d2225c), the
/// overload over parameters: the longest draft any configured type asks for.
pub fn nMaxParams(spec: *const SpecParams) i32 {
    var n_max: i32 = 0;

    for (spec.types.items) |t| {
        switch (t) {
            .draft_simple, .draft_eagle3, .draft_mtp, .draft_dflash, .draft_dspark => n_max = @max(n_max, @max(0, spec.draft.n_max)),
            .ngram_simple => n_max = @max(n_max, @as(i32, spec.ngram_simple.size_m)),
            .ngram_map_k => n_max = @max(n_max, @as(i32, spec.ngram_map_k.size_m)),
            .ngram_map_k4v => n_max = @max(n_max, @as(i32, spec.ngram_map_k4v.size_m)),
            .ngram_mod => n_max = @max(n_max, @max(0, spec.ngram_mod.n_max)),
            .ngram_cache => n_max = @max(n_max, 8),
            .none, .count => {},
        }
    }

    return n_max;
}

/// Ports `common_speculative_n_max` (speculative.cpp:2477 @620d2225c), the
/// overload over a context: the longest draft any drafter will produce.
pub fn nMax(spec: ?*const Speculative) i32 {
    var n_max: i32 = 0;

    const s = spec orelse return n_max;

    for (s.impls) |impl| n_max = @max(n_max, @max(0, impl.base.n_max));

    return n_max;
}

/// Ports `common_speculative_synth_rates_resolve` (speculative.cpp:2491
/// @620d2225c): the unconditional synthetic acceptance rates, given as a
/// list or solved from a mean length by bisection.
///
/// Return: the rates, the allocator's; empty when synthetic acceptance is
/// off.
pub fn synthRatesResolve(a: std.mem.Allocator, spec: *const SpecParams, n_max: i32) InitError![]f64 {
    const has_length = spec.synth_len != -1.0;
    const has_rates = spec.synth_rates.items.len != 0;

    if (!has_length and !has_rates) return a.alloc(f64, 0);
    if (has_length and has_rates) return throwInvalid("synthetic acceptance length and rates are mutually exclusive", .{});

    if (n_max <= 0) return throwInvalid("synthetic acceptance requires at least one speculative token", .{});

    if (has_rates) {
        const rates = spec.synth_rates.items;
        if (rates.len != @as(usize, @intCast(n_max))) {
            return throwInvalid("synthetic acceptance rates must contain %d values, got %zu", .{ n_max, rates.len });
        }

        for (rates, 0..) |r, i| {
            if (!std.math.isFinite(r) or r < 0.0 or r > 1.0) {
                return throwInvalid("synthetic acceptance rates must be finite and within [0, 1]", .{});
            }
            if (i > 0 and r > rates[i - 1]) {
                return throwInvalid("synthetic acceptance rates must be monotonically non-increasing", .{});
            }
        }

        return a.dupe(f64, rates);
    }

    const length = spec.synth_len;
    const length_max = @as(f64, @floatFromInt(n_max)) + 1.0;
    if (!std.math.isFinite(length) or length < 1.0 or length > length_max) {
        return throwInvalid("synthetic acceptance length must be finite and within [1, %.0f]", .{length_max});
    }

    var p: f64 = 0.0;
    if (length == length_max) {
        p = 1.0;
    } else if (length > 1.0) {
        var p_min: f64 = 0.0;
        var p_max: f64 = 1.0;
        for (0..32) |_| {
            const p_mid = 0.5 * (p_min + p_max);
            var sum: f64 = 0.0;
            var term = p_mid;
            var j: i32 = 0;
            while (j < n_max) : (j += 1) {
                sum += term;
                term *= p_mid;
            }

            if (sum < length - 1.0) {
                p_min = p_mid;
            } else {
                p_max = p_mid;
            }
        }
        p = 0.5 * (p_min + p_max);
    }

    const rates = try a.alloc(f64, @intCast(n_max));
    var rate = p;
    for (rates) |*r| {
        r.* = rate;
        rate *= p;
    }

    return rates;
}

/// Ports `common_speculative_get_synth_probs` (speculative.cpp:2567
/// @620d2225c): the conditional synthetic acceptance probabilities.
///
/// Return: borrowed from `spec`.
pub fn getSynthProbs(spec: ?*const Speculative) []const f64 {
    assertC(spec != null, 2568, "spec");
    return spec.?.synth_probs.items;
}

/// Ports `common_base_params_to_speculative` (speculative.cpp:2572
/// @620d2225c): the parameters a draft model and its context load with.
///
/// Return: a copy of `params`; its lists are shared with `params`.
pub fn baseParamsToSpeculative(params: *const Params) Params {
    const has_draft = params.speculative.hasDft();

    const params_spec = &params.speculative.draft;
    var result = params.*;

    result.embedding = false;
    result.pooling_type = c.LLAMA_POOLING_TYPE_UNSPECIFIED;

    if (has_draft) {
        // default to global devices value
        if (params_spec.devices.items.len != 0) result.devices = params_spec.devices;
        result.model = params_spec.mparams;
        result.n_gpu_layers = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        // a draft pinned to a single device doesn't need the meta wrapper an inherited -sm tensor would give it
        // (the device list is null-terminated, so a single device means size 2)
        var n_devs: usize = 0;
        for (params_spec.devices.items) |d| {
            if (d != null) n_devs += 1;
        }
        if (n_devs == 1) result.split_mode = c.LLAMA_SPLIT_MODE_LAYER;

        if (params_spec.cpuparams.n_threads > 0) {
            result.cpuparams.n_threads = params_spec.cpuparams.n_threads;
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }
    }

    result.cache_type_k = params_spec.cache_type_k;
    result.cache_type_v = params_spec.cache_type_v;
    result.n_outputs_max = params.n_parallel;
    result.n_outputs_max_per_seq = 1;

    if (params_spec.n_ubatch > 0) {
        result.n_ubatch = params_spec.n_ubatch;
    }

    // the MoE cache is only used by the target context
    result.moe_cache_size = 0;

    // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend
    var has_block_draft = false;
    for (params.speculative.types.items) |t| {
        if (t == .draft_dflash or t == .draft_dspark) has_block_draft = true;
    }
    if (has_block_draft) {
        // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
        const per_seq: i32 = @max(1, params_spec.n_max + 1);
        result.n_outputs_max = params.n_parallel * per_seq;
        if (params_spec.backend_sampling) result.n_outputs_max_per_seq = per_seq;
    }

    return result;
}

/// Ports `common_speculative_init_result` (speculative.h:111 @620d2225c):
/// the draft model and context, owned. The context frees first, as the
/// C++'s member order decides.
pub const InitResult = struct {
    model_: ?*c.llama_model = null,
    context_: ?*c.llama_context = null,

    /// Ports `common_speculative_init_result` (speculative.cpp:2645
    /// @620d2225c), the constructor: a draft model and its context, or an
    /// MTP context over the target model.
    pub fn init(params: *Params, model_tgt: ?*c.llama_model, ctx_tgt: ?*c.llama_context) !InitResult {
        const func: [*:0]const u8 = "common_speculative_init_result";
        var self: InitResult = .{};
        const has_draft = params.speculative.hasDft();
        var spec_mtp = false;
        for (params.speculative.types.items) |t| {
            if (t == .draft_mtp) spec_mtp = true;
        }

        const mparams = common.modelParamsToLlama(params);
        var cparams = common.contextParamsToLlama(params);

        if (spec_mtp) cparams.ctx_type = c.LLAMA_CONTEXT_TYPE_MTP;

        // the draft context holds as many tokens per sequence as the target context
        cparams.n_ctx = c.llama_n_ctx(ctx_tgt);

        // note: for small models maybe we can set this to the maximum possible draft from all speculative types
        //       the extra memory for small models is likely negligible?
        cparams.n_rs_seq = 0;
        cparams.ctx_other = ctx_tgt;

        if (has_draft) {
            const model_path = try gpa.dupeSentinel(u8, params.speculative.draft.mparams.path, 0);
            defer gpa.free(model_path);
            log.inf("%s: loading draft model '%s'\n", .{ func, model_path.ptr });

            const model_dft = c.llama_model_load_from_file(model_path.ptr, mparams) orelse {
                log.err("%s: failed to load draft model, '%s'\n", .{ func, model_path.ptr });
                return self;
            };

            self.model_ = model_dft;

            const ctx_dft = c.llama_init_from_model(model_dft, cparams) orelse {
                log.err("%s: failed to create MTP context\n", .{func});
                return self;
            };

            self.context_ = ctx_dft;
        } else if (spec_mtp) {
            const model_path = try gpa.dupeSentinel(u8, params.model.path, 0);
            defer gpa.free(model_path);

            log.inf("%s: creating MTP draft context against the target model '%s'\n", .{ func, model_path.ptr });

            const ctx_dft = c.llama_init_from_model(model_tgt, cparams) orelse {
                log.err("%s: failed to create MTP context\n", .{func});
                return self;
            };

            self.context_ = ctx_dft;
        }
        return self;
    }

    /// Ports the destructor of `common_speculative_init_result`
    /// (speculative.cpp:2705 @620d2225c): context, then model.
    pub fn deinit(self: *InitResult) void {
        if (self.context_) |ctx| c.llama_free(ctx);
        if (self.model_) |m| c.llama_model_free(m);
        self.* = .{};
    }

    /// Ports `model` (speculative.cpp:2707 @620d2225c), a member of
    /// `common_speculative_init_result`.
    pub fn model(self: *const InitResult) ?*c.llama_model {
        return self.model_;
    }

    /// Ports `context` (speculative.cpp:2711 @620d2225c), a member of
    /// `common_speculative_init_result`.
    pub fn context(self: *const InitResult) ?*c.llama_context {
        return self.context_;
    }
};

/// Ports `common_speculative_init_from_params` (speculative.cpp:2715
/// @620d2225c).
pub fn initFromParams(params: *Params, model_tgt: ?*c.llama_model, ctx_tgt: ?*c.llama_context) !InitResult {
    return InitResult.init(params, model_tgt, ctx_tgt);
}

/// Ports `common_speculative_output_limits` (speculative.h:40 @620d2225c).
pub const OutputLimits = struct {
    total: i32,
    per_seq: i32,
};

/// Ports `common_speculative_get_output_limits` (speculative.cpp:2719
/// @620d2225c): how many outputs a batch and a sequence need when every
/// sequence verifies a draft of `n_draft`.
pub fn getOutputLimits(n_batch: i32, n_parallel: i32, n_draft: i32) OutputLimits {
    const per_seq: i64 = 1 + @as(i64, @max(0, n_draft));
    const total: i64 = @as(i64, n_parallel) * per_seq;

    return .{
        .total = @truncate(@min(@as(i64, n_batch), total)),
        .per_seq = @truncate(@min(@as(i64, n_batch), per_seq)),
    };
}

/// Ports `common_speculative_init` (speculative.cpp:2732 @620d2225c): every
/// configured drafter, in order of preference -- the n-gram ones first,
/// then those that run a model -- and the synthetic acceptance rates.
///
/// Return: the context, freed with `free`; null when no drafter applies.
pub fn init(params: *SpecParams, n_seq: u32) InitError!?*Speculative {
    const func = "common_speculative_init";
    // Compute the implementations to use based on the config and their order of preference
    var configs: std.ArrayList(Type) = .empty; // list of speculative configs to try
    defer configs.deinit(gpa);
    {
        const enabled_configs = getEnabledConfigs(params.types.items);

        const has_ctx_dft = params.draft.ctx_dft != null;
        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        const order = [_]struct { Type, bool }{
            .{ .ngram_simple, true },
            .{ .ngram_map_k, true },
            .{ .ngram_map_k4v, true },
            .{ .ngram_mod, true },
            .{ .ngram_cache, true },
            .{ .draft_simple, true },
            .{ .draft_eagle3, has_ctx_dft },
            .{ .draft_mtp, has_ctx_dft },
            .{ .draft_dflash, has_ctx_dft },
            .{ .draft_dspark, has_ctx_dft },
        };
        for (order) |o| {
            if (o[1] and (enabled_configs & (@as(u32, 1) << @intCast(@backingInt(o[0])))) != 0) try configs.append(gpa, o[0]);
        }
    }

    var impls: std.ArrayList(*Impl) = .empty;
    errdefer {
        for (impls.items) |impl| impl.destroy();
        impls.deinit(gpa);
    }

    for (configs.items) |t| {
        const impl = try gpa.create(Impl);
        impl.base = .{ .type = t, .n_seq = n_seq, .n_max = 0 };
        var ok = true;
        defer if (!ok) gpa.destroy(impl);
        errdefer ok = false;
        switch (t) {
            .none, .count => {
                gpa.destroy(impl);
                continue;
            },
            .draft_simple => {
                impl.base.n_max = params.draft.n_max;
                impl.kind = .{ .draft_simple = try draft_impls.DraftSimple.init(&impl.base, params) };
            },
            .draft_eagle3 => {
                impl.base.n_max = params.draft.n_max;
                impl.kind = .{ .eagle3 = try draft_impls.Eagle3.init(&impl.base, params) };
            },
            .draft_mtp => {
                impl.base.n_max = params.draft.n_max;
                impl.kind = .{ .mtp = try draft_impls.Mtp.init(&impl.base, params) };
            },
            .draft_dflash, .draft_dspark => {
                impl.base.n_max = params.draft.n_max;
                impl.kind = .{ .dflash = try draft_impls.DFlash.init(&impl.base, params, t) };
            },
            .ngram_simple => {
                var map = getCommonNgramMap(t, params.ngram_simple);
                defer map.deinit();

                const ngram_size_key = map.size_key;
                const mgram_size_value = map.size_value;

                const config_simple: ngram_map.SimpleConfig = .{
                    .size_ngram = ngram_size_key,
                    .size_mgram = mgram_size_value,
                };
                impl.base.n_max = params.ngram_simple.size_m;
                impl.kind = .{ .ngram_simple = NgramSimple.init(params, config_simple) };
            },
            .ngram_map_k, .ngram_map_k4v => {
                var map = getCommonNgramMap(t, if (t == .ngram_map_k) params.ngram_map_k else params.ngram_map_k4v);
                defer map.deinit();
                impl.base.type = if (map.key_only) .ngram_map_k else .ngram_map_k4v;
                impl.base.n_max = map.size_value;
                impl.kind = .{ .ngram_map_k = NgramMapK.init(&impl.base, &map) };
            },
            .ngram_mod => {
                impl.base.n_max = params.ngram_mod.n_max;
                impl.kind = .{ .ngram_mod = NgramMod.init(&impl.base, params) };
            },
            .ngram_cache => {
                impl.base.n_max = 8;
                impl.kind = .{ .ngram_cache = createStateNgramCache(&impl.base, params, params.ngram_cache.lookup_cache_static, params.ngram_cache.lookup_cache_dynamic) };
            },
        }
        try impls.append(gpa, impl);
    }

    if (impls.items.len == 0) {
        spcTrc(func, "%s", .{@as([*:0]const u8, "no implementations specified for speculative decoding\n")});
        impls.deinit(gpa);
        return null;
    }

    const result = try gpa.create(Speculative);
    result.* = .{
        .dparams = try gpa.alloc(DraftParams, n_seq),
        .impls = try impls.toOwnedSlice(gpa),
        .impl_last = try gpa.alloc(?*Impl, n_seq),
    };
    @memset(result.dparams, .{});
    @memset(result.impl_last, null);
    errdefer free(result);

    const n_max_configured = nMaxParams(params);
    const n_max_effective = nMax(result);
    const rates = try synthRatesResolve(gpa, params, n_max_effective);
    defer gpa.free(rates);

    var rates_str: std.ArrayList(u8) = .empty;
    defer rates_str.deinit(gpa);
    try result.synth_probs.ensureTotalCapacity(gpa, rates.len);
    var rate_prev: f64 = 1.0;
    var acceptance_length: f64 = 1.0;
    for (rates, 0..) |rate, i| {
        result.synth_probs.appendAssumeCapacity(if (rate_prev > 0.0) rate / rate_prev else 0.0);
        var buf: [64]u8 = undefined;
        const n = snprintf(&buf, buf.len, "%.6g", rate);
        if (i > 0) try rates_str.appendSlice(gpa, ", ");
        try rates_str.appendSlice(gpa, buf[0..@intCast(n)]);
        rate_prev = rate;
        acceptance_length += rate;
    }
    if (result.synth_probs.items.len != 0) {
        spcWrn(func, "%s", .{@as([*:0]const u8, "synthetic speculative acceptance is enabled for benchmarking; generated output is not valid\n")});
        if (n_max_effective != n_max_configured) {
            spcWrn(func, "synthetic acceptance draft limit was reduced from %d to %d by the initialized speculative implementations\n", .{ n_max_configured, n_max_effective });
        }
        try rates_str.append(gpa, 0);
        spcInf(func, "synthetic acceptance: n_max = %zu, mean length = %.6f, rates = [%s]\n", .{ rates.len, acceptance_length, @as([*:0]const u8, @ptrCast(rates_str.items.ptr)) });
    }

    return result;
}

extern fn snprintf(buf: [*]u8, size: usize, format: [*:0]const u8, ...) c_int;

/// Ports `common_speculative_free` (speculative.cpp:2877 @620d2225c).
pub fn free(spec: ?*Speculative) void {
    const s = spec orelse return;

    for (s.impls) |impl| impl.destroy();
    gpa.free(s.impls);
    gpa.free(s.dparams);
    gpa.free(s.impl_last);
    s.synth_probs.deinit(gpa);
    gpa.destroy(s);
}

/// Ports `common_speculative_get_draft_params` (speculative.cpp:2885
/// @620d2225c): the slot a caller fills before `draft`.
pub fn getDraftParams(spec: ?*Speculative, seq_id: c.llama_seq_id) *DraftParams {
    assertC(spec != null, 2888, "spec");
    assertC(seq_id < @as(c.llama_seq_id, @intCast(spec.?.dparams.len)), 2837, "seq_id < (llama_seq_id) spec->dparams.size()");

    return &spec.?.dparams[@intCast(seq_id)];
}

/// Ports `common_speculative_begin` (speculative.cpp:2894 @620d2225c):
/// optionally called once at the start of a generation.
pub fn begin(spec: ?*Speculative, seq_id: c.llama_seq_id, prompt: []const c.llama_token) void {
    const s = spec orelse return;

    for (s.impls) |impl| {
        const t0 = timeStart(!impl.base.gen_perf);
        impl.begin(seq_id, prompt);
        impl.base.n_call_begin += 1;
        timeEnd(t0, &impl.base.t_begin_us);
    }
}

/// Ports `common_speculative_process` (speculative.cpp:2906 @620d2225c):
/// the target's batch through every drafter that keeps state.
///
/// Return: false once one fails; the rest are skipped, as `&&` skips them.
pub fn process(spec: ?*Speculative, batch: *const common.Batch) bool {
    var result = true;

    const s = spec orelse return result;

    for (s.impls) |impl| result = result and impl.process(batch);

    return result;
}

/// Ports `common_speculative_draft` (speculative.cpp:2920 @620d2225c): each
/// drafting sequence asks the drafters in order until one produces a draft.
pub fn draft(spec: ?*Speculative) void {
    const func = "common_speculative_draft";
    const s = spec orelse return;

    const dparams = s.dparams;

    {
        var n_drafting: c_int = 0;

        for (dparams) |*dp| {
            assertC(!dp.drafting or dp.result.?.items.len == 0, 2931, "!dp.drafting || dp.result->empty()");

            if (dp.drafting) n_drafting += 1;
        }

        if (n_drafting == 0) return;
    }

    for (s.impls) |impl| {
        {
            const t0 = timeStart(!impl.base.gen_perf);
            impl.draft(dparams);
            impl.base.n_call_draft += 1;
            timeEnd(t0, &impl.base.t_draft_us);
        }

        var n_drafting: c_int = 0;

        for (dparams, 0..) |*dp, seq_id| {
            if (!dp.drafting) continue;

            const result = dp.result.?;

            // a new draft has been sampled
            if (dp.drafting and result.items.len != 0) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (result.items.len != 0 and @as(i64, @intCast(result.items.len)) > dp.n_max) {
                        spcDbg(func, "truncating draft to %d tokens\n", .{dp.n_max});
                        result.shrinkRetainingCapacity(@intCast(dp.n_max));

                        // trim the candidates only if the drafter produced them (n-gram drafters do not)
                        if (dp.result_q) |q| {
                            if (q.items.len != 0) resizeCandidates(q, @intCast(dp.n_max));
                        }
                    }
                }

                if (result.items.len != 0) {
                    spcDbg(func, "called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", .{ typeToStr(impl.base.type).ptr, dp.prompt.len, impl.base.n_call_draft, result.items.len });

                    // remember which implementation was used
                    s.impl_last[seq_id] = impl;

                    impl.base.n_gen_drafts += 1;
                    impl.base.n_gen_tokens += result.items.len;
                }
            }

            if (dp.drafting) n_drafting += 1;
        }

        if (n_drafting == 0) break;
    }

    // these sequences failed to generate a draft
    for (dparams) |*dp| {
        if (dp.drafting) dp.drafting = false;
    }
}

/// Ports `common_speculative_accept` (speculative.cpp:3010 @620d2225c):
/// `n_accepted` tokens of the last draft were accepted by the target.
pub fn accept(spec: *Speculative, seq_id: c.llama_seq_id, n_accepted: u16) void {
    const impl = spec.impl_last[@intCast(seq_id)] orelse {
        assertC(n_accepted == 0, 3014, "n_accepted == 0");
        return;
    };

    {
        const t0 = timeStart(!impl.base.gen_perf);

        if (impl.base.n_acc_tokens_per_pos.items.len < n_accepted) {
            const old = impl.base.n_acc_tokens_per_pos.items.len;
            impl.base.n_acc_tokens_per_pos.resize(gpa, n_accepted) catch oom();
            @memset(impl.base.n_acc_tokens_per_pos.items[old..], 0);
        }

        for (0..n_accepted) |i| impl.base.n_acc_tokens_per_pos.items[i] += 1;

        if (n_accepted > 0) {
            impl.base.n_acc_drafts += 1;
            impl.base.n_acc_tokens += n_accepted;
        }

        impl.accept(seq_id, n_accepted, false);
        impl.base.n_call_accept += 1;
        timeEnd(t0, &impl.base.t_accept_us);
    }

    // accept with the rest of the implementations, using is_other == true
    for (spec.impls) |impl_other| {
        if (impl_other != impl) impl_other.accept(seq_id, n_accepted, true);
    }
}

/// Ports `common_speculative_get_state` (speculative.cpp:3047 @620d2225c):
/// the first drafter's per-sequence state, if any has one.
///
/// Return: whether `data` was filled; grown with `c_allocator`.
pub fn getState(spec: ?*Speculative, seq_id: c.llama_seq_id, data: *std.ArrayList(u8)) bool {
    const s = spec orelse return false;

    for (s.impls) |impl| {
        if (impl.getState(seq_id, data)) return true;
    }

    return false;
}

/// Ports `common_speculative_set_state` (speculative.cpp:3061 @620d2225c).
pub fn setState(spec: ?*Speculative, seq_id: c.llama_seq_id, data: []const u8) void {
    const s = spec orelse return;

    for (s.impls) |impl| impl.setState(seq_id, data);
}

/// Ports `common_speculative_print_stats` (speculative.cpp:3071
/// @620d2225c): one line per drafter -- calls, drafts, tokens, the mean
/// accepted length, the acceptance rate per position, the time spent.
pub fn printStats(spec: ?*const Speculative) void {
    const s = spec orelse return;

    for (s.impls) |impl| {
        const b = &impl.base;
        var str_perf: std.ArrayList(u8) = .empty;
        defer str_perf.deinit(gpa);
        if (b.gen_perf) {
            appendFmt(&str_perf, ", dur(b,g,a) = %.3f, ", .{@as(f64, @floatFromInt(b.t_begin_us)) / 1000.0});
            appendFmt(&str_perf, "%.3f, ", .{@as(f64, @floatFromInt(b.t_draft_us)) / 1000.0});
            appendFmt(&str_perf, "%.3f ms", .{@as(f64, @floatFromInt(b.t_accept_us)) / 1000.0});
        }

        var str_stats: std.ArrayList(u8) = .empty;
        defer str_stats.deinit(gpa);
        if (b.n_call_accept > 0) {
            const mean = 1.0 + @as(f64, @floatFromInt(b.n_acc_tokens)) / @as(f64, @floatFromInt(b.n_call_accept));
            appendFmt(&str_stats, ", #mean acc len = %.2f, #acc rate/pos = (", .{mean});
            for (b.n_acc_tokens_per_pos.items, 0..) |n, i| {
                if (i > 0) str_stats.appendSlice(gpa, ", ") catch oom();
                appendFmt(&str_stats, "%.3f", .{@as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(b.n_call_accept))});
            }
            str_stats.append(gpa, ')') catch oom();
        }
        str_stats.append(gpa, 0) catch oom();
        str_perf.append(gpa, 0) catch oom();

        spcTrc("common_speculative_print_stats", "statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n", .{
            typeToStr(b.type).ptr,
            b.n_call_begin,
            b.n_call_draft,
            b.n_call_accept,
            b.n_gen_drafts,
            b.n_acc_drafts,
            b.n_gen_tokens,
            b.n_acc_tokens,
            @as([*:0]const u8, @ptrCast(str_stats.items.ptr)),
            @as([*:0]const u8, @ptrCast(str_perf.items.ptr)),
        });
    }
}

/// `oss << std::fixed << std::setprecision(n) << x`, which formats as
/// `printf`'s `%.nf` does.
fn appendFmt(out: *std.ArrayList(u8), comptime fmt: [*:0]const u8, args: anytype) void {
    var buf: [128]u8 = undefined;
    const n = @call(.auto, snprintf, .{ &buf, buf.len, fmt } ++ args);
    out.appendSlice(gpa, buf[0..@min(@as(usize, @intCast(n)), buf.len - 1)]) catch oom();
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the type names round-trip and none ends a list" {
    try std.testing.expectEqual(Type.ngram_map_k4v, typeFromName("ngram-map-k4v"));
    try std.testing.expectEqual(Type.count, typeFromName("bogus"));
    try std.testing.expectEqualStrings("none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache", allTypesStr());
}

test "the output limits clamp to the batch" {
    const l = getOutputLimits(8, 4, 3);
    try std.testing.expectEqual(@as(i32, 8), l.total);
    try std.testing.expectEqual(@as(i32, 4), l.per_seq);
}

test "a synthetic length solves to rates that sum to it" {
    var p: SpecParams = .{};
    p.synth_len = 2.5;
    const rates = try synthRatesResolve(std.testing.allocator, &p, 4);
    defer std.testing.allocator.free(rates);
    var sum: f64 = 1.0;
    for (rates) |r| sum += r;
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), sum, 1e-6);
}
