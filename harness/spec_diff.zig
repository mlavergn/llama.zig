//! The port side of `scripts/spec-diff`: `harness/spec_ref.cpp`'s script,
//! line for line, over `tools/common/`'s sampler, reasoning budget,
//! speculative drafters and n-gram stores. Both write one transcript to
//! stdout through the C library's `printf` -- so `%a`, `%zu` and `%016llx`
//! format alike -- and log through `common_log` to stderr.
//!
//! **Not a port.** See `harness/spec_ref.cpp` for the modes and the
//! environment.

const std = @import("std");
const tc = @import("tools_common");
const c = tc.c;
const sampling = tc.sampling;
const spec_mod = tc.speculative;
const ngram_cache = tc.ngram_cache;
const ngram_map = tc.ngram_map;
const ngram_mod = tc.ngram_mod;
const rbm = tc.reasoning_budget;
const params_mod = tc.params;

comptime {
    // libllama, compiled in (see `build/tools.zig`).
    _ = @import("llama");
}

const gpa = std.heap.c_allocator;

const printf = c.printf;

fn fnv(bytes: []const u8, h0: u64) u64 {
    var h = h0;
    for (bytes) |b| {
        h ^= b;
        h *%= 1099511628211;
    }
    return h;
}
const fnv0: u64 = 1469598103934665603;

fn printTokens(label: [*:0]const u8, v: []const c.llama_token) void {
    _ = printf("%s [", label);
    for (v, 0..) |t, i| _ = printf(if (i != 0) ",%d" else "%d", t);
    _ = printf("]");
}

const StreamGen = struct {
    x: u32,
    fn next(self: *StreamGen) u32 {
        self.x = self.x *% 1103515245 +% 12345;
        return (self.x >> 16) & 0x7fff;
    }
    fn make(self: *StreamGen, n: usize, vocab: u32, rep: u32) []c.llama_token {
        var v: std.ArrayList(c.llama_token) = .empty;
        while (v.items.len < n) {
            if (v.items.len > 8 and self.next() % 100 < rep) {
                const len = 2 + self.next() % 10;
                const from = self.next() % (v.items.len - 1);
                var k: usize = 0;
                while (k < len and v.items.len < n) : (k += 1) v.append(gpa, v.items[(from + k) % v.items.len]) catch unreachable;
            } else {
                v.append(gpa, @intCast(self.next() % vocab)) catch unreachable;
            }
        }
        return v.toOwnedSlice(gpa) catch unreachable;
    }
};

fn dumpCache(label: [*:0]const u8, nc: *const ngram_cache.Cache) void {
    _ = printf("%s size=%zu buckets=%zu\n", label, nc.count(), nc.bucketCount());
    var it = nc.iterator();
    while (it.next()) |e| {
        _ = printf("  [%d,%d,%d,%d] n=%zu b=%zu:", e.key.tokens[0], e.key.tokens[1], e.key.tokens[2], e.key.tokens[3], e.value.count(), e.value.bucketCount());
        var pi = e.value.iterator();
        while (pi.next()) |p| _ = printf(" %d:%d", p.key, p.value.*);
        _ = printf("\n");
    }
}

fn fileHash(path: [:0]const u8, size: *usize) u64 {
    const f = c.fopen(path.ptr, "rb") orelse {
        size.* = 0;
        return 0;
    };
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    var tmp: [4096]u8 = undefined;
    while (true) {
        const n = c.fread(&tmp, 1, tmp.len, f);
        if (n == 0) break;
        buf.appendSlice(gpa, tmp[0..n]) catch unreachable;
    }
    _ = c.fclose(f);
    size.* = buf.items.len;
    return fnv(buf.items, fnv0);
}

fn apiTypes() void {
    _ = printf("all_types %s\n", spec_mod.allTypesStr().ptr);
    const lists = [_][]const []const u8{
        &.{"ngram-mod"},         &.{ "none", "ngram-mod" },          &.{ "ngram-mod", "none" }, &.{ "draft-simple", "ngram-cache", "ngram-map-k" },
        &.{"bogus"},             &.{ "ngram-map-k4v", "Ngram-mod" }, &.{},                      &.{ "draft-eagle3", "draft-mtp", "draft-dflash", "draft-dspark", "ngram-simple" },
        &.{ "ngram-map-k", "" },
    };
    for (lists) |l| {
        _ = printf("from_names");
        for (l) |s| {
            const z = gpa.dupeSentinel(u8, s, 0) catch unreachable;
            _ = printf(" '%s'", z.ptr);
        }
        if (spec_mod.typesFromNames(gpa, l)) |t| {
            const s = spec_mod.typeNameStr(gpa, t) catch unreachable;
            _ = printf(" -> '%s' (%zu)\n", s.ptr, t.len);
        } else |e| switch (e) {
            error.InvalidArgument => _ = printf(" -> invalid_argument: %.*s\n", @as(c_int, @intCast(tc.arg.what().len)), tc.arg.what().ptr),
            else => unreachable,
        }
    }
    for ([_][:0]const u8{ "none", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash", "draft-dspark", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache", "x", "", "NONE" }) |n| {
        _ = printf("from_name '%s' -> %d\n", n.ptr, @as(c_int, @backingInt(spec_mod.typeFromName(n))));
    }
    inline for (@typeInfo(params_mod.SpeculativeType).@"enum".field_values) |v| {
        _ = printf("to_str %d -> %s\n", @as(c_int, v), spec_mod.typeToStr(@fromBackingInt(@intCast(v))).ptr);
    }
}

fn apiNMax() void {
    var k: i32 = 0;
    while (k < 24) : (k += 1) {
        var p: params_mod.Speculative = .{};
        var t: i32 = 0;
        while (t < @backingInt(params_mod.SpeculativeType.count)) : (t += 1) {
            if (@mod(k * 7 + t * 3, 5) == 0) p.types.append(gpa, @fromBackingInt(@intCast(t))) catch unreachable;
        }
        p.draft.n_max = @mod(k, 7) - 2;
        p.ngram_simple.size_m = @intCast(k * 3);
        p.ngram_map_k.size_m = @intCast(@mod(k * 5, 17));
        p.ngram_map_k4v.size_m = @intCast(@mod(k * 11, 23));
        p.ngram_mod.n_max = @mod(k, 9) * 9 - 10;
        const s = spec_mod.typeNameStr(gpa, p.types.items) catch unreachable;
        _ = printf("n_max types='%s' -> %d\n", s.ptr, spec_mod.nMaxParams(&p));
    }
}

fn apiSynth() void {
    const C = struct { len: f64, rates: []const f64, n_max: i32 };
    const nan = std.math.nan(f64);
    const cases = [_]C{
        .{ .len = -1.0, .rates = &.{}, .n_max = 3 },                .{ .len = 2.5, .rates = &.{}, .n_max = 3 },                 .{ .len = 1.0, .rates = &.{}, .n_max = 3 },
        .{ .len = 4.0, .rates = &.{}, .n_max = 3 },                 .{ .len = 4.5, .rates = &.{}, .n_max = 3 },                 .{ .len = nan, .rates = &.{}, .n_max = 3 },
        .{ .len = 0.5, .rates = &.{}, .n_max = 3 },                 .{ .len = -1.0, .rates = &.{ 0.9, 0.8, 0.5 }, .n_max = 3 }, .{ .len = -1.0, .rates = &.{ 0.9, 0.8 }, .n_max = 3 },
        .{ .len = -1.0, .rates = &.{ 0.5, 0.8, 0.9 }, .n_max = 3 }, .{ .len = -1.0, .rates = &.{ 1.5, 0.8, 0.5 }, .n_max = 3 }, .{ .len = 2.0, .rates = &.{ 0.5, 0.4, 0.3 }, .n_max = 3 },
        .{ .len = 2.0, .rates = &.{}, .n_max = 0 },                 .{ .len = -1.0, .rates = &.{0.5}, .n_max = 0 },             .{ .len = 1.7, .rates = &.{}, .n_max = 8 },
        .{ .len = 1.0000001, .rates = &.{}, .n_max = 2 },           .{ .len = 3.999999, .rates = &.{}, .n_max = 3 },            .{ .len = -1.0, .rates = &.{ 1.0, 1.0, 0.0 }, .n_max = 3 },
        .{ .len = -1.0, .rates = &.{ nan, 0.1, 0.0 }, .n_max = 3 }, .{ .len = std.math.inf(f64), .rates = &.{}, .n_max = 4 },   .{ .len = -2.0, .rates = &.{}, .n_max = 4 },
        .{ .len = 7.25, .rates = &.{}, .n_max = 16 },
    };
    for (cases) |cs| {
        var p: params_mod.Speculative = .{};
        p.synth_len = cs.len;
        p.synth_rates.appendSlice(gpa, cs.rates) catch unreachable;
        _ = printf("synth len=%a rates=%zu n_max=%d ->", cs.len, cs.rates.len, cs.n_max);
        if (spec_mod.synthRatesResolve(gpa, &p, cs.n_max)) |r| {
            for (r) |x| _ = printf(" %a", x);
            _ = printf("\n");
        } else |e| switch (e) {
            error.InvalidArgument => _ = printf(" invalid_argument: %.*s\n", @as(c_int, @intCast(tc.arg.what().len)), tc.arg.what().ptr),
            else => unreachable,
        }
    }
    var k: i32 = 0;
    while (k < 12) : (k += 1) {
        const l = spec_mod.getOutputLimits(@mod(k * 37, 50), @mod(k, 5), k - 3);
        _ = printf("limits %d %d %d -> %d %d\n", @mod(k * 37, 50), @mod(k, 5), k - 3, l.total, l.per_seq);
    }
}

fn apiGguf() void {
    const env = c.getenv("SPEC_GGUFS") orelse return;
    var it = std.mem.splitScalar(u8, std.mem.span(env), ':');
    while (it.next()) |path| {
        if (path.len == 0) continue;
        const z = gpa.dupeSentinel(u8, path, 0) catch unreachable;
        const t = spec_mod.typesFromGguf(gpa, z) catch unreachable;
        const s = spec_mod.typeNameStr(gpa, t) catch unreachable;
        _ = printf("from_gguf %s -> '%s'\n", z.ptr, s.ptr);
        // v0.7.0: common_get_gguf_info replaces common_get_decision_type(fname)
        const info = tc.common.getGgufInfo(path);
        _ = printf("gguf_info %s -> decision %d n_ctx_train %u\n", z.ptr, @backingInt(info.decision_type), info.n_ctx_train);
    }
}

fn apiNgramSimple() void {
    var g: StreamGen = .{ .x = 7 };
    var k: u32 = 0;
    while (k < 40) : (k += 1) {
        const all = g.make(20 + k * 5, 6 + k % 20, 30 + k);
        const cfg: ngram_map.SimpleConfig = .{ .size_ngram = @intCast(1 + k % 4), .size_mgram = @intCast(1 + k % 9) };
        const sampled = all[all.len - 1];
        var d = ngram_map.simpleDraft(gpa, cfg, all[0 .. all.len - 1], sampled) catch unreachable;
        defer d.deinit(gpa);
        _ = printf("simple %d n=%d m=%d", k, @as(c_int, cfg.size_ngram), @as(c_int, cfg.size_mgram));
        printTokens("", d.items);
        _ = printf("\n");
    }
}

fn dumpMap(m: *const ngram_map.Map) void {
    var nz: usize = 0;
    for (m.key_map) |v| {
        if (v != 0) nz += 1;
    }
    _ = printf("  map keys=%zu last_begin=%zu created=%d key_idx=%zu val_idx=%u check=%zu kmlast=%u nz=%zu kmhash=%016llx\n", m.keys.items.len, m.size_last_begin, @as(c_int, @intFromBool(m.last_draft_created)), m.last_draft_key_idx, @as(c_uint, m.last_draft_value_idx), m.idx_last_check, m.key_map_last_idx, nz, fnv(std.mem.sliceAsBytes(m.key_map), fnv0));
    for (m.keys.items) |k| {
        _ = printf("   key %zu %zu %u:", k.key_idx, k.stat_idx, @as(c_uint, k.key_num));
        for (k.values) |v| _ = printf(" (%zu,%u,%d)", v.value_idx, @as(c_uint, v.value_num), @as(c_int, v.n_accepted));
        _ = printf("\n");
    }
}

fn apiNgramMap() void {
    var g: StreamGen = .{ .x = 11 };
    var k: u32 = 0;
    while (k < 16) : (k += 1) {
        const key_only = k % 2 == 0;
        var m = ngram_map.Map.init(gpa, @intCast(2 + k % 3), @intCast(2 + k % 5), key_only, @intCast(1 + k % 3)) catch unreachable;
        defer m.deinit();
        const all = g.make(400, 5 + k % 7, 60);
        var len: usize = 30 + k;
        ngram_map.begin(&m, all[0..len]);
        _ = printf("map %d n=%d m=%d key_only=%d hits=%d\n", k, @as(c_int, m.size_key), @as(c_int, m.size_value), @as(c_int, @intFromBool(key_only)), @as(c_int, m.min_hits));
        var step: u32 = 0;
        while (step < 40 and len + 1 < all.len) : (step += 1) {
            var d: std.ArrayList(c.llama_token) = .empty;
            defer d.deinit(gpa);
            ngram_map.draft(&m, all[0..len], all[len], &d) catch unreachable;
            _ = printf("  step %d len %zu", step, len);
            printTokens(" draft", d.items);
            _ = printf("\n");
            var acc: usize = 0;
            while (acc < d.items.len and len + 1 + acc < all.len and all[len + 1 + acc] == d.items[acc]) acc += 1;
            ngram_map.accept(&m, @intCast(acc));
            len += 1 + acc;
            if (step % 13 == 12) {
                const back = if (len > 40) len - 25 else len;
                ngram_map.begin(&m, all[0..back]);
                len = back;
                _ = printf("  rebegin %zu\n", back);
            }
        }
        dumpMap(&m);
    }
}

fn apiNgramMod() void {
    var g: StreamGen = .{ .x = 13 };
    var k: u32 = 0;
    while (k < 6) : (k += 1) {
        var mod = ngram_mod.Mod.init(gpa, @intCast(2 + k), 97 + k * 31) catch unreachable;
        defer mod.deinit();
        const toks = g.make(300, 4 + k * 3, 50);
        const n = mod.getN();
        var i: usize = 0;
        while (i + n < toks.len) : (i += 1) mod.add(toks.ptr + i);
        _ = printf("mod %d n=%zu used=%zu size=%zu bytes=%zu:", k, n, mod.getUsed(), mod.size(), mod.sizeBytes());
        var h: u64 = fnv0;
        i = 0;
        while (i + n < toks.len) : (i += 1) {
            const v: i32 = mod.get(toks.ptr + i);
            const ix: usize = mod.idx(toks.ptr + i);
            h = fnv(std.mem.asBytes(&v), h);
            h = fnv(std.mem.asBytes(&ix), h);
        }
        _ = printf(" %016llx\n", h);
        mod.reset();
        _ = printf("  reset used=%zu\n", mod.getUsed());
    }
}

fn apiNgramCache(dir: []const u8) void {
    var g: StreamGen = .{ .x = 17 };
    var k: u32 = 0;
    while (k < 8) : (k += 1) {
        var ctx = ngram_cache.Cache.init(gpa) catch unreachable;
        var dyn = ngram_cache.Cache.init(gpa) catch unreachable;
        var sta = ngram_cache.Cache.init(gpa) catch unreachable;
        const a = g.make(200 + k * 40, 6 + k * 4, 55);
        const b = g.make(150, 6 + k * 4, 55);
        ngram_cache.update(&ctx, ngram_cache.ngram_min, ngram_cache.ngram_max, a, @intCast(a.len), false) catch unreachable;
        ngram_cache.update(&sta, 2, 2, b, @intCast(b.len), false) catch unreachable;
        const more = g.make(30, 6 + k * 4, 55);
        const a2 = std.mem.concat(gpa, c.llama_token, &.{ a, more }) catch unreachable;
        ngram_cache.update(&ctx, ngram_cache.ngram_min, ngram_cache.ngram_max, a2, @intCast(more.len), false) catch unreachable;
        ngram_cache.update(&dyn, ngram_cache.ngram_min, ngram_cache.ngram_max, b, @intCast(b.len), false) catch unreachable;
        _ = printf("cache %d\n", k);
        dumpCache(" ctx", &ctx);
        dumpCache(" sta", &sta);
        var d: u32 = 0;
        while (d < 20) : (d += 1) {
            const inp = a2[0 .. 10 + (d * 37) % (a2.len - 10)];
            var draft: std.ArrayList(c.llama_token) = .empty;
            defer draft.deinit(gpa);
            draft.append(gpa, inp[inp.len - 1]) catch unreachable;
            ngram_cache.draft(inp, &draft, gpa, @intCast(1 + d % 9), ngram_cache.ngram_min, ngram_cache.ngram_max, &ctx, &dyn, &sta) catch unreachable;
            _ = printf(" draft %d", d);
            printTokens("", draft.items);
            _ = printf("\n");
        }
        const f1 = std.fmt.allocPrintSentinel(gpa, "{s}/cache_{d}.bin", .{ dir, k }, 0) catch unreachable;
        ngram_cache.save(&ctx, f1);
        var sz: usize = 0;
        var h = fileHash(f1, &sz);
        _ = printf(" saved %zu %016llx\n", sz, h);
        var loaded = ngram_cache.load(gpa, f1) catch unreachable;
        dumpCache(" loaded", &loaded);
        ngram_cache.merge(&loaded, &dyn) catch unreachable;
        dumpCache(" merged", &loaded);
        const f2 = std.fmt.allocPrintSentinel(gpa, "{s}/merged_{d}.bin", .{ dir, k }, 0) catch unreachable;
        ngram_cache.save(&loaded, f2);
        h = fileHash(f2, &sz);
        _ = printf(" saved merged %zu %016llx\n", sz, h);
        var copy = ngram_cache.cloneCache(gpa, &loaded) catch unreachable;
        dumpCache(" copy", &copy);
        ngram_cache.deinitCache(&copy);
        ngram_cache.deinitCache(&loaded);
        ngram_cache.deinitCache(&ctx);
        ngram_cache.deinitCache(&dyn);
        ngram_cache.deinitCache(&sta);
    }
    const missing = std.fmt.allocPrintSentinel(gpa, "{s}/does-not-exist.bin", .{dir}, 0) catch unreachable;
    if (ngram_cache.load(gpa, missing)) |_| {} else |_| {
        // std::ifstream::failure's what(): the message and the iostream category's text
        _ = printf("load missing: Unable to open file %s: unspecified iostream_category error\n", missing.ptr);
    }
}

fn rbState(s: rbm.State) [*:0]const u8 {
    return switch (s) {
        .idle => "idle",
        .counting => "counting",
        .forcing => "forcing",
        .waiting_utf8 => "waiting_utf8",
        .done => "done",
    };
}

fn rbPrint(label: [*:0]const u8, s: ?*c.llama_sampler) void {
    _ = printf("  %s %s", label, rbState(rbm.getState(s)));
    if (rbm.getEndMatch(s)) |em| printTokens(" end", em);
    var d: [12]c.llama_token_data = undefined;
    for (&d, 0..) |*x, i| x.* = .{ .id = @intCast(i), .logit = 1.0 + @as(f32, @floatFromInt(i)), .p = 0.0 };
    var a: c.llama_token_data_array = .{ .data = &d, .size = 12, .selected = -1, .sorted = false };
    c.llama_sampler_apply(s, &a);
    _ = printf(" apply:");
    for (d, 0..) |x, i| {
        if (std.math.isInf(x.logit)) _ = printf(" -%d", @as(c_int, @intCast(i)));
    }
    _ = printf("\n");
}

fn apiReasoningBudget(vocab: ?*const c.llama_vocab) void {
    var g: StreamGen = .{ .x = 19 };
    const T = c.llama_token;
    const starts = [_][]const []const T{ &.{&.{ 1, 2 }}, &.{ &.{1}, &.{ 1, 2 }, &.{}, &.{1} }, &.{ &.{ 9, 1, 2 }, &.{2} } };
    const ends = [_][]const []const T{ &.{ &.{ 3, 4 }, &.{5} }, &.{&.{4}}, &.{ &.{ 3, 4, 5 }, &.{ 4, 5 }, &.{ 3, 4 } } };
    const forced = [_][]const T{ &.{ 7, 3, 4 }, &.{}, &.{ 8, 5 } };
    const budgets = [_]i32{ 0, 1, 2, 5, std.math.maxInt(i32), -3 };
    var n: u32 = 0;
    for (0..starts.len) |si| {
        for (budgets) |budget| {
            var st: u32 = 0;
            while (st < 5) : (st += 1) {
                const use_vocab = n % 3 != 0;
                const s = rbm.init(if (use_vocab) vocab else null, starts[si], ends[si], forced[si], budget, @fromBackingInt(@intCast(st))) catch unreachable;
                _ = printf("rb %d start=%zu budget=%d init=%s vocab=%d name=%s\n", n, si, budget, rbState(@fromBackingInt(@intCast(st))), @as(c_int, @intFromBool(use_vocab)), c.llama_sampler_name(s));
                var cl: ?*c.llama_sampler = null;
                const toks = g.make(60, 12, 20);
                for (toks, 0..) |tk, i| {
                    const t: T = if (i % 11 == 10 and use_vocab) @intCast(100000 + (g.next() % 50000)) else tk;
                    c.llama_sampler_accept(s, t);
                    _ = printf("  tok %d ->", t);
                    rbPrint("", s);
                    if (i == 20) cl = c.llama_sampler_clone(s);
                    if (i == 30) _ = printf("  force %d\n", @as(c_int, @intFromBool(rbm.force(s))));
                    if (i == 40 and cl != null) {
                        c.llama_sampler_copy(cl, s);
                        rbPrint("copied", s);
                    }
                    if (i == 50) {
                        c.llama_sampler_reset(s);
                        rbPrint("reset", s);
                    }
                }
                if (cl) |cc| {
                    c.llama_sampler_accept(cc, 1);
                    c.llama_sampler_accept(cc, 2);
                    rbPrint("clone", cc);
                    c.llama_sampler_free(cc);
                }
                c.llama_sampler_free(s);
                n += 1;
            }
        }
    }
    _ = printf("rb null %s %d %d\n", rbState(rbm.getState(null)), @as(c_int, @intFromBool(rbm.getEndMatch(null) == null)), @as(c_int, @intFromBool(rbm.force(null))));
}

// every one- to four-byte string over the bytes that decide a UTF-8 boundary
fn apiUtf8() void {
    const bytes = [_]u8{ 0x00, 0x41, 0x7f, 0x80, 0xbf, 0xc0, 0xc3, 0xdf, 0xe0, 0xe6, 0xef, 0xf0, 0xf4, 0xf7, 0xf8, 0xff };
    var h: u64 = fnv0;
    var n_complete: usize = 0;
    var n: usize = 0;
    var len: usize = 0;
    while (len <= 4) : (len += 1) {
        var total: usize = 1;
        for (0..len) |_| total *= bytes.len;
        for (0..total) |idx| {
            var str: [4]u8 = undefined;
            var x = idx;
            for (0..len) |k| {
                str[k] = bytes[x % bytes.len];
                x /= bytes.len;
            }
            const ok = rbm.utf8IsComplete(str[0..len]);
            n_complete += @intFromBool(ok);
            n += 1;
            const b: u8 = @intFromBool(ok);
            h = fnv(&.{b}, h);
        }
    }
    _ = printf("utf8 %zu strings, %zu complete, %016llx\n", n, n_complete, h);
}

fn typeStr(t: c_int) [*:0]const u8 {
    inline for (@typeInfo(params_mod.SamplerType).@"enum".field_values) |v| {
        if (v == t) return (gpa.dupeSentinel(u8, @as(params_mod.SamplerType, @fromBackingInt(@intCast(v))).toStr(), 0) catch unreachable).ptr;
    }
    return "";
}

fn typeChr(t: c_int) c_int {
    inline for (@typeInfo(params_mod.SamplerType).@"enum".field_values) |v| {
        if (v == t) return @as(params_mod.SamplerType, @fromBackingInt(@intCast(v))).toChr();
    }
    return '?';
}

fn apiSamplerTables() void {
    var p: params_mod.Sampling = .{};
    var s = sampling.paramsPrint(gpa, &p) catch unreachable;
    _ = printf("print default\n%.*s\n", @as(c_int, @intCast(s.len)), s.ptr);
    p.penalty_last_n = 13;
    p.penalty_repeat = 1.1;
    p.top_k = 7;
    p.top_p = 0.5;
    p.temp = 0.123456;
    p.mirostat = 2;
    p.adaptive_target = 0.33;
    s = sampling.paramsPrint(gpa, &p) catch unreachable;
    _ = printf("print set\n%.*s\n", @as(c_int, @intCast(s.len)), s.ptr);
    const names = [_][]const u8{ "top_k", "TOP-K", "topk", "nucleus", "temp", "typ", "typ_p", "Min-P", "minp", "dry", "xtc", "infill", "penalties", "adaptive-p", "adaptivep", "top_n_sigma", "top-n-sigma", "topnsigma", "bogus", "", "top_k " };
    for (sampling.typesFromNames(gpa, &names) catch unreachable) |t| _ = printf("name -> %s %c\n", typeStr(@backingInt(t)), typeChr(@backingInt(t)));
    for (sampling.typesFromChars(gpa, "dkypsmtxiea?Kz") catch unreachable) |t| _ = printf("chr -> %s\n", typeStr(@backingInt(t)));
    var t: c_int = 0;
    while (t <= 13) : (t += 1) _ = printf("type %d %s %c\n", t, typeStr(t), typeChr(t));
}

// v0.7.0's common_sampling_add_preserved_tokens and common_sampling_add_grammar_triggers
fn printSamplingLists(s: *const params_mod.Sampling) void {
    _ = printf("  preserved");
    for (s.preserved_tokens.items) |t| _ = printf(" %d", t);
    _ = printf("\n  triggers");
    for (s.grammar_triggers.items) |t| _ = printf(" {%d '%.*s' %d}", @backingInt(t.type), @as(c_int, @intCast(t.value.len)), t.value.ptr, t.token);
    _ = printf("\n");
}

fn apiSamplingLists(vocab: ?*const c.llama_vocab) void {
    if (vocab == null) return;
    const pres = [_][]const u8{ "<think>", "</think>", "hello world", "<|im_start|>", "", "<tool_call>", "</think>", " the", "the" };
    var k: usize = 0;
    while (k < 6) : (k += 1) {
        var s: params_mod.Sampling = .{};
        s.grammar_lazy = k % 2 == 1;
        sampling.addPreservedTokens(gpa, &s, vocab, pres[0 .. (k * 2) % pres.len]) catch unreachable;
        var trig: std.ArrayList(params_mod.GrammarTrigger) = .empty;
        if (k >= 1) trig.append(gpa, .{ .type = .word, .value = "<think>" }) catch unreachable;
        if (k >= 2) trig.append(gpa, .{ .type = .word, .value = "hello world" }) catch unreachable;
        if (k >= 3) trig.append(gpa, .{ .type = .pattern, .value = "a|b" }) catch unreachable;
        if (k >= 3) trig.append(gpa, .{ .type = .token, .value = "x", .token = 42 }) catch unreachable;
        if (k == 4) trig.append(gpa, .{ .type = .word, .value = "the" }) catch unreachable;
        if (k == 5) trig.append(gpa, .{ .type = .pattern_full, .value = "^x$" }) catch unreachable;
        if (k == 1) trig.clearRetainingCapacity();
        _ = printf("sampling lists %d lazy=%d\n", @as(c_int, @intCast(k)), @as(c_int, @intFromBool(s.grammar_lazy)));
        sampling.addGrammarTriggers(gpa, &s, vocab, trig.items) catch |e| switch (e) {
            error.Runtime => _ = printf("  runtime_error: %.*s\n", @as(c_int, @intCast(tc.arg.what().len)), tc.arg.what().ptr),
            else => unreachable,
        };
        printSamplingLists(&s);
    }
}

// the self-speculative drafters through the public API, without a model
fn apiSpec() void {
    var g: StreamGen = .{ .x = 23 };
    const T = params_mod.SpeculativeType;
    const combos = [_][]const T{
        &.{.ngram_simple},                  &.{.ngram_map_k}, &.{.ngram_map_k4v},
        &.{.ngram_mod},                     &.{.ngram_cache}, &.{ .ngram_mod, .ngram_map_k, .ngram_simple },
        &.{ .ngram_cache, .ngram_map_k4v }, &.{.none},        &.{.draft_eagle3},
        &.{.draft_simple},                  &.{.ngram_mod},   &.{.ngram_mod},
    };
    for (combos, 0..) |combo, ci| {
        var p: params_mod.Speculative = .{};
        p.types.appendSlice(gpa, combo) catch unreachable;
        p.ngram_simple.size_n = @intCast(2 + ci % 3);
        p.ngram_simple.size_m = @intCast(3 + ci % 5);
        p.ngram_map_k.size_n = 2;
        p.ngram_map_k.size_m = 4;
        p.ngram_map_k4v.size_n = 2;
        p.ngram_map_k4v.size_m = 3;
        p.ngram_map_k4v.min_hits = @intCast(1 + ci % 2);
        p.ngram_mod.n_match = @intCast(2 + ci % 3);
        p.ngram_mod.n_max = if (ci == 11) 4 else 6;
        p.ngram_mod.n_min = if (ci == 10) 3 else 1;
        const names = spec_mod.typeNameStr(gpa, p.types.items) catch unreachable;
        _ = printf("spec %zu '%s'", ci, names.ptr);
        const spec = spec_mod.init(&p, 2) catch |e| switch (e) {
            error.OutOfMemory => unreachable,
            else => {
                _ = printf(" exception: %.*s\n", @as(c_int, @intCast(tc.arg.what().len)), tc.arg.what().ptr);
                continue;
            },
        };
        _ = printf(" -> %s n_max=%d\n", @as([*:0]const u8, if (spec != null) "yes" else "null"), spec_mod.nMax(spec));
        const sp = spec orelse continue;
        var all: [2][]c.llama_token = undefined;
        var draft: [2]std.ArrayList(c.llama_token) = .{ .empty, .empty };
        var len = [2]usize{ 40, 52 };
        for (0..2) |s| {
            all[s] = g.make(700, @intCast(6 + ci + s), 75);
            spec_mod.begin(sp, @intCast(s), all[s][0..len[s]]);
        }
        var step: c_int = 0;
        while (step < 70) : (step += 1) {
            for (0..2) |s| {
                draft[s].clearRetainingCapacity();
                spec_mod.getDraftParams(sp, @intCast(s)).* = .{
                    .drafting = if (s == 1 and @mod(step, 9) == 4) false else true,
                    .n_max = if (@mod(step, 7) == 3) 2 else -1,
                    .pos0 = @intCast(len[s]),
                    .id_last = all[s][len[s]],
                    .prompt = all[s][0..len[s]],
                    .result = &draft[s],
                };
            }
            spec_mod.draft(sp);
            for (0..2) |s| {
                var acc: usize = 0;
                while (acc < draft[s].items.len and all[s][len[s] + 1 + acc] == draft[s].items[acc]) acc += 1;
                if (step >= 12 and step < 24) acc = 0;
                // a quarter accepted exactly, the streak's boundary
                if (ci == 11) acc = if (draft[s].items.len == 4) 1 else 0;
                _ = printf("  step %d seq %d len %zu", step, @as(c_int, @intCast(s)), len[s]);
                printTokens(" draft", draft[s].items);
                _ = printf(" acc %zu\n", acc);
                spec_mod.accept(sp, @intCast(s), @intCast(acc));
                len[s] += 1 + acc;
                if (@mod(step, 29) == 28) {
                    len[s] -= 20;
                    spec_mod.begin(sp, @intCast(s), all[s][0..len[s]]);
                }
            }
        }
        spec_mod.printStats(sp);
        spec_mod.free(sp);
    }
}

fn runApi(vocab: ?*const c.llama_vocab) void {
    apiTypes();
    apiNMax();
    apiSynth();
    apiGguf();
    apiNgramSimple();
    apiNgramMap();
    apiNgramMod();
    const dir = c.getenv("SPEC_TMP");
    apiNgramCache(if (dir) |d| std.mem.span(d) else ".");
    apiReasoningBudget(vocab);
    apiUtf8();
    apiSamplerTables();
    apiSamplingLists(vocab);
    // v0.7.0's common_log_set_stdo: off, a line is written nowhere; on again, the next one is
    tc.log.setStdo(false);
    tc.log.inf("%s: stdo off, not written\n", .{@as([*:0]const u8, "run_api")});
    tc.log.setStdo(true);
    tc.log.inf("%s: stdo on, written\n", .{@as([*:0]const u8, "run_api")});
    apiSpec();
}

fn promptTokens(ctx: ?*c.llama_context, text: []const u8) []c.llama_token {
    if (std.mem.startsWith(u8, text, "#ids:")) {
        var r: std.ArrayList(c.llama_token) = .empty;
        const z = gpa.dupeSentinel(u8, text[5..], 0) catch unreachable;
        var p: [*:0]u8 = z.ptr;
        while (p[0] != 0) {
            var end: [*c]u8 = undefined;
            r.append(gpa, @intCast(c.strtol(p, &end, 10))) catch unreachable;
            p = @ptrCast(end);
            if (p[0] == ',') p += 1;
        }
        return r.toOwnedSlice(gpa) catch unreachable;
    }
    return tc.common.tokenize(gpa, c.llama_model_get_vocab(c.llama_get_model(ctx)), text, true, true) catch unreachable;
}

fn printCands(smpl: *sampling.Sampler, s: c_int) void {
    const cur = smpl.getCandidates(true);
    _ = printf("  cand seq %d size %zu sel %lld sorted %d:", s, cur.size, @as(c_longlong, cur.selected), @as(c_int, @intFromBool(cur.sorted)));
    for (0..@min(cur.size, 6)) |i| {
        _ = printf(" %d:%08x:%08x", cur.data[i].id, @as(u32, @bitCast(cur.data[i].p)), @as(u32, @bitCast(cur.data[i].logit)));
    }
    _ = printf("\n");
}

extern fn setlocale(category: c_int, locale: [*:0]const u8) ?[*:0]u8;
/// `LC_NUMERIC`: 4 on Darwin, 1 on glibc.
const lc_numeric: c_int = if (@import("builtin").os.tag == .linux) 1 else 4;

fn noLog(_: c.ggml_log_level, _: [*c]const u8, _: ?*anyopaque) callconv(.c) void {}

fn tokEnv(ctx: ?*c.llama_context, env: [*:0]const u8, def: []const u8) std.ArrayList(c.llama_token) {
    const v: []const u8 = if (c.getenv(env)) |x| std.mem.span(x) else def;
    const t = tc.common.tokenize(gpa, c.llama_model_get_vocab(c.llama_get_model(ctx)), v, false, true) catch unreachable;
    return std.ArrayList(c.llama_token).fromOwnedSlice(t);
}

fn runDecode(params: *params_mod.Params) !u8 {
    // --spec-synth-len and --spec-synth-rates are the server's options, not this example's
    if (c.getenv("SPEC_SYNTH_LEN")) |v| params.speculative.synth_len = c.atof(v);
    if (c.getenv("SPEC_SYNTH_RATES")) |v| {
        var p: [*c]u8 = @constCast(v);
        while (p[0] != 0) {
            var end: [*c]u8 = undefined;
            try params.speculative.synth_rates.append(gpa, c.strtod(p, &end));
            p = if (end[0] == ',') end + 1 else end;
        }
    }
    const output_limits = spec_mod.getOutputLimits(params.n_batch, params.n_parallel, spec_mod.nMaxParams(&params.speculative));
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;
    _ = printf("limits %d %d n_max(params)=%d\n", output_limits.total, output_limits.per_seq, spec_mod.nMaxParams(&params.speculative));

    var init_tgt = try tc.init.initFromParams(gpa, params, false);
    defer init_tgt.deinit();
    const model_tgt = init_tgt.model;
    const ctx_tgt = init_tgt.context;
    if (model_tgt == null or ctx_tgt == null) {
        _ = printf("target failed\n");
        return 1;
    }
    c.llama_log_set(noLog, null);

    var spec_init: spec_mod.InitResult = .{};
    defer spec_init.deinit();
    {
        var params_dft = spec_mod.baseParamsToSpeculative(params);
        const mp = try gpa.dupeSentinel(u8, params_dft.model.path, 0);
        _ = printf("params_dft model='%s' ngl=%d split=%d ctk=%d ctv=%d outs=%d per_seq=%d embd=%d pool=%d threads=%d/%d\n", mp.ptr, params_dft.n_gpu_layers, @as(c_int, @intCast(params_dft.split_mode)), @as(c_int, @intCast(params_dft.cache_type_k)), @as(c_int, @intCast(params_dft.cache_type_v)), params_dft.n_outputs_max, params_dft.n_outputs_max_per_seq, @as(c_int, @intFromBool(params_dft.embedding)), @as(c_int, @intCast(params_dft.pooling_type)), params_dft.cpuparams.n_threads, params_dft.cpuparams_batch.n_threads);
        // v0.7.0: the draft's own n_ubatch and no MoE cache, beside the target's context parameters
        const cp_tgt = tc.common.contextParamsToLlama(params);
        const cp_dft = tc.common.contextParamsToLlama(&params_dft);
        _ = printf("cparams ubatch=%u/%u moe_cache=%zu/%zu\n", cp_tgt.n_ubatch, cp_dft.n_ubatch, cp_tgt.moe_cache_size, cp_dft.moe_cache_size);
        spec_init = try spec_mod.initFromParams(&params_dft, model_tgt, ctx_tgt);
        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init.context();
        _ = printf("draft model=%d ctx=%d\n", @as(c_int, @intFromBool(spec_init.model() != null)), @as(c_int, @intFromBool(spec_init.context() != null)));
        // SPEC_SEQ_RM: the server's probe of both contexts, before anything is decoded; a draft
        // context with a state input decodes it with zeros (v0.7.0's common_batch_set_zero_state)
        if (c.getenv("SPEC_SEQ_RM") != null) {
            _ = printf("seq_rm tgt=%d\n", @backingInt(tc.common.contextCanSeqRm(ctx_tgt)));
            if (spec_init.context()) |cd| _ = printf("seq_rm dft=%d\n", @backingInt(tc.common.contextCanSeqRm(cd)));
        }
    }
    const ctx_dft = params.speculative.draft.ctx_dft;

    const n_seq: usize = @intCast(@max(1, params.n_parallel));
    const inp = try gpa.alloc([]c.llama_token, n_seq);
    {
        var parts: std.ArrayList([]const u8) = .empty;
        var it = std.mem.splitScalar(u8, params.prompt, '|');
        while (it.next()) |p| try parts.append(gpa, p);
        for (0..n_seq) |i| {
            inp[i] = promptTokens(ctx_tgt, parts.items[i % parts.items.len]);
            printTokens("prompt", inp[i]);
            _ = printf("\n");
        }
    }

    const smpl = try gpa.alloc(*sampling.Sampler, n_seq);
    if (c.getenv("SPEC_RBUDGET")) |rb| {
        params.sampling.reasoning_budget_tokens = c.atoi(rb);
        params.sampling.reasoning_budget_start = tokEnv(ctx_tgt, "SPEC_RB_START", "<think>");
        params.sampling.reasoning_budget_end = .empty;
        try params.sampling.reasoning_budget_end.append(gpa, tokEnv(ctx_tgt, "SPEC_RB_END", "</think>"));
        params.sampling.reasoning_budget_forced = tokEnv(ctx_tgt, "SPEC_RB_FORCED", " ok</think>");
    }
    if (c.getenv("SPEC_GEN_PROMPT")) |gp| params.sampling.generation_prompt = std.mem.span(gp);
    // a lazy grammar's triggers and a tool-call grammar, which no option sets
    if (c.getenv("SPEC_TRIGGER")) |t| {
        params.sampling.grammar_lazy = true;
        try params.sampling.grammar_triggers.append(gpa, .{ .type = .word, .value = std.mem.span(t) });
    }
    if (c.getenv("SPEC_TRIGGER_PATTERN")) |t| {
        params.sampling.grammar_lazy = true;
        try params.sampling.grammar_triggers.append(gpa, .{ .type = .pattern, .value = std.mem.span(t) });
    }
    if (c.getenv("SPEC_TRIGGER_FULL")) |t| {
        params.sampling.grammar_lazy = true;
        try params.sampling.grammar_triggers.append(gpa, .{ .type = .pattern_full, .value = std.mem.span(t) });
    }
    if (c.getenv("SPEC_TRIGGER_TOKEN")) |t| {
        params.sampling.grammar_lazy = true;
        try params.sampling.grammar_triggers.append(gpa, .{ .type = .token, .token = c.atoi(t) });
    }
    if (c.getenv("SPEC_TOOL_GRAMMAR") != null and params.sampling.grammar.len != 0) params.sampling.grammar_type = .tool_calls;
    // --n-probs is the server's and completion's option, not this example's
    if (c.getenv("SPEC_N_PROBS")) |v| params.sampling.n_probs = c.atoi(v);
    for (0..n_seq) |s| {
        smpl[s] = sampling.init(model_tgt, &params.sampling) catch |e| switch (e) {
            error.OutOfMemory => return e,
            else => {
                _ = printf("exception: %.*s\n", @as(c_int, @intCast(tc.arg.what().len)), tc.arg.what().ptr);
                return 3;
            },
        };
        const pr = try smpl[s].print(gpa);
        _ = printf("sampler %d: %.*s seed=%u\n", @as(c_int, @intCast(s)), @as(c_int, @intCast(pr.len)), pr.ptr, smpl[s].getSeed());
    }

    // -lcs and -lcd are the lookup and server examples' options, not this one's
    if (c.getenv("SPEC_LCS")) |v| params.speculative.ngram_cache.lookup_cache_static = std.mem.span(v);
    if (c.getenv("SPEC_LCD")) |v| params.speculative.ngram_cache.lookup_cache_dynamic = std.mem.span(v);
    const spec = spec_mod.init(&params.speculative, @intCast(n_seq)) catch |e| switch (e) {
        error.OutOfMemory => return e,
        else => {
            _ = printf("exception: %.*s\n", @as(c_int, @intCast(tc.arg.what().len)), tc.arg.what().ptr);
            return 3;
        },
    };
    {
        const probs: []const f64 = if (spec != null) spec_mod.getSynthProbs(spec) else &.{};
        _ = printf("spec %s n_max=%d synth=%zu:", @as([*:0]const u8, if (spec != null) "yes" else "null"), spec_mod.nMax(spec), probs.len);
        for (probs) |p| _ = printf(" %a", p);
        _ = printf("\n");
    }

    // prefill every sequence but its last token
    {
        var n: usize = 0;
        for (inp) |v| n += v.len;
        _ = &n;
        var b = tc.common.Batch.init(gpa, ctx_tgt);
        defer b.deinit();
        // SPEC_PREFILL_EMBD: the prefill as embedding rows, as an image's are, a
        // fixed pattern per position (v0.7.0's MTP drafter catches them up too)
        var embd: []f32 = &.{};
        defer gpa.free(embd);
        if (c.getenv("SPEC_PREFILL_EMBD") != null) {
            const n_embd: usize = @intCast(c.llama_model_n_embd_inp(model_tgt));
            var n_rows: usize = 0;
            for (0..n_seq) |s| n_rows += if (inp[s].len > 0) inp[s].len - 1 else 0;
            embd = try gpa.alloc(f32, n_rows * n_embd);
            for (0..n_rows) |r| {
                for (0..n_embd) |j| embd[r * n_embd + j] = @as(f32, @floatFromInt((r * 31 + j * 7) % 97)) / 97.0 - 0.5;
            }
            var r: usize = 0;
            for (0..n_seq) |s| {
                var i: usize = 0;
                while (i + 1 < inp[s].len) : ({
                    i += 1;
                    r += 1;
                }) {
                    const p: c.llama_pos = @intCast(i);
                    const pos = [4]c.llama_pos{ p, p, p, 0 };
                    _ = b.addEmbd(.{ .data = embd.ptr + r * n_embd, .n_rows = 1, .n_embd = n_embd }, &pos, @intCast(s), false);
                }
            }
        } else {
            // SPEC_PREFILL_LAST_OUT: an output on each sequence's last prefill token, as the
            // server's prompt batch has (v0.9.0's MTP keeps only that row on a shared memory)
            const last_out = c.getenv("SPEC_PREFILL_LAST_OUT") != null;
            for (0..n_seq) |s| {
                var i: usize = 0;
                while (i + 1 < inp[s].len) : (i += 1) _ = b.add(inp[s][i], @intCast(i), @intCast(s), last_out and i + 2 == inp[s].len);
            }
        }
        const rc = c.llama_process(ctx_tgt, c.LLAMA_PROCESS_TYPE_DECODE, b.get());
        _ = printf("prefill rc=%d\n", rc);
        _ = printf("process prefill %d\n", @as(c_int, @intFromBool(spec_mod.process(spec, &b))));
    }

    const id_last = try gpa.alloc(c.llama_token, n_seq);
    const prompt_tgt = try gpa.alloc(std.ArrayList(c.llama_token), n_seq);
    const n_past = try gpa.alloc(c_int, n_seq);
    const n_gen = try gpa.alloc(c_int, n_seq);
    const done = try gpa.alloc(bool, n_seq);
    const draft = try gpa.alloc(std.ArrayList(c.llama_token), n_seq);
    for (0..n_seq) |s| {
        id_last[s] = inp[s][inp[s].len - 1];
        prompt_tgt[s] = .empty;
        try prompt_tgt[s].appendSlice(gpa, inp[s][0 .. inp[s].len - 1]);
        n_past[s] = @intCast(inp[s].len - 1);
        n_gen[s] = 0;
        done[s] = false;
        draft[s] = .empty;
        spec_mod.begin(spec, @intCast(s), prompt_tgt[s].items);
    }

    const vocab = c.llama_model_get_vocab(model_tgt);
    var batch_tgt = tc.common.Batch.init(gpa, ctx_tgt);
    defer batch_tgt.deinit();
    const n_predict: c_int = if (params.n_predict < 0) 32 else params.n_predict;
    // verify by rejection, as the server does at temp > 0: the drafters keep their candidates
    const reject = c.getenv("SPEC_REJECT") != null;
    const draft_q = try gpa.alloc(spec_mod.Candidates, n_seq);
    for (draft_q) |*q| q.* = .empty;
    defer {
        for (draft_q) |*q| spec_mod.freeCandidates(q);
        gpa.free(draft_q);
    }

    var it: c_int = 0;
    while (it < 4 * n_predict) : (it += 1) {
        var any = false;
        for (0..n_seq) |s| {
            if (done[s]) continue;
            any = true;
            var n_draft_max: c_int = @as(c_int, @intCast(c.llama_n_ctx(ctx_tgt) / @as(u32, @intCast(n_seq)))) - n_past[s] - 2;
            n_draft_max = @min(n_draft_max, n_predict - n_gen[s] - 1);
            n_draft_max = @max(n_draft_max, 0);
            draft[s].clearRetainingCapacity();
            spec_mod.truncateCandidates(&draft_q[s], 0);
            if (spec != null) {
                const dp = spec_mod.getDraftParams(spec, @intCast(s));
                dp.* = .{ .drafting = true, .n_max = n_draft_max, .pos0 = n_past[s], .id_last = id_last[s], .prompt = prompt_tgt[s].items, .result = &draft[s] };
                if (reject) {
                    dp.result_q = &draft_q[s];
                    dp.temp = params.sampling.temp;
                    dp.seed = params.sampling.seed;
                }
            }
        }
        if (!any) break;
        spec_mod.draft(spec);

        batch_tgt.clear();
        const idxs = try gpa.alloc(std.ArrayList(i32), n_seq);
        defer gpa.free(idxs);
        for (0..n_seq) |s| {
            idxs[s] = .empty;
            if (done[s]) continue;
            const sid: c.llama_seq_id = @intCast(s);
            _ = printf("it %d seq %d pos %d last %d", it, sid, n_past[s], id_last[s]);
            printTokens(" draft", draft[s].items);
            if (reject) {
                var h = fnv0;
                for (draft_q[s].items) |q| {
                    const n: usize = q.items.len;
                    h = fnv(std.mem.asBytes(&n), h);
                    h = fnv(std.mem.sliceAsBytes(q.items), h);
                }
                _ = printf(" q %zu %016llx", draft_q[s].items.len, h);
            }
            _ = printf("\n");
            if (ctx_dft != null) _ = c.llama_memory_seq_rm(c.llama_get_memory(ctx_dft), sid, n_past[s], -1);
            try idxs[s].append(gpa, batch_tgt.size());
            _ = batch_tgt.add(id_last[s], n_past[s], sid, true);
            for (draft[s].items, 0..) |d, i| {
                try idxs[s].append(gpa, batch_tgt.size());
                _ = batch_tgt.add(d, n_past[s] + 1 + @as(c_int, @intCast(i)), sid, true);
            }
        }
        const rc = c.llama_process(ctx_tgt, c.LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
        if (rc != 0) {
            _ = printf("decode rc=%d\n", rc);
            break;
        }
        if (!spec_mod.process(spec, &batch_tgt)) {
            _ = printf("process failed\n");
            break;
        }

        for (0..n_seq) |s| {
            if (done[s]) continue;
            const sid: c.llama_seq_id = @intCast(s);
            const keep: ?*sampling.Sampler = if (@mod(it, 3) == 1) smpl[s].clone() else null;
            // the server's rule: rejection only over candidates for every drafted token
            const use_rejection = reject and draft_q[s].items.len != 0 and draft_q[s].items.len == draft[s].items.len;
            const ids = if (use_rejection) blk: {
                const qs = try gpa.alloc([]const c.llama_token_data, draft_q[s].items.len);
                defer gpa.free(qs);
                for (qs, draft_q[s].items) |*d, q| d.* = q.items;
                break :blk try smpl[s].sampleAndAcceptNRejection(gpa, ctx_tgt, idxs[s].items, draft[s].items, qs, false);
            } else try smpl[s].sampleAndAcceptNIdxs(gpa, ctx_tgt, idxs[s].items, draft[s].items, false);
            if (use_rejection) _ = printf("  rejection\n");
            printTokens("  ids", ids);
            _ = printf(" acc %zu last %d\n", ids.len - 1, smpl[s].last());
            printCands(smpl[s], sid);
            if (keep) |k| {
                for (ids) |t| k.accept(t, true);
                sampling.Sampler.copy(k, smpl[s]);
                k.destroy();
                if (c.llama_vocab_type(vocab) != c.LLAMA_VOCAB_TYPE_NONE) {
                    const ps = try smpl[s].prevStr(gpa, ctx_tgt, 6);
                    _ = printf("  prev '%.*s'\n", @as(c_int, @intCast(ps.len)), ps.ptr);
                }
            }
            if (spec) |sp| spec_mod.accept(sp, sid, @intCast(ids.len - 1));
            var st: std.ArrayList(u8) = .empty;
            if (spec_mod.getState(spec, sid, &st)) {
                _ = printf("  state %zu %016llx\n", st.items.len, fnv(st.items, fnv0));
                spec_mod.setState(spec, sid, st.items);
            }
            try prompt_tgt[s].append(gpa, id_last[s]);
            try prompt_tgt[s].appendSlice(gpa, ids[0 .. ids.len - 1]);
            n_past[s] += @intCast(ids.len);
            id_last[s] = ids[ids.len - 1];
            _ = c.llama_memory_seq_rm(c.llama_get_memory(ctx_tgt), sid, n_past[s], -1);
            n_gen[s] += @intCast(ids.len);
            if (n_gen[s] >= n_predict or c.llama_vocab_is_eog(vocab, id_last[s])) done[s] = true;
        }
    }
    for (0..n_seq) |s| {
        _ = printf("seq %d generated %d", @as(c_int, @intCast(s)), n_gen[s]);
        printTokens(" all", prompt_tgt[s].items);
        _ = printf("\n");
        if (s == 0 and c.getenv("SPEC_RBUDGET") != null) _ = printf("rb force %d\n", @as(c_int, @intFromBool(smpl[s].reasoningBudgetForce())));
    }
    spec_mod.printStats(spec);
    spec_mod.free(spec);
    for (smpl) |s| s.destroy();
    return 0;
}

fn run(argv_in: []const [:0]const u8) !u8 {
    _ = setlocale(lc_numeric, "C");
    if (argv_in.len < 2) return 2;
    const mode = argv_in[1];
    var argv = try gpa.dupe([:0]const u8, argv_in[1..]);
    argv[0] = argv_in[0];

    var params = try params_mod.Params.init(gpa);
    tc.common.init();
    if (!(tc.arg.parse(argv, &params, .speculative, null) catch {
        _ = printf("parse: %.*s\n", @as(c_int, @intCast(tc.arg.what().len)), tc.arg.what().ptr);
        return 1;
    })) return 1;
    c.llama_backend_init();
    c.llama_numa_init(params.numa);

    var rc: u8 = 0;
    if (std.mem.eql(u8, mode, "api")) {
        c.llama_log_set(noLog, null);
        var mp = c.llama_model_default_params();
        mp.vocab_only = true;
        const path = try gpa.dupeSentinel(u8, params.model.path, 0);
        const m = c.llama_model_load_from_file(path.ptr, mp);
        runApi(if (m != null) c.llama_model_get_vocab(m) else null);
        if (m) |mm| c.llama_model_free(mm);
    } else {
        rc = runDecode(&params) catch |e| blk: {
            _ = printf("exception: %s\n", @errorName(e).ptr);
            break :blk 3;
        };
    }
    _ = c.fflush(null);
    c.llama_backend_free();
    return rc;
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return run(argv);
}
