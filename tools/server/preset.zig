//! INI model presets: a preset is a set of `llama-server` options keyed by
//! the option, which router mode renders into a child's command line, and
//! the context that reads them from an INI file, the model cache, a models
//! directory or the router's own command line.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/preset.cpp` and `llama.cpp/common/preset.h`
//! (v0.5.0, `7fe450e19`) whole, with what they need from
//! `llama.cpp/common/arg.cpp` -- `common_params_to_map`,
//! `common_params_add_preset_options`, `common_arg::get_args` and
//! `get_env` -- and from `llama.cpp/common/common.cpp`, `fs_list`. Each
//! declaration names the C++ it replaces and the line it began at.
//!
//! # Its shape
//!
//! `common_preset::options` is a `std::map<common_arg, std::string>`,
//! ordered by `common_arg::operator<` -- `strcmp` of the first spelling --
//! and that order is what `to_args` and `to_ini` write, so `Preset.options`
//! is a list kept in that order with the same equivalence (two options are
//! one key when their first spellings agree). An option is copied by value
//! as the C++ copies a `common_arg`; its strings belong to the parser
//! context it came from, which lives as long as the router. A preset owns
//! its name and values, allocated from the allocator `Preset` methods take
//! (the server's thread-safe C allocator), and `clone`/`deinit` are the
//! C++'s copy and destructor.

const std = @import("std");
const tc = @import("tools_common");
const peg = @import("peg");
const regex = @import("llama").regex;

const arg = tc.arg;
const log = tc.log;
const Arg = arg.Arg;
const Params = tc.params.Params;
const Allocator = std.mem.Allocator;

/// Ports `COMMON_PRESET_DEFAULT_NAME` (preset.h:15 @7fe450e19).
pub const default_name = "default";

/// Ports `COMMON_ARG_PRESET_LOAD_ON_STARTUP`, `COMMON_ARG_PRESET_STOP_TIMEOUT`
/// and `COMMON_ARG_PRESET_DEDUP_CACHE_MODELS` (arg.h:14, 15, 16 @7fe450e19).
pub const env_load_on_startup = "__PRESET_LOAD_ON_STARTUP";
pub const env_stop_timeout = "__PRESET_STOP_TIMEOUT";
pub const env_dedup_cache_models = "__PRESET_DEDUP_CACHE_MODELS";

/// A thrown `std::runtime_error` or `std::invalid_argument`: the error, with
/// its `what()` in `arg.what()`.
pub const Error = error{ Throw, OutOfMemory };

// -----------------------------------------------------------------------------
// helpers

/// Ports `rm_leading_dashes` (preset.cpp:12 @7fe450e19).
fn rmLeadingDashes(str: []const u8) []const u8 {
    var pos: usize = 0;
    while (pos < str.len and str[pos] == '-') pos += 1;
    return str[pos..];
}

fn upperInPlace(s: []u8) void {
    for (s) |*ch| ch.* = std.ascii.toUpper(ch.*);
}

/// Ports `canonical_tag` (preset.cpp:20 @7fe450e19): a quant tag in upper
/// case, the part after the last `-` or `.` when that part is a tag.
fn canonicalTag(a: Allocator, tag: []const u8) Error![]u8 {
    var re = regex.Regex.compileWith("[-.]([A-Z0-9_]+)$", .{ .icase = true }) catch |err| switch (err) {
        error.Range, error.Collate, error.Ctype, error.Escape, error.Backref, error.Brack, error.Paren, error.Brace, error.BadBrace, error.BadRepeat, error.Complexity, error.Parse => unreachable,
    };
    defer re.deinit();
    var m = regex.Match.init();
    defer m.deinit();
    if (re.search(tag, &m)) {
        const canon = try a.dupe(u8, tag[m.position(1)..][0..m.length(1)]);
        upperInPlace(canon);
        return canon;
    }
    const upper = try a.dupe(u8, tag);
    upperInPlace(upper);
    return upper;
}

/// Ports `operator`< (arg.h:91 @7fe450e19), a member of `common_arg`:
/// `strcmp` of the first spellings. An option with no spellings is never
/// less.
fn argLess(x: *const Arg, y: *const Arg) bool {
    if (x.args.len == 0 or y.args.len == 0) return false;
    return std.mem.order(u8, x.args[0], y.args[0]) == .lt;
}

/// Ports `operator`== (arg.h:97 @7fe450e19), a member of `common_arg`,
/// which `std::map` reaches as neither being less.
fn argEql(x: *const Arg, y: *const Arg) bool {
    return !argLess(x, y) and !argLess(y, x);
}

/// Ports `get_env` (arg.cpp:233 @7fe450e19), a member of `common_arg`: the
/// variable and, for an option with a negative spelling, its
/// `LLAMA_ARG_NO_` twin. `string_replace_all` replaces every occurrence.
fn getEnvs(a: Allocator, opt: *const Arg) Error![]const []const u8 {
    const env = opt.env orelse return &.{};
    if (opt.args_neg.len == 0) {
        const out = try a.alloc([]const u8, 1);
        out[0] = env;
        return out;
    }
    const out = try a.alloc([]const u8, 2);
    out[0] = env;
    out[1] = try std.mem.replaceOwned(u8, a, env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
    return out;
}

// -----------------------------------------------------------------------------
// common_preset

/// One entry of `common_preset::options`.
pub const Option = struct {
    opt: Arg,
    value: []const u8,
};

/// Ports `common_preset` (preset.h:19 @7fe450e19).
pub const Preset = struct {
    name: []const u8 = "",
    // options are stored as common_arg to string mapping, representing CLI arg and its value
    options: std.ArrayList(Option) = .empty,

    /// The C++ destructor: the name, the values and the list.
    pub fn deinit(self: *Preset, gpa: Allocator) void {
        gpa.free(self.name);
        for (self.options.items) |o| gpa.free(o.value);
        self.options.deinit(gpa);
        self.* = .{};
    }

    /// The C++ copy constructor.
    ///
    /// Return: a preset owning copies of every string; `deinit` frees it.
    pub fn clone(self: *const Preset, gpa: Allocator) Error!Preset {
        var out: Preset = .{ .name = try gpa.dupe(u8, self.name) };
        errdefer out.deinit(gpa);
        try out.options.ensureTotalCapacity(gpa, self.options.items.len);
        for (self.options.items) |o| out.options.appendAssumeCapacity(.{ .opt = o.opt, .value = try gpa.dupe(u8, o.value) });
        return out;
    }

    /// `name = s`, copied.
    pub fn setName(self: *Preset, gpa: Allocator, s: []const u8) Error!void {
        const n = try gpa.dupe(u8, s);
        gpa.free(self.name);
        self.name = n;
    }

    /// `options[opt] = value`, `value` copied: replaces an equivalent key's
    /// value (the key itself stays, as `std::map` keeps the first one), or
    /// inserts in order.
    pub fn put(self: *Preset, gpa: Allocator, opt: *const Arg, value: []const u8) Error!void {
        const v = try gpa.dupe(u8, value);
        errdefer gpa.free(v);
        var lo: usize = 0;
        var hi = self.options.items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (argLess(&self.options.items[mid].opt, opt)) lo = mid + 1 else hi = mid;
        }
        if (lo < self.options.items.len and argEql(&self.options.items[lo].opt, opt)) {
            gpa.free(self.options.items[lo].value);
            self.options.items[lo].value = v;
            return;
        }
        try self.options.insert(gpa, lo, .{ .opt = opt.*, .value = v });
    }

    /// `options.find(opt)`.
    pub fn find(self: *const Preset, opt: *const Arg) ?*Option {
        for (self.options.items) |*o| if (argEql(&o.opt, opt)) return o;
        return null;
    }

    /// Ports `to_args` (preset.cpp:37 @7fe450e19), a member of
    /// `common_preset`: the preset as a command line.
    ///
    /// Parameters:
    /// - `a`: holds the list; the strings are borrowed from the preset.
    /// - `bin_path`: the program, first when not empty.
    ///
    /// Return: the arguments, or a throw for an option with two values.
    pub fn toArgs(self: *const Preset, a: Allocator, bin_path: []const u8) Error!std.ArrayList([]const u8) {
        var args: std.ArrayList([]const u8) = .empty;

        if (bin_path.len != 0) try args.append(a, bin_path);

        for (self.options.items) |*o| {
            const opt = &o.opt;
            const value = o.value;
            if (opt.is_preset_only) {
                continue; // skip preset-only options (they are not CLI args)
            }

            // use the last arg as the main arg (i.e. --long-form)
            try args.append(a, opt.args[opt.args.len - 1]);

            // handle value(s)
            if (opt.value_hint == null and opt.value_hint_2 == null) {
                // flag option, no value
                if (arg.isFalsey(value)) {
                    // use negative arg if available
                    if (opt.args_neg.len != 0) {
                        args.items[args.items.len - 1] = opt.args_neg[opt.args_neg.len - 1];
                    } else {
                        // otherwise, skip the flag
                        // TODO: maybe throw an error instead?
                        _ = args.pop();
                    }
                }
            }
            if (opt.value_hint != null) {
                // single value
                try args.append(a, value);
            }
            if (opt.value_hint != null and opt.value_hint_2 != null) {
                return arg.throw("common_preset::to_args(): option '%s' has two values, which is not supported yet", .{opt.args[opt.args.len - 1].ptr});
            }
        }

        return args;
    }

    /// Ports `to_ini` (preset.cpp:81 @7fe450e19), a member of
    /// `common_preset`.
    ///
    /// Return: the INI text, owned by `a`.
    pub fn toIni(self: *const Preset, a: Allocator) Error![]u8 {
        var ss: std.ArrayList(u8) = .empty;

        try ss.print(a, "[{s}]\n", .{self.name});
        for (self.options.items) |*o| {
            const espaced_value = try std.mem.replaceOwned(u8, a, o.value, "\n", "\\\n");
            try ss.appendSlice(a, rmLeadingDashes(o.opt.args[o.opt.args.len - 1]));
            try ss.appendSlice(a, " = ");
            try ss.appendSlice(a, espaced_value);
            try ss.append(a, '\n');
        }
        try ss.append(a, '\n');

        return ss.items;
    }

    /// Ports `set_option` (preset.cpp:96 @7fe450e19), a member of
    /// `common_preset`: the option whose variable is `env` takes `value`.
    pub fn setOption(self: *Preset, gpa: Allocator, ctx: *const Context, env: []const u8, value: []const u8) Error!void {
        // try if option exists, update it
        for (self.options.items) |*o| {
            if (o.opt.env) |e| if (std.mem.eql(u8, env, e)) {
                const v = try gpa.dupe(u8, value);
                gpa.free(o.value);
                o.value = v;
                return;
            };
        }
        // if option does not exist, we need to add it
        const opt = ctx.key_to_opt.getPtr(env) orelse {
            const z = try gpa.dupeSentinel(u8, env, 0);
            defer gpa.free(z);
            return arg.throw("%s: option with env '%s' not found in ctx_params", .{ @as([*:0]const u8, "set_option"), z.ptr });
        };
        try self.put(gpa, opt, value);
    }

    /// Ports `unset_option` (preset.cpp:114 @7fe450e19), a member of
    /// `common_preset`: the first option whose variable is `env` goes.
    pub fn unsetOption(self: *Preset, gpa: Allocator, env: []const u8) void {
        for (self.options.items, 0..) |o, i| {
            if (o.opt.env) |e| if (std.mem.eql(u8, env, e)) {
                gpa.free(o.value);
                _ = self.options.orderedRemove(i);
                return;
            };
        }
    }

    /// Ports `get_option` (preset.cpp:126 @7fe450e19), a member of
    /// `common_preset`.
    ///
    /// Return: the value, borrowed, or null.
    pub fn getOption(self: *const Preset, env: []const u8) ?[]const u8 {
        for (self.options.items) |o| {
            if (o.opt.env) |e| if (std.mem.eql(u8, env, e)) return o.value;
        }
        return null;
    }

    /// Ports `merge` (preset.cpp:136 @7fe450e19), a member of
    /// `common_preset`: `other`'s options overwrite.
    pub fn merge(self: *Preset, gpa: Allocator, other: *const Preset) Error!void {
        for (other.options.items) |*o| {
            try self.put(gpa, &o.opt, o.value); // overwrite existing options
        }
    }

    /// Ports `apply_to_params` (preset.cpp:142 @7fe450e19), a member of
    /// `common_preset`: each option's handler on `params`, or only those
    /// whose variable is in `handled_keys` when it is not empty.
    pub fn applyToParams(self: *const Preset, params: *Params, handled_keys: []const []const u8) anyerror!void {
        for (self.options.items) |*o| {
            const opt = &o.opt;
            if (handled_keys.len != 0) {
                const env = opt.env orelse continue;
                var found = false;
                for (handled_keys) |k| {
                    if (std.mem.eql(u8, k, env)) found = true;
                }
                if (!found) continue;
            }
            const val = try params.arena.dupeSentinel(u8, o.value, 0);
            // apply each option to params
            switch (opt.handler) {
                .string => |h| try h(params, val),
                .int => |h| try h(params, try arg.stoi(val)),
                .bool_ => |h| try h(params, arg.isTruthy(val)),
                // not supported yet
                .str_str => return arg.throw("%s: option with two values is not supported yet", .{@as([*:0]const u8, "apply_to_params")}),
                .void_ => |h| try h(params),
            }
        }
    }
};

/// Ports `common_presets` (preset.h:51 @7fe450e19): a
/// `std::map<std::string, common_preset>`, in name order.
pub const Presets = struct {
    entries: std.ArrayList(Preset) = .empty,

    pub fn deinit(self: *Presets, gpa: Allocator) void {
        for (self.entries.items) |*p| p.deinit(gpa);
        self.entries.deinit(gpa);
        self.* = .{};
    }

    fn lowerBound(self: *const Presets, name: []const u8) usize {
        var lo: usize = 0;
        var hi = self.entries.items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (std.mem.order(u8, self.entries.items[mid].name, name) == .lt) lo = mid + 1 else hi = mid;
        }
        return lo;
    }

    /// `find(name)`.
    pub fn get(self: *const Presets, name: []const u8) ?*Preset {
        const i = self.lowerBound(name);
        if (i < self.entries.items.len and std.mem.eql(u8, self.entries.items[i].name, name)) return &self.entries.items[i];
        return null;
    }

    /// `out[name] = preset`: takes `preset`, whose name is the key,
    /// replacing (and freeing) an existing entry.
    pub fn putOwned(self: *Presets, gpa: Allocator, preset: Preset) Error!void {
        const i = self.lowerBound(preset.name);
        if (i < self.entries.items.len and std.mem.eql(u8, self.entries.items[i].name, preset.name)) {
            self.entries.items[i].deinit(gpa);
            self.entries.items[i] = preset;
            return;
        }
        try self.entries.insert(gpa, i, preset);
    }

    pub fn count(self: *const Presets) usize {
        return self.entries.items.len;
    }

    /// The C++ copy.
    pub fn clone(self: *const Presets, gpa: Allocator) Error!Presets {
        var out: Presets = .{};
        errdefer out.deinit(gpa);
        for (self.entries.items) |*p| try out.entries.append(gpa, try p.clone(gpa));
        return out;
    }
};

// -----------------------------------------------------------------------------
// INI

/// One section of `parse_ini_from_file`'s result: its keys and values in key
/// order.
const IniSection = struct {
    name: []const u8,
    kv: std.ArrayList(struct { key: []const u8, value: []const u8 }) = .empty,

    fn set(self: *IniSection, a: Allocator, key: []const u8, value: []const u8) Allocator.Error!void {
        var lo: usize = 0;
        var hi = self.kv.items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (std.mem.order(u8, self.kv.items[mid].key, key) == .lt) lo = mid + 1 else hi = mid;
        }
        if (lo < self.kv.items.len and std.mem.eql(u8, self.kv.items[lo].key, key)) {
            self.kv.items[lo].value = value;
            return;
        }
        try self.kv.insert(a, lo, .{ .key = key, .value = value });
    }
};

const Ini = struct {
    sections: std.ArrayList(IniSection) = .empty,

    /// `parsed[name]`, created empty when missing.
    fn section(self: *Ini, a: Allocator, name: []const u8) Allocator.Error!*IniSection {
        var lo: usize = 0;
        var hi = self.sections.items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (std.mem.order(u8, self.sections.items[mid].name, name) == .lt) lo = mid + 1 else hi = mid;
        }
        if (lo < self.sections.items.len and std.mem.eql(u8, self.sections.items[lo].name, name)) return &self.sections.items[lo];
        try self.sections.insert(a, lo, .{ .name = name });
        return &self.sections.items[lo];
    }
};

/// The grammar `parse_ini_from_file` gives `build_peg_parser` (preset.cpp:184
/// @7fe450e19).
fn buildIni(_: void, p: *peg.Builder) peg.Error!peg.ParserId {
    // newline ::= "\r\n" / "\n" / "\r"
    const newline = try p.rule("newline", try p.choice(&.{ try p.literal("\r\n"), try p.literal("\n"), try p.literal("\r") }), false);

    // ws ::= [ \t]*
    const ws = try p.rule("ws", try p.chars("[ \t]", 0, -1), false);

    // comment ::= [;#] (!newline .)*
    const comment = try p.rule("comment", try p.plus(try p.chars("[;#]", 1, 1), try p.zeroOrMore(try p.plus(try p.negate(newline), try p.any()))), false);

    // eol ::= ws comment? (newline / EOF)
    const eol = try p.rule("eol", try p.plus(try p.plus(ws, try p.optional(comment)), try p.orr(newline, try p.end())), false);

    // ident ::= [a-zA-Z_] [a-zA-Z0-9_.-]*
    const ident = try p.rule("ident", try p.plus(try p.chars("[a-zA-Z_]", 1, 1), try p.chars("[a-zA-Z0-9_.-]", 0, -1)), false);

    // value ::= (!eol-start .)*
    const eol_start = try p.rule("eol-start", try p.plus(ws, try p.orr(try p.orr(try p.chars("[;#]", 1, 1), newline), try p.end())), false);
    const value = try p.rule("value", try p.zeroOrMore(try p.plus(try p.negate(eol_start), try p.any())), false);

    // header-line ::= "[" ws ident ws "]" eol
    const header_line = try p.rule("header-line", try p.plus(try p.plus(try p.plus(try p.plus(try p.plus(try p.literal("["), ws), try p.tag("section-name", try p.chars("[^]]", 1, -1))), ws), try p.literal("]")), eol), false);

    // kv-line ::= ident ws "=" ws value eol
    const kv_line = try p.rule("kv-line", try p.plus(try p.plus(try p.plus(try p.plus(try p.plus(try p.tag("key", ident), ws), try p.literal("=")), ws), try p.tag("value", value)), eol), false);

    // comment-line ::= ws comment (newline / EOF)
    const comment_line = try p.rule("comment-line", try p.plus(try p.plus(ws, comment), try p.orr(newline, try p.end())), false);

    // blank-line ::= ws (newline / EOF)
    const blank_line = try p.rule("blank-line", try p.plus(ws, try p.orr(newline, try p.end())), false);

    // line ::= header-line / kv-line / comment-line / blank-line
    const line = try p.rule("line", try p.orr(try p.orr(try p.orr(header_line, kv_line), comment_line), blank_line), false);

    // ini ::= line* EOF
    const ini = try p.rule("ini", try p.plus(try p.zeroOrMore(line), try p.end()), false);

    return ini;
}

/// What the AST visitor in `parse_ini_from_file` keeps between nodes.
const IniVisitor = struct {
    a: Allocator,
    parsed: *Ini,
    current_section: []const u8 = default_name,
    current_key: []const u8 = "",

    fn visit(self: *IniVisitor, node: *const peg.AstNode) peg.Error!void {
        if (std.mem.eql(u8, node.tag, "section-name")) {
            const s = try self.a.dupe(u8, node.text);
            self.current_section = s;
            const sec = try self.parsed.section(self.a, s);
            sec.kv.clearRetainingCapacity(); // parsed[current_section] = {};
        } else if (std.mem.eql(u8, node.tag, "key")) {
            self.current_key = try self.a.dupe(u8, node.text);
        } else if (std.mem.eql(u8, node.tag, "value") and self.current_key.len != 0 and self.current_section.len != 0) {
            const sec = try self.parsed.section(self.a, self.current_section);
            try sec.set(self.a, self.current_key, try self.a.dupe(u8, node.text));
            self.current_key = "";
        }
    }
};

/// Reads a whole file as `std::ifstream` and `istreambuf_iterator` do: a
/// directory opens and reads as nothing.
///
/// Return: the contents owned by `a`, or null when the file cannot be
/// opened.
fn readWhole(a: Allocator, path: []const u8) Error!?[]u8 {
    const z = try a.dupeSentinel(u8, path, 0);
    const fd = std.c.open(z.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var out: std.ArrayList(u8) = .empty;
    var buf: [65536]u8 = undefined;
    while (true) {
        const n = std.c.read(fd, &buf, buf.len);
        if (n <= 0) break;
        try out.appendSlice(a, buf[0..@intCast(n)]);
    }
    return out.items;
}

/// `std::filesystem::exists(path)`: `stat` succeeds.
pub fn fsExists(path: []const u8) bool {
    var buf: [std.fs.max_path_bytes + 1]u8 = undefined;
    if (path.len >= buf.len) return false;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    return tc.fs_stat.stat(@ptrCast(&buf), true) != null;
}

/// `std::filesystem::is_directory(path)`: `stat` says a directory.
pub fn fsIsDirectory(path: []const u8) bool {
    var buf: [std.fs.max_path_bytes + 1]u8 = undefined;
    if (path.len >= buf.len) return false;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    const st = tc.fs_stat.stat(@ptrCast(&buf), true) orelse return false;
    return st.mode & tc.fs_stat.IFMT == tc.fs_stat.IFDIR;
}

/// `std::filesystem::is_regular_file(path)`: `stat` says a regular file.
pub fn fsIsRegularFile(path: []const u8) bool {
    var buf: [std.fs.max_path_bytes + 1]u8 = undefined;
    if (path.len >= buf.len) return false;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    const st = tc.fs_stat.stat(@ptrCast(&buf), true) orelse return false;
    return st.mode & tc.fs_stat.IFMT == tc.fs_stat.IFREG;
}

/// Ports `parse_ini_from_file` (preset.cpp:170 @7fe450e19): sections in
/// name order, each its keys in key order. A section named twice keeps
/// only what follows its last header.
fn parseIniFromFile(a: Allocator, path: []const u8) Error!Ini {
    var parsed: Ini = .{};

    if (!fsExists(path)) {
        return throwConcat(a, "preset file does not exist: ", path);
    }

    const contents = (try readWhole(a, path)) orelse {
        return throwConcat(a, "failed to open server preset file: ", path);
    };

    var parser = peg.buildPegParser(a, {}, buildIni) catch |e| return pegThrow(e);
    defer parser.deinit();

    var ctx = peg.ParseContext.init(a, contents, peg.Flags.none) catch |e| return pegThrow(e);
    defer ctx.deinit();
    const result = parser.parse(&ctx, 0) catch |e| return pegThrow(e);
    if (!result.success()) {
        return throwConcat(a, "failed to parse server config file: ", path);
    }

    var visitor: IniVisitor = .{ .a = a, .parsed = &parsed };
    ctx.ast.visitResult(&result, &visitor, IniVisitor.visit) catch |e| return pegThrow(e);

    return parsed;
}

fn pegThrow(e: peg.Error) Error {
    if (e == error.OutOfMemory) return error.OutOfMemory;
    return arg.throwStr(peg.what(e));
}

/// `throw std::runtime_error(prefix + s)`.
pub fn throwConcat(a: Allocator, prefix: []const u8, s: []const u8) Error {
    const msg = try std.mem.concat(a, u8, &.{ prefix, s });
    return arg.throwStr(msg);
}

// -----------------------------------------------------------------------------
// fs_list

/// Ports `common_file_info` (common/common.h:904 @7fe450e19).
pub const FileInfo = struct {
    path: []const u8 = "",
    name: []const u8 = "",
    size: usize = 0, // in bytes
    is_dir: bool = false,
};

/// Ports `fs_list` (common/common.cpp:1128 @7fe450e19): the regular files,
/// and the directories when asked, of `path`, in `readdir`'s order (which
/// `directory_iterator` follows); a symbolic link counts as what it points
/// to.
///
/// Return: the entries; strings owned by `a`.
pub fn fsList(a: Allocator, path: []const u8, include_directories: bool) Error![]FileInfo {
    var files: std.ArrayList(FileInfo) = .empty;
    if (path.len == 0) return files.items;

    if (!fsExists(path) or !fsIsDirectory(path)) {
        return files.items;
    }

    const z = try a.dupeSentinel(u8, path, 0);
    const dir = std.c.opendir(z.ptr) orelse return files.items;
    defer _ = std.c.closedir(dir);
    while (std.c.readdir(dir)) |ent| {
        const name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&ent.name)), 0);
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        // `path / name`: a separator is added unless `path` ends in one
        const p = if (std.mem.endsWith(u8, path, "/"))
            try std.mem.concat(a, u8, &.{ path, name })
        else
            try std.mem.concat(a, u8, &.{ path, "/", name });
        // Only include regular files (skip directories)
        if (fsIsRegularFile(p)) {
            var info: FileInfo = .{ .path = p, .name = try a.dupe(u8, name), .is_dir = false };
            const pz = try a.dupeSentinel(u8, p, 0);
            info.size = if (tc.fs_stat.stat(pz.ptr, true)) |st| @intCast(st.size) else 0;
            try files.append(a, info);
        } else if (include_directories and fsIsDirectory(p)) {
            try files.append(a, .{ .path = p, .name = try a.dupe(u8, name), .size = 0, .is_dir = true });
        }
    }

    return files.items;
}

// -----------------------------------------------------------------------------
// common_preset_context

/// Ports `common_preset_context` (preset.h:54 @7fe450e19).
pub const Context = struct {
    gpa: Allocator,
    /// Backs the parser context's tables, the options' help text and
    /// `key_to_opt`; lives as long as the context.
    arena: *std.heap.ArenaAllocator,
    default_params: *Params, // unused for now
    ctx_params: arg.Context,
    key_to_opt: std.StringHashMapUnmanaged(Arg) = .empty,

    filter_allowed_keys: bool = false,
    allowed_keys: []const []const u8 = &.{},

    // if true, options unknown to the current example are skipped instead of being an error
    // used for config files shared by all binaries, where each binary only knows a subset of options
    ignore_unknown_keys: bool = false,

    /// Ports `common_preset_context` (preset.cpp:279 @7fe450e19), the
    /// constructor.
    ///
    /// Parameters:
    /// - `gpa`: backs the context's arena and the presets it makes.
    /// - `ex`: the program whose options apply.
    ///
    /// Return: the context, heap-allocated so its parser context's pointer
    /// to `default_params` stays valid; `deinit` frees it.
    pub fn create(gpa: Allocator, ex: tc.params.Example) Error!*Context {
        const arena = try gpa.create(std.heap.ArenaAllocator);
        arena.* = .init(gpa);
        errdefer {
            arena.deinit();
            gpa.destroy(arena);
        }
        const a = arena.allocator();
        const dp = try a.create(Params);
        dp.* = Params.init(a) catch return error.OutOfMemory;
        const self = try a.create(Context);
        self.* = .{
            .gpa = gpa,
            .arena = arena,
            .default_params = dp,
            .ctx_params = arg.parserInit(dp, ex, null) catch |e| return mapErr(e),
        };
        try addPresetOptions(&self.ctx_params);
        try getMapKeyOpt(a, &self.ctx_params, &self.key_to_opt);
        return self;
    }

    /// Frees the context and its tables; presets it made stay valid only
    /// until here, as their options borrow its strings.
    pub fn destroy(self: *Context) void {
        const gpa = self.gpa;
        const arena = self.arena;
        arena.deinit();
        gpa.destroy(arena);
    }

    /// Ports `load_from_ini` (preset.cpp:285 @7fe450e19): every section but
    /// `*`, which goes to `global`, and an empty `default`.
    pub fn loadFromIni(self: *const Context, path: []const u8, global: *Preset) Error!Presets {
        const gpa = self.gpa;
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();

        var out: Presets = .{};
        errdefer out.deinit(gpa);
        const ini_data = try parseIniFromFile(sa, path);

        for (ini_data.sections.items) |section| {
            var preset: Preset = .{};
            errdefer preset.deinit(gpa);
            var section_name: []const u8 = if (section.name.len == 0) default_name else section.name;
            if (!std.mem.eql(u8, section_name, "*") and !std.mem.eql(u8, section_name, default_name)) {
                if (std.mem.lastIndexOfScalar(u8, section_name, ':')) |colon_idx| {
                    const tag = section_name[colon_idx + 1 ..];
                    const canon_tag = try canonicalTag(sa, tag);
                    if (!std.mem.eql(u8, canon_tag, tag)) {
                        section_name = try std.mem.concat(sa, u8, &.{ section_name[0 .. colon_idx + 1], canon_tag });
                    }
                }
            }
            try preset.setName(gpa, section_name);
            const name_z = try sa.dupeSentinel(u8, preset.name, 0);
            log.dbg("loading preset: %s\n", .{name_z.ptr});
            for (section.kv.items) |kv| {
                const key = kv.key;
                const value = kv.value;
                if (std.mem.eql(u8, key, "version")) {
                    // skip version key (reserved for future use)
                    continue;
                }

                const key_z = try sa.dupeSentinel(u8, key, 0);
                const value_z = try sa.dupeSentinel(u8, value, 0);
                log.dbg("option: %s = %s\n", .{ key_z.ptr, value_z.ptr });
                if (self.filter_allowed_keys) {
                    var allowed = false;
                    for (self.allowed_keys) |k| {
                        if (std.mem.eql(u8, k, key)) allowed = true;
                    }
                    if (!allowed) return arg.throw("option '%s' is not allowed in remote presets", .{key_z.ptr});
                }
                if (self.key_to_opt.getPtr(key)) |opt| {
                    if (isBoolArg(opt)) {
                        try preset.put(gpa, opt, parseBoolArg(opt, key, value));
                    } else {
                        try preset.put(gpa, opt, value);
                    }
                    const stored = try sa.dupeSentinel(u8, preset.find(opt).?.value, 0);
                    log.dbg("accepted option: %s = %s\n", .{ key_z.ptr, stored.ptr });
                } else if (self.ignore_unknown_keys) {
                    const path_z = try sa.dupeSentinel(u8, path, 0);
                    log.wrn("ignoring option '%s' from %s: not supported by this program\n", .{ key_z.ptr, path_z.ptr });
                } else {
                    return arg.throw("option '%s' not recognized in preset '%s'", .{ key_z.ptr, name_z.ptr });
                }
            }

            if (std.mem.eql(u8, preset.name, default_name) and preset.options.items.len == 0) {
                preset.deinit(gpa);
                continue;
            }

            if (std.mem.eql(u8, preset.name, "*")) {
                // handle global preset
                global.deinit(gpa);
                global.* = preset;
            } else {
                try out.putOwned(gpa, preset);
            }
        }

        return out;
    }

    /// Ports `load_from_cache` (preset.cpp:350 @7fe450e19): one preset per
    /// cached model, `repo:tag`, naming its repository.
    pub fn loadFromCache(self: *const Context) Error!Presets {
        const gpa = self.gpa;
        var out: Presets = .{};
        errdefer out.deinit(gpa);

        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const cached_models = @import("download.zig").listCachedModels(scratch.allocator()) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            // a runtime_error from the cache lookup, carried on as one
            else => return arg.throwStr(@import("common.zig").what(e)),
        };
        for (cached_models) |model| {
            var preset: Preset = .{};
            errdefer preset.deinit(gpa);
            const s = try model.toString(scratch.allocator());
            try preset.setName(gpa, s);
            try preset.setOption(gpa, self, "LLAMA_ARG_HF_REPO", s);
            try out.putOwned(gpa, preset);
        }

        return out;
    }

    /// Ports `load_from_models_dir` (preset.cpp:387 @7fe450e19): one preset
    /// per model file, and per sub-directory holding one, with its mmproj
    /// and draft sidecars.
    pub fn loadFromModelsDir(self: *const Context, models_dir: []const u8) Error!Presets {
        const gpa = self.gpa;
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();

        if (!fsExists(models_dir) or !fsIsDirectory(models_dir)) {
            const z = try sa.dupeSentinel(u8, models_dir, 0);
            return arg.throw("error: '%s' does not exist or is not a directory\n", .{z.ptr});
        }

        var models: std.ArrayList(LocalModel) = .empty;

        const files = try fsList(sa, models_dir, true);
        for (files) |file| {
            if (file.is_dir) {
                try scanSubdir(sa, &models, file.path, file.name);
            } else if (std.mem.endsWith(u8, file.name, ".gguf")) {
                if (isMmprojFile(file.name) or isDraftFile(file.name)) {
                    continue; // companion file, cannot be loaded as a model on its own
                }
                // single file model
                const name = try std.mem.replaceOwned(u8, sa, file.name, ".gguf", "");
                try models.append(sa, .{ .name = name, .path = file.path, .path_mmproj = "", .path_draft = "" });
            }
        }

        // convert local models to presets
        var out: Presets = .{};
        errdefer out.deinit(gpa);
        for (models.items) |model| {
            var preset: Preset = .{};
            errdefer preset.deinit(gpa);
            try preset.setName(gpa, model.name);
            try preset.setOption(gpa, self, "LLAMA_ARG_MODEL", model.path);
            if (model.path_mmproj.len != 0) {
                try preset.setOption(gpa, self, "LLAMA_ARG_MMPROJ", model.path_mmproj);
            }
            if (model.path_draft.len != 0) {
                try preset.setOption(gpa, self, "LLAMA_ARG_SPEC_DRAFT_MODEL", model.path_draft);
            }
            try out.putOwned(gpa, preset);
        }

        return out;
    }

    /// Ports `load_from_args` (preset.cpp:465 @7fe450e19): the command line
    /// as one preset named `default`.
    pub fn loadFromArgs(self: *const Context, argv: []const [:0]const u8) Error!Preset {
        var preset: Preset = .{};
        errdefer preset.deinit(self.gpa);
        try preset.setName(self.gpa, default_name);

        try paramsToMap(self.gpa, self.arena.allocator(), argv, self.ctx_params.ex, &preset);

        return preset;
    }

    /// Ports `cascade` (preset.cpp:477 @7fe450e19), the overload taking two
    /// sets: `added` merged into a copy of `base`, name by name.
    pub fn cascadeSets(self: *const Context, base: *const Presets, added: *const Presets) Error!Presets {
        const gpa = self.gpa;
        var out = try base.clone(gpa); // copy
        errdefer out.deinit(gpa);
        for (added.entries.items) |*preset_added| {
            if (out.get(preset_added.name)) |target| {
                // if exists, merge
                try target.merge(gpa, preset_added);
            } else {
                // otherwise, add directly
                try out.putOwned(gpa, try preset_added.clone(gpa));
            }
        }
        return out;
    }

    /// Ports `cascade` (preset.cpp:492 @7fe450e19), the overload taking a
    /// base preset: each preset over a copy of `base`, under its own name.
    pub fn cascade(self: *const Context, base: *const Preset, presets: *const Presets) Error!Presets {
        const gpa = self.gpa;
        var out: Presets = .{};
        errdefer out.deinit(gpa);
        for (presets.entries.items) |*preset| {
            var tmp = try base.clone(gpa); // copy
            errdefer tmp.deinit(gpa);
            try tmp.setName(gpa, preset.name);
            try tmp.merge(gpa, preset);
            try out.putOwned(gpa, tmp);
        }
        return out;
    }
};

fn mapErr(e: anyerror) Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.Throw,
    };
}

/// Ports `get_map_key_opt` (preset.cpp:251 @7fe450e19), over `get_args`
/// (arg.cpp:222 @7fe450e19): every option by each of its variables and each
/// spelling without its dashes; a later option takes a key an earlier one
/// had.
fn getMapKeyOpt(a: Allocator, ctx_params: *const arg.Context, mapping: *std.StringHashMapUnmanaged(Arg)) Error!void {
    for (ctx_params.options.items) |*opt| {
        for (try getEnvs(a, opt)) |env| {
            try mapping.put(a, env, opt.*);
        }
        // `get_args()`: every spelling, the negative ones last
        for (opt.args) |s| try mapping.put(a, rmLeadingDashes(s), opt.*);
        for (opt.args_neg) |s| try mapping.put(a, rmLeadingDashes(s), opt.*);
    }
}

/// Ports `is_bool_arg` (preset.cpp:264 @7fe450e19).
fn isBoolArg(opt: *const Arg) bool {
    return opt.args_neg.len != 0;
}

/// Ports `parse_bool_arg` (preset.cpp:268 @7fe450e19): a negative spelling
/// reverses the value.
fn parseBoolArg(opt: *const Arg, key: []const u8, value: []const u8) []const u8 {
    // if this is a negated arg, we need to reverse the value
    for (opt.args_neg) |neg_arg| {
        if (std.mem.eql(u8, rmLeadingDashes(neg_arg), key)) {
            return if (arg.isTruthy(value)) "false" else "true";
        }
    }
    // otherwise, not negated
    return value;
}

/// Ports `local_model` (preset.cpp:364 @7fe450e19).
const LocalModel = struct {
    name: []const u8,
    path: []const u8,
    path_mmproj: []const u8,
    path_draft: []const u8,
};

/// Ports `draft_prefixes` (preset.cpp:372 @7fe450e19).
// TODO @ngxson: handle "eagle3-" when it's supported by common_speculative_types_from_gguf()
const draft_prefixes = [_][]const u8{ "mtp-", "dspark-", "dflash-" };

/// Ports `is_mmproj_file` (preset.cpp:374 @7fe450e19).
fn isMmprojFile(fname: []const u8) bool {
    return std.mem.indexOf(u8, fname, "mmproj") != null;
}

/// Ports `is_draft_file` (preset.cpp:378 @7fe450e19).
fn isDraftFile(fname: []const u8) bool {
    for (draft_prefixes) |prefix| {
        if (std.mem.startsWith(u8, fname, prefix)) return true;
    }
    return false;
}

/// The `scan_subdir` (preset.cpp:393 @7fe450e19) lambda of
/// `load_from_models_dir`: a sub-directory's model, its first shard if it
/// is split, and its sidecars.
fn scanSubdir(a: Allocator, models: *std.ArrayList(LocalModel), subdir_path: []const u8, name: []const u8) Error!void {
    const files = try fsList(a, subdir_path, false);
    var model_file: FileInfo = .{};
    var first_shard_file: FileInfo = .{};
    var mmproj_file: FileInfo = .{};
    var draft_file: FileInfo = .{};
    for (files) |file| {
        if (std.mem.endsWith(u8, file.name, ".gguf")) {
            if (isMmprojFile(file.name)) {
                mmproj_file = file;
            } else if (isDraftFile(file.name)) {
                if (draft_file.path.len == 0) {
                    draft_file = file; // first sidecar found wins
                }
            } else if (std.mem.indexOf(u8, file.name, "-00001-of-") != null) {
                first_shard_file = file;
            } else {
                model_file = file;
            }
        }
    }
    // single file model
    const model: LocalModel = .{
        .name = name,
        .path = if (first_shard_file.path.len == 0) model_file.path else first_shard_file.path,
        .path_mmproj = mmproj_file.path, // can be empty
        .path_draft = draft_file.path, // can be empty
    };
    if (model.path.len != 0) {
        try models.append(a, model);
    }
}

// -----------------------------------------------------------------------------
// From arg.cpp

/// Ports `common_params_add_preset_options` (arg.cpp:4738 @7fe450e19): the
/// options only a preset can set.
pub fn addPresetOptions(ctx: *arg.Context) Error!void {
    const a = ctx.params.arena;
    const Unused = struct {
        fn str(_: *Params, _: [:0]const u8) anyerror!void {}
        fn int(_: *Params, _: c_int) anyerror!void {}
    };
    // arguments below won't be treated as CLI args, only preset options
    try ctx.options.append(a, (Arg{
        .args = &.{"load-on-startup"},
        .value_hint = "NAME",
        .help = "in server router mode, autoload this model on startup",
        .handler = .{ .string = Unused.str },
    }).setEnv(env_load_on_startup).setPresetOnly());

    try ctx.options.append(a, (Arg{
        .args = &.{"stop-timeout"},
        .value_hint = "SECONDS",
        .help = "in server router mode, force-kill model instance after this many seconds of graceful shutdown",
        .handler = .{ .int = Unused.int },
    }).setEnv(env_stop_timeout).setPresetOnly());

    try ctx.options.append(a, (Arg{
        .args = &.{"dedup-cache-models"},
        .value_hint = "0|1",
        .help = "in server router mode, hide a cached model from the model list when this preset resolves to the same model file",
        .handler = .{ .string = Unused.str },
    }).setEnv(env_dedup_cache_models).setPresetOnly());
}

/// Ports `common_params_to_map` (arg.cpp:1184 @7fe450e19): the command
/// line into `out_map` -- each option's value, `"1"` or `"0"` for a flag
/// by the spelling used -- without running any handler.
///
/// Parameters:
/// - `gpa`: owns the values `out_map` keeps.
/// - `a`: backs the parser context, which the options borrow from and so
///   must outlive `out_map`.
fn paramsToMap(gpa: Allocator, a: Allocator, argv: []const [:0]const u8, ex: tc.params.Example, out_map: *Preset) Error!void {
    const dummy_params = try a.create(Params);
    dummy_params.* = Params.init(a) catch return error.OutOfMemory;
    const ctx_arg = try a.create(arg.Context);
    ctx_arg.* = arg.parserInit(dummy_params, ex, null) catch |e| return mapErr(e);

    var arg_to_options: std.StringHashMapUnmanaged(*Arg) = .empty;
    for (ctx_arg.options.items) |*opt| {
        for (opt.args) |s| try arg_to_options.put(a, s, opt);
        for (opt.args_neg) |s| try arg_to_options.put(a, s, opt);
    }

    // TODO @ngxson : find a way to deduplicate this code

    var seen_args: std.StringHashMapUnmanaged(void) = .empty;

    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg_prefix = "--";

        const arg_buf = try a.dupeSentinel(u8, argv[i], 0);
        if (std.mem.startsWith(u8, arg_buf, arg_prefix)) {
            std.mem.replaceScalar(u8, arg_buf, '_', '-');
        }
        const s: [:0]const u8 = arg_buf;
        const opt = arg_to_options.get(s) orelse {
            return arg.throw("error: invalid argument: %s", .{s.ptr});
        };
        const gop = try seen_args.getOrPut(a, s);
        if (gop.found_existing) {
            const skip = std.mem.eql(u8, s, "--spec-type");

            if (!skip) {
                log.wrn("DEPRECATED: argument '%s' specified multiple times, use comma-separated values instead (only last value will be used)\n", .{s.ptr});
            }
        }
        var val: []const u8 = "";
        if (opt.value_hint == null and opt.value_hint_2 == null) {
            // bool arg (need to reverse the meaning for negative args)
            var is_neg = false;
            for (opt.args_neg) |n| {
                if (std.mem.eql(u8, n, s)) is_neg = true;
            }
            val = if (is_neg) "0" else "1";
        }
        if (opt.value_hint != null) {
            // arg with single value
            if (i + 1 >= argv.len) return arg.throwStr("expected value for argument");
            i += 1;
            val = argv[i];
        }
        if (opt.value_hint_2 != null) {
            // TODO: support arg with 2 values
            return arg.throwStr("error: argument with 2 values is not yet supported\n");
        }
        try out_map.put(gpa, opt, val);
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "canonical tags take the part after the last separator, upper-cased" {
    const a = std.testing.allocator;
    const t1 = try canonicalTag(a, "model-q4_k_m");
    defer a.free(t1);
    try std.testing.expectEqualStrings("Q4_K_M", t1);
    const t2 = try canonicalTag(a, "q8_0");
    defer a.free(t2);
    try std.testing.expectEqualStrings("Q8_0", t2);
    const t3 = try canonicalTag(a, "a-b c");
    defer a.free(t3);
    try std.testing.expectEqualStrings("A-B C", t3);
}
