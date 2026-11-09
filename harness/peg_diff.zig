//! `make peg-diff`: `cli/peg/` and `cli/common_chat/chat_peg.zig` against
//! upstream's `peg-parser.cpp` and `chat-peg-parser.cpp`.
//!
//! **Not a port.** Both sides run the same programs: `harness/peg_ref.cpp`
//! interprets them over upstream's builder, this file over the port's, and
//! every record must be byte-identical. A program is a list of cases; a
//! case is a parser -- a description of builder calls -- and the inputs to
//! run it over. Per case: the build's outcome, the arena's `dump`, its JSON
//! form (`harness/peg_serialize.zig`; upstream's `save` until v0.7.0), its
//! GBNF eager and lazy; per input, the parse complete and
//! lenient and of every prefix (the partial state a stream sees), each
//! with its AST, the three chat mappers' messages and the exceptions on
//! the way, the tagged parser's three entry points, and the debug trace.
//! Every exception is recorded by class (`IA:` `std::invalid_argument`,
//! `JE:` `common_json_error`, `OR:` `std::out_of_range`, `RE:`
//! `std::runtime_error`) and `what()`.
//!
//! The parsers: every one upstream's PEG tests build, recorded as they run
//! (`harness/peg_rec.h`, `ref_peg_upstream`) together with the inputs they
//! parse; random descriptions over every builder call; chat grammars in the
//! shapes upstream's templates take, over random tools; and Gemma 4- and
//! MiniMax-M3-shaped grammars for their mappers. Upstream removed
//! `load`/`from_json` at v0.7.0, and with them the mutated saved arenas
//! this gate fed them. The inputs: the recorded ones, strings derived from
//! the grammar itself, and both mutated.
//!
//! The reference runs in forked children, a batch at a time: a
//! left-recursive grammar recurses until the stack runs out, upstream and
//! port alike, and so does `dump` of a rule that cycles through a tag. A
//! batch whose child dies is rerun one case per child; a case that kills
//! the reference is skipped on both sides and counted.
//!
//! A tool-related node after a `tool-close` -- undefined in the C++ until
//! v0.6.0 cleared `current_tool` there (see `chat_peg.zig`) -- is compared
//! like any other, and some mapping must reach one.
//!
//! Usage: peg_diff [case substring]
//!   PEG_DIFF_SEED=n    the generators' seed (default 1)
//!   PEG_DIFF_SCALE=n   multiplies every generated count (default 1)
//!   PEG_DIFF_STATS=1   lists every distinct error seen

const std = @import("std");
const builtin = @import("builtin");
const win_child = @import("win_child.zig");
const json = @import("json");
const peg = @import("peg");
const chat = @import("common_chat");
const cpeg = chat.chat_peg;
const peg_serialize = @import("peg_serialize.zig");

extern fn ref_peg_run(prog: [*]const u8, len: usize, out_len: *usize) [*]u8;
extern fn ref_peg_upstream(out_len: *usize) [*]u8;

extern "c" fn free(p: ?*anyopaque) void;
extern "c" fn fork() c_int;
extern "c" fn pipe(fds: *[2]c_int) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn waitpid(pid: c_int, status: *c_int, options: c_int) c_int;
extern "c" fn _exit(code: c_int) noreturn;
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn dup2(a: c_int, b: c_int) c_int;
extern "c" fn alarm(seconds: c_uint) c_uint;

const gpa = std.heap.smp_allocator;
const ParserId = peg.ParserId;

fn oom() noreturn {
    std.debug.print("FAIL: out of memory\n", .{});
    std.process.exit(2);
}

// -----------------------------------------------------------------------------
// The bytecode: harness/peg_rec.h's `peg_op`, in its order

const Op = enum(u8) {
    eps,
    start,
    end,
    literal,
    seq_empty,
    seq,
    choice_empty,
    choice,
    one_or_more,
    zero_or_more,
    optional,
    peek,
    negate,
    any,
    chars,
    ref,
    space,
    until,
    until_one_of,
    rest,
    repeat,
    repeat_n,
    dq_string,
    sq_string,
    quoted_string,
    string_content,
    json,
    json_object,
    json_string,
    json_array,
    json_number,
    json_bool,
    json_null,
    json_member,
    py_value,
    py_dict,
    py_string,
    py_array,
    py_number,
    py_bool,
    py_null,
    marker,
    schema,
    rule,
    rule_fn,
    trigger_rule,
    trigger_rule_fn,
    atomic,
    tag,
    gbnf,
    ac,
    ac_one,
    op_plus,
    op_shl,
    op_or,
    op_plus_lit,
    op_shl_lit,
    op_or_lit,
    op_lit_plus,
    op_lit_shl,
    op_lit_or,
    op_plus_assign,
    op_or_assign,
    reasoning_block,
    reasoning,
    content,
    tag_safe,
    tool,
    tool_open,
    tool_close,
    tool_id,
    tool_name,
    tool_args,
    tool_arg,
    tool_arg_open,
    tool_arg_close,
    tool_arg_name,
    tool_arg_value,
    tool_arg_string_value,
    tool_arg_json_value,
    permute,
    prefix,
    optspace,
    std_json_tools,
    std_constructed_tools,
    py_style_tools,
    set_root,
    _,
};
const BODY_END: u8 = 0xFE;
const NO_SLOT: u32 = 0xFFFFFFFF;

const MODE_FULL: u8 = 1;
const MODE_LENIENT: u8 = 2;
const MODE_PREFIXES: u8 = 4;
const MODE_DEBUG: u8 = 8;
const MODE_TAGGED: u8 = 16;

/// A description being written: every call, its slot the call's ordinal.
const Desc = struct {
    buf: std.ArrayList(u8) = .empty,
    n: u32 = 0,

    fn byte(self: *Desc, v: u8) void {
        self.buf.append(gpa, v) catch oom();
    }
    fn u32_(self: *Desc, v: u32) void {
        self.buf.appendSlice(gpa, std.mem.asBytes(&v)) catch oom();
    }
    fn i32_(self: *Desc, v: i32) void {
        self.buf.appendSlice(gpa, std.mem.asBytes(&v)) catch oom();
    }
    fn str(self: *Desc, s: []const u8) void {
        self.u32_(@intCast(s.len));
        self.buf.appendSlice(gpa, s) catch oom();
    }
    fn strs(self: *Desc, v: []const []const u8) void {
        self.u32_(@intCast(v.len));
        for (v) |s| self.str(s);
    }
    fn op(self: *Desc, code: Op) u32 {
        self.byte(@backingInt(code));
        defer self.n += 1;
        return self.n;
    }
    fn slots(self: *Desc, v: []const u32) void {
        self.u32_(@intCast(v.len));
        for (v) |s| self.u32_(s);
    }
    fn bodyEnd(self: *Desc, slot: u32) void {
        self.byte(BODY_END);
        self.u32_(slot);
    }

    // convenience: one call each
    fn leaf(self: *Desc, code: Op) u32 {
        return self.op(code);
    }
    fn un(self: *Desc, code: Op, a: u32) u32 {
        const w = self.op(code);
        self.u32_(a);
        return w;
    }
    fn bin(self: *Desc, code: Op, a: u32, b: u32) u32 {
        const w = self.op(code);
        self.u32_(a);
        self.u32_(b);
        return w;
    }
    fn lit(self: *Desc, s: []const u8) u32 {
        const w = self.op(.literal);
        self.str(s);
        return w;
    }
    fn strOp(self: *Desc, code: Op, s: []const u8) u32 {
        const w = self.op(code);
        self.str(s);
        return w;
    }
    fn plusLit(self: *Desc, a: u32, s: []const u8) u32 {
        const w = self.op(.op_plus_lit);
        self.u32_(a);
        self.str(s);
        return w;
    }
    fn litPlus(self: *Desc, s: []const u8, a: u32) u32 {
        const w = self.op(.op_lit_plus);
        self.str(s);
        self.u32_(a);
        return w;
    }
    fn seq(self: *Desc, v: []const u32) u32 {
        const w = self.op(.seq);
        self.slots(v);
        return w;
    }
    fn choice(self: *Desc, v: []const u32) u32 {
        const w = self.op(.choice);
        self.slots(v);
        return w;
    }
    fn tag(self: *Desc, t: []const u8, a: u32) u32 {
        const w = self.op(.tag);
        self.str(t);
        self.u32_(a);
        return w;
    }
    fn rule(self: *Desc, name: []const u8, a: u32, trigger: bool) u32 {
        const w = self.op(.rule);
        self.str(name);
        self.u32_(a);
        self.byte(@intFromBool(trigger));
        return w;
    }
    fn chars(self: *Desc, cls: []const u8, min: i32, max: i32) u32 {
        const w = self.op(.chars);
        self.str(cls);
        self.i32_(min);
        self.i32_(max);
        return w;
    }
    fn repeat(self: *Desc, a: u32, min: i32, max: i32) u32 {
        const w = self.op(.repeat);
        self.u32_(a);
        self.i32_(min);
        self.i32_(max);
        return w;
    }
    fn setRoot(self: *Desc, a: u32) void {
        _ = self.op(.set_root);
        self.u32_(a);
    }
};

// -----------------------------------------------------------------------------
// The port's interpreter

threadlocal var harness_msg: ?[]u8 = null;

fn harnessError(comptime fmt: []const u8, args: anytype) peg.Error {
    if (harness_msg) |m| gpa.free(m);
    harness_msg = std.fmt.allocPrint(gpa, fmt, args) catch oom();
    return error.RuntimeError;
}

/// `err()` of harness/peg_ref.cpp: the class and `what()`.
fn errText(e: anyerror) []u8 {
    if (harness_msg) |m| {
        harness_msg = null;
        defer gpa.free(m);
        return std.fmt.allocPrint(gpa, "EX:{s}", .{m}) catch oom();
    }
    const cls = switch (e) {
        error.InvalidArgument => "IA",
        error.JsonError => "JE",
        error.OutOfRange => "OR",
        error.RuntimeError => "RE",
        else => "EX",
    };
    return std.fmt.allocPrint(gpa, "{s}:{s}", .{ cls, peg.what(e) }) catch oom();
}

const Reader = struct {
    buf: []const u8,
    i: usize = 0,

    fn byte(self: *Reader) u8 {
        defer self.i += 1;
        return self.buf[self.i];
    }
    fn u32_(self: *Reader) u32 {
        defer self.i += 4;
        return std.mem.readInt(u32, self.buf[self.i..][0..4], .little);
    }
    fn i32_(self: *Reader) i32 {
        return @bitCast(self.u32_());
    }
    fn str(self: *Reader) []const u8 {
        const n = self.u32_();
        defer self.i += n;
        return self.buf[self.i..][0..n];
    }
    fn strs(self: *Reader, a: std.mem.Allocator) [][]const u8 {
        const n = self.u32_();
        const out = a.alloc([]const u8, n) catch oom();
        for (out) |*s| s.* = self.str();
        return out;
    }
};

const Interp = struct {
    cb: *cpeg.ChatBuilder,
    scratch: std.mem.Allocator,
    slots: std.ArrayList(?ParserId) = .empty,
    n: u32 = 0,

    fn get(self: *Interp, s: u32) peg.Error!ParserId {
        if (s < self.slots.items.len) if (self.slots.items[s]) |id| return id;
        return harnessError("HE:unset slot {d}", .{s});
    }
    fn put(self: *Interp, s: u32, id: ParserId) void {
        while (self.slots.items.len <= s) self.slots.append(self.scratch, null) catch oom();
        self.slots.items[s] = id;
    }
    fn list(self: *Interp, r: *Reader, exec: bool) peg.Error![]ParserId {
        const k = r.u32_();
        const out = self.scratch.alloc(ParserId, k) catch oom();
        for (out) |*x| {
            const s = r.u32_();
            x.* = if (exec) try self.get(s) else 0;
        }
        return out;
    }
    fn body(self: *Interp, r: *Reader, exec: bool) peg.Error!u32 {
        while (r.buf[r.i] != BODY_END) try self.step(r, exec);
        _ = r.byte();
        return r.u32_();
    }

    const BodyCtx = struct { it: *Interp, r: *Reader, called: bool = false };

    fn runBody(c: *BodyCtx) peg.Error!ParserId {
        c.called = true;
        const s = try c.it.body(c.r, true);
        if (s == NO_SLOT) return harnessError("HE:a body that returned nothing", .{});
        return c.it.get(s);
    }

    fn step(self: *Interp, r: *Reader, exec: bool) peg.Error!void {
        const code: Op = @fromBackingInt(@intCast(r.byte()));
        const me = self.n;
        self.n += 1;
        const cb = self.cb;
        const b = &cb.b;
        switch (code) {
            .eps, .start, .end, .seq_empty, .choice_empty, .any, .space, .rest, .dq_string, .sq_string, .quoted_string, .json, .json_object, .json_string, .json_array, .json_number, .json_bool, .json_null, .py_value, .py_dict, .py_string, .py_array, .py_number, .py_bool, .py_null, .marker => {
                if (!exec) return;
                self.put(me, switch (code) {
                    .eps => try b.eps(),
                    .start => try b.start(),
                    .end => try b.end(),
                    .seq_empty => try b.sequenceEmpty(),
                    .choice_empty => try b.choiceEmpty(),
                    .any => try b.any(),
                    .space => try b.space(),
                    .rest => try b.rest(),
                    .dq_string => try b.doubleQuotedString(),
                    .sq_string => try b.singleQuotedString(),
                    .quoted_string => try b.quotedString(),
                    .json => try b.json(),
                    .json_object => try b.jsonObject(),
                    .json_string => try b.jsonString(),
                    .json_array => try b.jsonArray(),
                    .json_number => try b.jsonNumber(),
                    .json_bool => try b.jsonBool(),
                    .json_null => try b.jsonNull(),
                    .py_value => try b.pythonValue(),
                    .py_dict => try b.pythonDict(),
                    .py_string => try b.pythonString(),
                    .py_array => try b.pythonArray(),
                    .py_number => try b.pythonNumber(),
                    .py_bool => try b.pythonBool(),
                    .py_null => try b.pythonNull(),
                    .marker => try b.marker(),
                    else => unreachable,
                });
            },
            .literal, .ref, .until, .optspace => {
                const s = r.str();
                if (!exec) return;
                self.put(me, switch (code) {
                    .literal => try b.literal(s),
                    .ref => try b.ref(s),
                    .until => try b.until(s),
                    .optspace => try cb.optspace(s),
                    else => unreachable,
                });
            },
            .seq, .choice => {
                const l = try self.list(r, exec);
                if (exec) self.put(me, if (code == .seq) try b.sequence(l) else try b.choice(l));
            },
            .one_or_more, .zero_or_more, .optional, .peek, .negate, .atomic, .reasoning_block, .reasoning, .content, .tool, .tool_open, .tool_close, .tool_id, .tool_name, .tool_args, .tool_arg, .tool_arg_open, .tool_arg_close, .tool_arg_name, .tool_arg_value, .tool_arg_string_value, .tool_arg_json_value => {
                const s = r.u32_();
                if (!exec) return;
                const p = try self.get(s);
                self.put(me, switch (code) {
                    .one_or_more => try b.oneOrMore(p),
                    .zero_or_more => try b.zeroOrMore(p),
                    .optional => try b.optional(p),
                    .peek => try b.peek(p),
                    .negate => try b.negate(p),
                    .atomic => try b.atomic(p),
                    .reasoning_block => try cb.reasoningBlock(p),
                    .reasoning => try cb.reasoning(p),
                    .content => try cb.content(p),
                    .tool => try cb.tool(p),
                    .tool_open => try cb.toolOpen(p),
                    .tool_close => try cb.toolClose(p),
                    .tool_id => try cb.toolId(p),
                    .tool_name => try cb.toolName(p),
                    .tool_args => try cb.toolArgs(p),
                    .tool_arg => try cb.toolArg(p),
                    .tool_arg_open => try cb.toolArgOpen(p),
                    .tool_arg_close => try cb.toolArgClose(p),
                    .tool_arg_name => try cb.toolArgName(p),
                    .tool_arg_value => try cb.toolArgValue(p),
                    .tool_arg_string_value => try cb.toolArgStringValue(p),
                    .tool_arg_json_value => try cb.toolArgJsonValue(p),
                    else => unreachable,
                });
            },
            .chars => {
                const s = r.str();
                const mn = r.i32_();
                const mx = r.i32_();
                if (exec) self.put(me, try b.chars(s, mn, mx));
            },
            .until_one_of => {
                const v = r.strs(self.scratch);
                if (exec) self.put(me, try b.untilOneOf(v));
            },
            .repeat => {
                const s = r.u32_();
                const mn = r.i32_();
                const mx = r.i32_();
                if (exec) self.put(me, try b.repeat(try self.get(s), mn, mx));
            },
            .repeat_n => {
                const s = r.u32_();
                const k = r.i32_();
                if (exec) self.put(me, try b.repeatN(try self.get(s), k));
            },
            .string_content => {
                const d = r.byte();
                if (exec) self.put(me, try b.stringContent(d));
            },
            .json_member => {
                const k = r.str();
                const s = r.u32_();
                if (exec) self.put(me, try b.jsonMember(k, try self.get(s)));
            },
            .schema => {
                const s = r.u32_();
                const name = r.str();
                const text = r.str();
                const raw = r.byte() != 0;
                if (!exec) return;
                const p = try self.get(s);
                var v = json.parse(gpa, text) catch |e| return peg.errors.fromJson(e);
                defer v.deinit(gpa);
                self.put(me, try b.schemaJson(p, name, &v, raw));
            },
            .rule => {
                const name = r.str();
                const s = r.u32_();
                const trig = r.byte() != 0;
                if (exec) self.put(me, try b.rule(name, try self.get(s), trig));
            },
            .trigger_rule => {
                const name = r.str();
                const s = r.u32_();
                if (exec) self.put(me, try b.triggerRule(name, try self.get(s)));
            },
            .rule_fn, .trigger_rule_fn => {
                const name = r.str();
                const trig = if (code == .rule_fn) r.byte() != 0 else true;
                if (!exec) {
                    _ = try self.body(r, false);
                    return;
                }
                var ctx: BodyCtx = .{ .it = self, .r = r };
                const res = if (code == .rule_fn)
                    try b.ruleFn(name, &ctx, runBody, trig)
                else
                    try b.triggerRuleFn(name, &ctx, runBody);
                if (!ctx.called) _ = try self.body(r, false);
                self.put(me, res);
            },
            .tag => {
                const t = r.str();
                const s = r.u32_();
                if (exec) self.put(me, try b.tag(t, try self.get(s)));
            },
            .gbnf => {
                const s = r.u32_();
                const g = r.str();
                if (exec) self.put(me, try b.gbnf(try self.get(s), g));
            },
            .ac => {
                const s = r.u32_();
                const v = r.strs(self.scratch);
                if (exec) self.put(me, try b.ac(try self.get(s), v));
            },
            .ac_one => {
                const s = r.u32_();
                const d = r.str();
                if (exec) self.put(me, try b.acOne(try self.get(s), d));
            },
            .op_plus, .op_shl, .op_or, .op_plus_assign, .op_or_assign => {
                const sa = r.u32_();
                const sb = r.u32_();
                if (!exec) return;
                var x = try self.get(sa);
                const y = try self.get(sb);
                switch (code) {
                    .op_plus => self.put(me, try b.plus(x, y)),
                    .op_shl => self.put(me, try b.shl(x, y)),
                    .op_or => self.put(me, try b.orr(x, y)),
                    .op_plus_assign => {
                        try b.plusAssign(&x, y);
                        self.put(me, x);
                    },
                    else => {
                        try b.orAssign(&x, y);
                        self.put(me, x);
                    },
                }
            },
            .op_plus_lit, .op_shl_lit, .op_or_lit => {
                const sa = r.u32_();
                const s = r.str();
                if (!exec) return;
                const x = try self.get(sa);
                self.put(me, switch (code) {
                    .op_plus_lit => try b.plusLit(x, s),
                    .op_shl_lit => try b.shlLit(x, s),
                    else => try b.orLit(x, s),
                });
            },
            .op_lit_plus, .op_lit_shl, .op_lit_or => {
                const s = r.str();
                const sa = r.u32_();
                if (!exec) return;
                const x = try self.get(sa);
                self.put(me, switch (code) {
                    .op_lit_plus => try b.litPlus(s, x),
                    .op_lit_shl => try b.litShl(s, x),
                    else => try b.litOr(s, x),
                });
            },
            .tag_safe => {
                const t = r.str();
                const m = r.str();
                const s = r.u32_();
                if (exec) self.put(me, try cb.tagWithSafeContent(t, m, try self.get(s)));
            },
            .permute => {
                const pre = r.str();
                const l = try self.list(r, exec);
                if (exec) self.put(me, try cb.permute(pre, l));
            },
            .prefix => {
                const s = r.str();
                const d = r.str();
                if (exec) self.put(me, try cb.prefix(s, d));
            },
            .std_json_tools => {
                const ss = r.str();
                const se = r.str();
                const tools = r.str();
                const par = r.byte() != 0;
                const force = r.byte() != 0;
                var o: cpeg.ChatBuilder.JsonToolsOptions = .{};
                o.name_key = r.str();
                o.args_key = r.str();
                o.array_wrapped = r.byte() != 0;
                o.function_is_key = r.byte() != 0;
                o.call_id_key = r.str();
                o.gen_call_id_key = r.str();
                o.parameters_order = r.strs(self.scratch);
                o.accept_openai_wrapper = r.byte() != 0;
                if (!exec) return;
                var v = json.parse(gpa, tools) catch |e| return peg.errors.fromJson(e);
                defer v.deinit(gpa);
                self.put(me, try cb.standardJsonTools(ss, se, &v, par, force, o));
            },
            .std_constructed_tools => {
                const k = r.u32_();
                var markers: std.ArrayList(cpeg.Marker) = .empty;
                for (0..k) |_| {
                    const key = r.str();
                    const val = r.str();
                    // a std::map built with emplace: the first of a key wins
                    var dup = false;
                    for (markers.items) |m| dup = dup or std.mem.eql(u8, m.key, key);
                    if (!dup) markers.append(self.scratch, .{ .key = key, .value = val }) catch oom();
                }
                const tools = r.str();
                const par = r.byte() != 0;
                const force = r.byte() != 0;
                if (!exec) return;
                var v = json.parse(gpa, tools) catch |e| return peg.errors.fromJson(e);
                defer v.deinit(gpa);
                self.put(me, try cb.standardConstructedTools(markers.items, &v, par, force));
            },
            .py_style_tools => {
                const tools = r.str();
                const par = r.byte() != 0;
                const lit = r.byte() != 0;
                if (!exec) return;
                var v = json.parse(gpa, tools) catch |e| return peg.errors.fromJson(e);
                defer v.deinit(gpa);
                self.put(me, try cb.pythonStyleToolCalls(&v, par, lit));
            },
            .set_root => {
                const s = r.u32_();
                if (exec) b.setRoot(try self.get(s));
            },
            _ => return harnessError("HE:unknown op {d}", .{@backingInt(code)}),
        }
    }
};

/// A description through the port's builder, built.
fn portBuild(ops: []const u8) peg.Error!peg.Arena {
    var cb = try cpeg.ChatBuilder.init(gpa);
    defer cb.deinit();
    var scratch = std.heap.ArenaAllocator.init(gpa);
    defer scratch.deinit();
    var it: Interp = .{ .cb = &cb, .scratch = scratch.allocator() };
    var r: Reader = .{ .buf = ops };
    while (r.i < ops.len) try it.step(&r, true);
    return cb.build();
}

// -----------------------------------------------------------------------------
// The port's records, in harness/peg_ref.cpp's format

const Records = std.ArrayList([]const u8);

fn rec(out: *Records, comptime fmt: []const u8, args: anytype) void {
    out.append(gpa, std.fmt.allocPrint(gpa, fmt, args) catch oom()) catch oom();
}

/// `guard`: "G:" and the value, or "E:" and the exception.
fn guardText(v: anyerror![]u8) []u8 {
    const s = v catch |e| {
        const t = errText(e);
        defer gpa.free(t);
        return std.fmt.allocPrint(gpa, "E:{s}", .{t}) catch oom();
    };
    defer gpa.free(s);
    return std.fmt.allocPrint(gpa, "G:{s}", .{s}) catch oom();
}

fn resultText(r: peg.ParseResult) []u8 {
    var s: std.ArrayList(u8) = .empty;
    s.print(gpa, "{s} {d} {d}", .{ r.type.name(), r.start, r.end }) catch oom();
    for (r.nodes) |n| s.print(gpa, " {d}", .{n}) catch oom();
    for (r.invalid_utf8) |u| s.print(gpa, " !{d}+{d}", .{ u.pos, u.len }) catch oom();
    return s.toOwnedSlice(gpa) catch oom();
}

/// `ast_invalid_text` in peg_ref.cpp: every node's invalid UTF-8 runs and
/// sanitized text (v0.5.0).
fn astInvalidText(ast: *const peg.AstArena) anyerror![]u8 {
    var s: std.ArrayList(u8) = .empty;
    for (ast.nodes.items, 0..) |*n, i| {
        if (n.invalid_utf8.len == 0) continue;
        try s.print(gpa, "\x1e{d}", .{i});
        for (n.invalid_utf8) |u| try s.print(gpa, " {d}+{d}", .{ u.pos, u.len });
        const t = try n.sanitizedText(gpa);
        defer gpa.free(t);
        try s.print(gpa, "\x1d{s}", .{t});
    }
    return s.toOwnedSlice(gpa);
}

fn msgText(m: *const chat.Msg) []u8 {
    var s: std.ArrayList(u8) = .empty;
    s.print(gpa, "{s}\x1f{s}\x1f{d}", .{ m.content, m.reasoning_content, m.tool_calls.items.len }) catch oom();
    for (m.tool_calls.items) |tc| s.print(gpa, "\x1e{s}\x1d{s}\x1d{s}", .{ tc.name, tc.arguments, tc.id }) catch oom();
    return s.toOwnedSlice(gpa) catch oom();
}

var after_close: usize = 0;

fn mapperText(ctx: *peg.ParseContext, result: *const peg.ParseResult, kind: cpeg.Mapper.Kind) ![]u8 {
    var m: chat.Msg = .{};
    defer m.deinit(gpa);
    var mapper = cpeg.Mapper.init(gpa, &m, kind);
    defer mapper.deinit();
    const r = mapper.fromAst(&ctx.ast, result);
    if (mapper.after_close) after_close += 1;
    try r;
    return msgText(&m);
}

fn parseRecords(arena: *const peg.Arena, input: []const u8, flags: u32, comptime tag: []const u8, tag_args: anytype, out: *Records) void {
    var ctx = peg.ParseContext.init(gpa, input, flags) catch oom();
    defer ctx.deinit();
    const result = arena.parse(&ctx, 0) catch |e| {
        const t = errText(e);
        defer gpa.free(t);
        rec(out, tag ++ "E:{s}", tag_args ++ .{t});
        return;
    };
    const rt = resultText(result);
    defer gpa.free(rt);
    rec(out, tag ++ "{s}", tag_args ++ .{rt});
    const ad = guardText(ctx.ast.dump(gpa));
    defer gpa.free(ad);
    rec(out, "A:{s}", .{ad});
    const vd = guardText(astInvalidText(&ctx.ast));
    defer gpa.free(vd);
    rec(out, "V:{s}", .{vd});
    inline for (.{ cpeg.Mapper.Kind.base, .gemma4, .minimax_m3 }, 0..) |kind, k| {
        if (mapperText(&ctx, &result, kind)) |s| {
            defer gpa.free(s);
            rec(out, "M{d}:G:{s}", .{ k, s });
        } else |e| {
            const t = errText(e);
            defer gpa.free(t);
            rec(out, "M{d}:E:{s}", .{ k, t });
        }
    }
}

fn tagsText(t: *const cpeg.TaggedParseResult) []u8 {
    var s: std.ArrayList(u8) = .empty;
    const rt = resultText(t.result);
    defer gpa.free(rt);
    s.appendSlice(gpa, rt) catch oom();
    for (t.tags.entries.items) |e| s.print(gpa, "\x1e{s}\x1d{s}", .{ e.key, e.value }) catch oom();
    return s.toOwnedSlice(gpa) catch oom();
}

fn taggedText(tp: *const cpeg.TaggedPegParser, input: []const u8, which: u8) anyerror![]u8 {
    var t = switch (which) {
        0 => try tp.parseAndExtract(gpa, input, peg.Flags.none),
        1 => try tp.parseAndExtract(gpa, input, peg.Flags.lenient),
        else => try tp.parseAnywhereAndExtract(gpa, input),
    };
    defer t.deinit();
    return tagsText(&t);
}

var debug_buf: std.ArrayList(u8) = .empty;

fn captureDebug(text: []const u8) void {
    debug_buf.appendSlice(gpa, text) catch oom();
}

fn grammarText(a: *const peg.Arena, lazy: bool) anyerror![]u8 {
    return peg.gbnf.grammar(gpa, a, lazy, .{});
}

fn dumpRoot(a: *const peg.Arena) anyerror![]u8 {
    return a.dump(gpa, a.root);
}

/// One case through the port: the same records `run_case` writes.
fn runPortCase(kind: u8, text: []const u8, inputs: []const Input, out: *Records) void {
    var arena_opt: ?peg.Arena = null;
    defer if (arena_opt) |*a| a.deinit();
    _ = kind;
    if (portBuild(text)) |a| {
        arena_opt = a;
        rec(out, "B:ok", .{});
    } else |e| {
        const t = errText(e);
        defer gpa.free(t);
        rec(out, "B:E:{s}", .{t});
    }
    defer rec(out, "\x01", .{});
    const a = &(arena_opt orelse return);
    rec(out, "N:{d}", .{a.size()});
    const d = guardText(dumpRoot(a));
    defer gpa.free(d);
    rec(out, "D:{s}", .{d});
    const saved = guardText(peg_serialize.save(gpa, a));
    defer gpa.free(saved);
    rec(out, "S:{s}", .{saved});
    const g = guardText(grammarText(a, false));
    defer gpa.free(g);
    rec(out, "G:{s}", .{g});
    const l = guardText(grammarText(a, true));
    defer gpa.free(l);
    rec(out, "L:{s}", .{l});
    for (inputs) |in| {
        rec(out, "I:{s}", .{in.text});
        if (in.modes & MODE_FULL != 0) parseRecords(a, in.text, peg.Flags.none, "P0:", .{}, out);
        if (in.modes & MODE_LENIENT != 0) parseRecords(a, in.text, peg.Flags.lenient, "P1:", .{}, out);
        if (in.modes & MODE_LENIENT != 0) parseRecords(a, in.text, peg.Flags.lenient | peg.Flags.complete, "P2:", .{}, out);
        if (in.modes & MODE_PREFIXES != 0) {
            const stride: usize = if (in.text.len <= 160) 1 else in.text.len / 120;
            var k: usize = 0;
            while (k < in.text.len) : (k += stride) parseRecords(a, in.text[0..k], peg.Flags.lenient, "Q{d}:", .{k}, out);
        }
        if (in.modes & MODE_DEBUG != 0) {
            debug_buf.clearRetainingCapacity();
            const prev = peg.errors.stderr_sink;
            peg.errors.stderr_sink = captureDebug;
            defer peg.errors.stderr_sink = prev;
            var ctx = peg.ParseContext.init(gpa, in.text, peg.Flags.lenient | peg.Flags.debug) catch oom();
            defer ctx.deinit();
            const res = if (a.parse(&ctx, 0)) |r| resultText(r) else |e| blk: {
                const t = errText(e);
                defer gpa.free(t);
                break :blk std.fmt.allocPrint(gpa, "E:{s}", .{t}) catch oom();
            };
            defer gpa.free(res);
            rec(out, "X:{s}\n{s}", .{ res, debug_buf.items });
        }
        if (in.modes & MODE_TAGGED != 0) {
            const tp: cpeg.TaggedPegParser = .{ .arena = a.* };
            inline for (0..3) |w| {
                const t = guardText(taggedText(&tp, in.text, w));
                defer gpa.free(t);
                rec(out, "T{d}:{s}", .{ w, t });
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Cases

const Input = struct { text: []const u8, modes: u8 };

const Case = struct {
    label: []const u8,
    kind: u8,
    /// the description, or the JSON a kind-3 case loads
    text: []const u8,
    inputs: []const Input,
};

var cases: std.ArrayList(Case) = .empty;

fn caseProgram(c: Case, id: u32, prog: *std.ArrayList(u8)) void {
    prog.appendSlice(gpa, std.mem.asBytes(&id)) catch oom();
    prog.append(gpa, c.kind) catch oom();
    const n: u32 = @intCast(c.text.len);
    prog.appendSlice(gpa, std.mem.asBytes(&n)) catch oom();
    prog.appendSlice(gpa, c.text) catch oom();
    const k: u32 = @intCast(c.inputs.len);
    prog.appendSlice(gpa, std.mem.asBytes(&k)) catch oom();
    for (c.inputs) |in| {
        const m: u32 = @intCast(in.text.len);
        prog.appendSlice(gpa, std.mem.asBytes(&m)) catch oom();
        prog.appendSlice(gpa, in.text) catch oom();
        prog.append(gpa, in.modes) catch oom();
    }
}

// -----------------------------------------------------------------------------
// The reference, in forked children

fn splitRecords(buf: []const u8, out: *Records) void {
    var i: usize = 0;
    while (i < buf.len) {
        const n = std.mem.readInt(u32, buf[i..][0..4], .little);
        out.append(gpa, gpa.dupe(u8, buf[i + 4 ..][0..n]) catch oom()) catch oom();
        i += 4 + n;
    }
}

/// Runs `prog` -- or, given null, upstream's tests -- through the reference
/// in a child, killed after `seconds`: a PEG without memoization backtracks
/// exponentially on some grammars, upstream and port alike. Return: whether
/// the child finished; its output is in `out`.
fn runForked(prog: ?[]const u8, out: *std.ArrayList(u8), seconds: c_uint) bool {
    if (comptime builtin.os.tag == .windows)
        return win_child.run(if (prog == null) "upstream" else "run", prog orelse "", out, seconds);
    var fds: [2]c_int = undefined;
    if (pipe(&fds) != 0) oom();
    const pid = fork();
    if (pid < 0) oom();
    if (pid == 0) {
        _ = close(fds[0]);
        // the converter's warnings go to stderr, as do the port's; the
        // debug trace is captured apart
        const devnull = open("/dev/null", 1);
        if (devnull >= 0) _ = dup2(devnull, 2);
        _ = alarm(seconds);
        var len: usize = 0;
        const p = if (prog) |pr| ref_peg_run(pr.ptr, pr.len, &len) else ref_peg_upstream(&len);
        var off: usize = 0;
        while (off < len) {
            const k = write(fds[1], p + off, len - off);
            if (k <= 0) _exit(3);
            off += @intCast(k);
        }
        _exit(0);
    }
    _ = close(fds[1]);
    var chunk: [65536]u8 = undefined;
    while (true) {
        const k = read(fds[0], &chunk, chunk.len);
        if (k <= 0) break;
        out.appendSlice(gpa, chunk[0..@intCast(k)]) catch oom();
    }
    _ = close(fds[0]);
    var status: c_int = 0;
    _ = waitpid(pid, &status, 0);
    return status == 0;
}

/// The reference's records for each case of a batch: per case its id, then
/// records until one holding \x01.
fn parseRefBatch(buf: []const u8, by_case: []?Records, first: usize) void {
    var i: usize = 0;
    while (i < buf.len) {
        const id = std.mem.readInt(u32, buf[i..][0..4], .little);
        i += 4;
        var list: Records = .empty;
        while (true) {
            const n = std.mem.readInt(u32, buf[i..][0..4], .little);
            const r = buf[i + 4 ..][0..n];
            i += 4 + n;
            list.append(gpa, gpa.dupe(u8, r) catch oom()) catch oom();
            if (n == 1 and r[0] == 1) break;
        }
        by_case[id - first] = list;
    }
}

// -----------------------------------------------------------------------------
// Generators

var prng: std.Random.DefaultPrng = undefined;

fn below(n: usize) usize {
    return prng.random().uintLessThan(usize, n);
}

fn chance(pct: usize) bool {
    return below(100) < pct;
}

fn pick(comptime T: type, xs: []const T) T {
    return xs[below(xs.len)];
}

const literal_pool = [_][]const u8{
    "\x0b",             "\x0c\x0b ",    "x\\\"{",
    "a",                "ab",           "abc",
    "<tool_call>",      "</tool_call>", "{",
    "}",                "\"",           ":",
    ",",                "[",            "]",
    "<think>",          "</think>",     " ",
    "\n",               "\xc3\xa9",     "\xe6\x97\xa5\xe6\x9c\xac",
    "\xf0\x9f\x98\x80", "x\xff",        "",
    "True",             "null",         "<",
    ">",                "\\",           "'",
    "=",                "(",            ")",
    "<function=",       "</function>",  "<param=",
    "</param>",         "hello",        "\t",
    "-",                "0",            "1.5",
    "\x00z",            "<|tool_call>", "<tool_call|>",
    "call:",            "<|\"|>",       "\r\n",
};

const class_pool = [_][]const u8{
    "[a-z]",                     "a-z",       "[^a-z]",  "[0-9]",   "[\\n\\t ]",    "[\\x41-\\x5A]", "[\\u00e9-\\u00ff]",
    "[\\U0001F600-\\U0001F64F]", "[^\"\\\\]", "[a\\-z]", "[\\]]",   "[^]",          "[]",            "[-a]",
    "[a-]",                      "^",         "[\\q]",   "[\\xZZ]", "[\xc3\xa9]",   "[\xc3]",        "[^\\n]",
    "[a-z-]",                    "[\\[\\]]",  "[\\r]",   "[z-a]",   "[\\u12]",      "[^:}]",         "[0-9.eE+-]",
    "a",                         "[\\",       "[\\\\]",  "[\\x]",   "[\\\xc3\xa9]", "[^\\\xc3\xa9]",
};

const name_pool = [_][]const u8{
    "a",             "b",     "value",    "x y",     "r!",   "tool-call", "json-value",
    "gemma4-string", "",      "\xc3\x9c", "rule_1",  "a.b",  "number",    "list",
    "content",       "space", "root",     "until-3", "ac-2", "x-y",       "tool-get",
};

const tag_pool = [_][]const u8{
    "reasoning",       "content",        "tool",          "tool-open",      "tool-close",    "tool-id",        "tool-name",
    "tool-args",       "tool-arg",       "tool-arg-open", "tool-arg-close", "tool-arg-name", "tool-arg-value", "tool-arg-string-value",
    "tool-arg-object", "tool-arg-array", "tool-arg-item", "t",              "x",             "",               "reasoning-block",
};

const schema_pool = [_][]const u8{
    "{\"type\":\"string\"}",
    "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"},\"b\":{\"type\":\"string\"}},\"required\":[\"a\"]}",
    "{\"type\":\"array\",\"items\":{\"type\":\"number\"}}",
    "{\"enum\":[\"x\",\"y\",1]}",
    "{\"type\":\"string\",\"pattern\":\"^[a-z]+$\"}",
    "{}",
    "{\"$ref\":\"#/nope\"}",
    "{\"type\":5}",
    "{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"null\"}]}",
    "{\"type\":\"integer\",\"minimum\":3,\"maximum\":12}",
    "{\"type\":\"string\",\"pattern\":\"(\"}",
    "{\"type\":[\"string\",\"integer\"]}",
    "true",
    "[]",
    "{\"const\":\"k\"}",
};

const grammar_pool = [_][]const u8{ "", "[a-z]+", "\"x\" | \"y\"", "space", "(" };

fn strPool(comptime pool: []const []const u8) []const u8 {
    return pick([]const u8, pool);
}

fn randomStr() []const u8 {
    if (chance(85)) return strPool(&literal_pool);
    const n = 1 + below(6);
    const out = gpa.alloc(u8, n) catch oom();
    for (out) |*c| c.* = if (chance(70)) pick(u8, "abc<>/\"{}[]: \n\\") else @intCast(below(256));
    return out;
}

fn nameStr() []const u8 {
    return strPool(&name_pool);
}

const MinMax = struct { i32, i32 };
const minmax_pool = [_]MinMax{ .{ 1, -1 }, .{ 0, -1 }, .{ 1, 1 }, .{ 0, 1 }, .{ 2, 4 }, .{ 3, 3 }, .{ 1, 2 }, .{ 0, 0 }, .{ 2, 1 }, .{ 2, -1 }, .{ 5, 5 } };

/// A random JSON tools array, sometimes malformed.
fn genTools() []const u8 {
    var t: std.ArrayList(u8) = .empty;
    const fnames = [_][]const u8{ "get_weather", "search", "special_function", "special_function_with_opt", "\xe6\x97\xa5\xe6\x9c\xac", "a b", "x" };
    const pnames = [_][]const u8{ "location", "unit", "days", "query", "q", "a", "b", "\xc3\xa9" };
    const ptypes = [_][]const u8{ "string", "integer", "number", "object", "array", "boolean" };
    if (chance(5)) return pick([]const u8, &.{ "{}", "5", "[]", "null", "[1,{}]" });
    t.appendSlice(gpa, "[") catch oom();
    const n = 1 + below(3);
    for (0..n) |i| {
        if (i > 0) t.appendSlice(gpa, ",") catch oom();
        if (chance(4)) {
            t.appendSlice(gpa, pick([]const u8, &.{ "{\"type\":\"function\"}", "7", "{\"function\":5}", "{\"function\":{\"name\":3}}", "{\"function\":{}}" })) catch oom();
            continue;
        }
        t.print(gpa, "{{\"type\":\"function\",\"function\":{{\"name\":\"{s}\"", .{fnames[below(fnames.len)]}) catch oom();
        if (chance(85)) {
            t.appendSlice(gpa, ",\"parameters\":") catch oom();
            if (chance(8)) {
                t.appendSlice(gpa, pick([]const u8, &.{ "null", "{}", "[1]", "\"s\"", "{\"properties\":[\"a\",\"b\"]}", "{\"properties\":5}" })) catch oom();
            } else {
                t.appendSlice(gpa, "{\"type\":\"object\",\"properties\":{") catch oom();
                const k = below(4);
                for (0..k) |j| {
                    if (j > 0) t.appendSlice(gpa, ",") catch oom();
                    t.print(gpa, "\"{s}{d}\":{{\"type\":\"{s}\"}}", .{ pnames[below(pnames.len)], j, ptypes[below(ptypes.len)] }) catch oom();
                }
                t.appendSlice(gpa, "}") catch oom();
                if (k > 0 and chance(60)) t.print(gpa, ",\"required\":[\"{s}0\"]", .{pnames[0]}) catch oom();
                t.appendSlice(gpa, "}") catch oom();
            }
        }
        t.appendSlice(gpa, "}}") catch oom();
    }
    t.appendSlice(gpa, "]") catch oom();
    return t.items;
}

/// A random parser expression, written into `d`; its slot.
fn genExpr(d: *Desc, depth: usize, avail: *std.ArrayList(u32)) u32 {
    const slot = genExprInner(d, depth, avail);
    avail.append(gpa, slot) catch oom();
    return slot;
}

fn sub(d: *Desc, depth: usize, avail: *std.ArrayList(u32)) u32 {
    if (avail.items.len > 0 and chance(20)) return pick(u32, avail.items);
    return genExpr(d, depth + 1, avail);
}

fn genExprInner(d: *Desc, depth: usize, avail: *std.ArrayList(u32)) u32 {
    const leafy = depth > 4 or chance(30);
    if (leafy) {
        return switch (below(24)) {
            0, 1, 2, 3, 4 => d.lit(randomStr()),
            5, 6 => blk: {
                const mm = pick(MinMax, &minmax_pool);
                break :blk d.chars(strPool(&class_pool), mm[0], mm[1]);
            },
            7 => d.leaf(.any),
            8 => d.leaf(.space),
            9 => d.leaf(.eps),
            10 => d.leaf(pick(Op, &.{ .start, .end })),
            11, 12 => d.strOp(.until, randomStr()),
            13 => blk: {
                const w = d.op(.until_one_of);
                const k = below(4);
                const v = gpa.alloc([]const u8, k) catch oom();
                for (v) |*s| s.* = randomStr();
                d.strs(v);
                break :blk w;
            },
            14 => d.leaf(.rest),
            15 => blk: {
                const w = d.op(.string_content);
                d.byte(pick(u8, "\"'x\xc3"));
                break :blk w;
            },
            16 => d.strOp(.ref, nameStr()),
            17 => d.leaf(pick(Op, &.{ .json, .json_object, .json_string, .json_array, .json_number, .json_bool, .json_null })),
            18 => d.leaf(pick(Op, &.{ .py_value, .py_dict, .py_string, .py_array, .py_number, .py_bool, .py_null })),
            19 => d.leaf(pick(Op, &.{ .dq_string, .sq_string, .quoted_string, .marker })),
            20 => blk: {
                const w = d.op(.prefix);
                d.str(randomStr());
                d.str(if (chance(50)) "" else randomStr());
                break :blk w;
            },
            21 => d.strOp(.optspace, pick([]const u8, &.{ "  <a> ", "<b>", "   ", "", "\n<x>\t", " x\xc3\xa9 " })),
            22 => d.leaf(pick(Op, &.{ .seq_empty, .choice_empty })),
            else => d.lit(randomStr()),
        };
    }
    switch (below(30)) {
        0, 1, 2, 3 => {
            const k = 1 + below(4);
            var v: [4]u32 = undefined;
            for (v[0..k]) |*x| x.* = sub(d, depth, avail);
            return if (chance(60)) d.seq(v[0..k]) else d.choice(v[0..k]);
        },
        4, 5 => {
            const a = sub(d, depth, avail);
            const b = sub(d, depth, avail);
            return d.bin(pick(Op, &.{ .op_plus, .op_shl, .op_or, .op_plus_assign, .op_or_assign }), a, b);
        },
        6 => {
            const a = sub(d, depth, avail);
            const s = randomStr();
            if (chance(50)) {
                const w = d.op(pick(Op, &.{ .op_plus_lit, .op_shl_lit, .op_or_lit }));
                d.u32_(a);
                d.str(s);
                return w;
            }
            const w = d.op(pick(Op, &.{ .op_lit_plus, .op_lit_shl, .op_lit_or }));
            d.str(s);
            d.u32_(a);
            return w;
        },
        7, 8 => return d.un(pick(Op, &.{ .one_or_more, .zero_or_more, .optional, .peek, .negate, .atomic }), sub(d, depth, avail)),
        9 => {
            const a = sub(d, depth, avail);
            const mm = pick(MinMax, &minmax_pool);
            if (chance(30)) {
                const w = d.op(.repeat_n);
                d.u32_(a);
                d.i32_(mm[0]);
                return w;
            }
            return d.repeat(a, mm[0], mm[1]);
        },
        10, 11 => return d.tag(strPool(&tag_pool), sub(d, depth, avail)),
        12 => return d.un(pick(Op, &.{ .reasoning_block, .reasoning, .content, .tool, .tool_open, .tool_close, .tool_id, .tool_name, .tool_args, .tool_arg, .tool_arg_open, .tool_arg_close, .tool_arg_name, .tool_arg_value, .tool_arg_string_value, .tool_arg_json_value }), sub(d, depth, avail)),
        13, 14 => return d.rule(nameStr(), sub(d, depth, avail), chance(30)),
        15, 16 => {
            const trigger_api = chance(25);
            const w = d.op(if (trigger_api) .trigger_rule_fn else .rule_fn);
            d.str(nameStr());
            if (!trigger_api) d.byte(@intFromBool(chance(20)));
            const inner = sub(d, depth, avail);
            d.bodyEnd(inner);
            return w;
        },
        17 => {
            const a = sub(d, depth, avail);
            const w = d.op(.gbnf);
            d.u32_(a);
            d.str(strPool(&grammar_pool));
            return w;
        },
        18 => {
            const a = sub(d, depth, avail);
            if (chance(40)) {
                const w = d.op(.ac_one);
                d.u32_(a);
                d.str(randomStr());
                return w;
            }
            const w = d.op(.ac);
            d.u32_(a);
            const k = below(4);
            const v = gpa.alloc([]const u8, k) catch oom();
            for (v) |*s| s.* = randomStr();
            d.strs(v);
            return w;
        },
        19 => {
            const a = sub(d, depth, avail);
            const w = d.op(.schema);
            d.u32_(a);
            d.str(nameStr());
            d.str(strPool(&schema_pool));
            d.byte(@intFromBool(chance(40)));
            return w;
        },
        20 => {
            const a = sub(d, depth, avail);
            const w = d.op(.json_member);
            d.str(randomStr());
            d.u32_(a);
            return w;
        },
        21 => {
            const a = sub(d, depth, avail);
            const w = d.op(.tag_safe);
            d.str(nameStr());
            d.str(if (chance(30)) "" else randomStr());
            d.u32_(a);
            return w;
        },
        22 => {
            const k = below(8);
            var v: [8]u32 = undefined;
            for (v[0..k]) |*x| x.* = sub(d, depth, avail);
            const w = d.op(.permute);
            d.str(nameStr());
            d.slots(v[0..k]);
            return w;
        },
        23, 24, 25 => return genToolsCall(d),
        else => return d.lit(randomStr()),
    }
}

fn genToolsCall(d: *Desc) u32 {
    const tools = genTools();
    switch (below(3)) {
        0 => {
            const w = d.op(.std_json_tools);
            d.str(pick([]const u8, &.{ "<tool_call>", "<tool_call>[", "", "[TOOL_CALLS]" }));
            d.str(pick([]const u8, &.{ "</tool_call>", "]</tool_call>", "", "\n" }));
            d.str(tools);
            d.byte(@intFromBool(chance(50)));
            d.byte(@intFromBool(chance(50)));
            d.str(pick([]const u8, &.{ "", "name", "function.name", "tool_name", "a.b.c" }));
            d.str(pick([]const u8, &.{ "", "arguments", "function.arguments", "parameters", "args" }));
            d.byte(@intFromBool(chance(30)));
            d.byte(@intFromBool(chance(25)));
            d.str(pick([]const u8, &.{ "", "", "id", "tool_call_id", "x.id" }));
            d.str(pick([]const u8, &.{ "", "", "tool_call_id", "call_id" }));
            const orders = [_][]const []const u8{ &.{}, &.{ "id", "name", "arguments" }, &.{ "arguments", "name" }, &.{"tool_call_id"}, &.{ "name", "x", "arguments" } };
            d.strs(orders[below(orders.len)]);
            d.byte(@intFromBool(chance(30)));
            return w;
        },
        1 => {
            const w = d.op(.std_constructed_tools);
            const keys = [_][]const u8{ "tool_call_start_marker", "tool_call_end_marker", "function_opener", "function_name_suffix", "function_closer", "parameter_key_prefix", "parameter_key_suffix", "parameter_closer", "other" };
            const vals = [_][]const u8{ "<tc>", "</tc>", "<f=", ">", "</f>", "<p=", "=", "</p>", "\n", "", "[", "]" };
            const k = below(5);
            d.u32_(@intCast(k));
            for (0..k) |_| {
                d.str(keys[below(keys.len)]);
                d.str(vals[below(vals.len)]);
            }
            d.str(tools);
            d.byte(@intFromBool(chance(50)));
            d.byte(@intFromBool(chance(50)));
            return w;
        },
        else => {
            const w = d.op(.py_style_tools);
            d.str(tools);
            d.byte(@intFromBool(chance(50)));
            d.byte(@intFromBool(chance(50)));
            return w;
        },
    }
}

/// A random description: expressions, a root.
fn genRandom() []const u8 {
    var d: Desc = .{};
    var avail: std.ArrayList(u32) = .empty;
    defer avail.deinit(gpa);
    const n = 1 + below(3);
    var last: u32 = 0;
    for (0..n) |_| last = genExpr(&d, 0, &avail);
    if (chance(40)) last = d.seq(&.{ last, d.leaf(.end) });
    // now and then no root at all: a parse throws, `dump` of the root reads
    // past the parsers -- the arms the loaded arenas reached until v0.7.0
    // removed `load`
    if (!chance(2)) d.setRoot(last);
    return d.buf.items;
}

/// A chat grammar in the shape upstream's templates take: optional
/// reasoning, content up to the tools, the tool section, the end.
fn genChat() []const u8 {
    var d: Desc = .{};
    const think_open = pick([]const u8, &.{ "<think>", "<|START_THINKING|>", "[THINK]" });
    const think_close = pick([]const u8, &.{ "</think>", "<|END_THINKING|>", "[/THINK]" });
    var reasoning = d.leaf(.eps);
    if (chance(70)) {
        const r = d.un(.reasoning, d.strOp(.until, think_close));
        const a = d.litPlus(think_open, r);
        const b = d.plusLit(a, think_close);
        const c = d.bin(.op_plus, b, d.leaf(.space));
        reasoning = d.un(.optional, c);
    }
    const tools = genToolsCall(&d);
    const tool_start = pick([]const u8, &.{ "<tool_call>", "[TOOL_CALLS]", "<tc>", "[", "<|tool_call>" });
    const content = if (chance(70)) d.un(.content, d.strOp(.until, tool_start)) else d.un(.content, d.leaf(.rest));
    const opt_tools = d.un(.optional, d.bin(.op_plus, d.leaf(.space), tools));
    const root = d.seq(&.{ reasoning, content, opt_tools, d.leaf(.space), d.leaf(.end) });
    d.setRoot(root);
    return d.buf.items;
}

/// The value grammar of `common/parsers/gemma4.cpp`, its rules named as
/// the gemma4 mapper reads them.
fn genGemma4() []const u8 {
    var d: Desc = .{};
    _ = d.rule("gemma4-string-content", d.strOp(.until, "<|\"|>"), false);
    _ = d.rule("gemma4-string", d.bin(.op_plus, d.bin(.op_plus, d.lit("<|\"|>"), d.strOp(.ref, "gemma4-string-content")), d.lit("<|\"|>")), false);
    _ = d.rule("gemma4-bool", d.leaf(.json_bool), false);
    _ = d.rule("gemma4-null", d.leaf(.json_null), false);
    _ = d.rule("gemma4-number", d.leaf(.json_number), false);
    const key_name = d.rule("gemma4-dict-key-name", d.chars("[^:}]", 1, -1), false);
    _ = d.rule("gemma4-dict-key", d.bin(.op_plus, key_name, d.lit(":")), false);
    _ = d.rule("gemma4-dict-kv", d.bin(.op_plus, d.bin(.op_plus, d.strOp(.ref, "gemma4-dict-key"), d.leaf(.space)), d.strOp(.ref, "gemma4-value")), false);
    {
        const w = d.op(.rule_fn);
        d.str("gemma4-dict");
        d.byte(0);
        const ws = d.leaf(.space);
        const member = d.strOp(.ref, "gemma4-dict-kv");
        const members = d.seq(&.{ member, d.un(.zero_or_more, d.seq(&.{ d.lit(","), ws, member })) });
        const body = d.seq(&.{ d.lit("{"), ws, d.choice(&.{ d.lit("}"), d.seq(&.{ members, ws, d.lit("}") }) }) });
        d.bodyEnd(body);
        _ = w;
    }
    {
        const w = d.op(.rule_fn);
        d.str("gemma4-array");
        d.byte(0);
        const ws = d.leaf(.space);
        const value = d.strOp(.ref, "gemma4-value");
        const elements = d.seq(&.{ value, d.un(.zero_or_more, d.seq(&.{ d.lit(","), ws, value })) });
        const body = d.seq(&.{ d.lit("["), ws, d.choice(&.{ d.lit("]"), d.seq(&.{ elements, ws, d.lit("]") }) }) });
        d.bodyEnd(body);
        _ = w;
    }
    {
        const w = d.op(.rule_fn);
        d.str("gemma4-value");
        d.byte(0);
        const body = d.choice(&.{ d.strOp(.ref, "gemma4-string"), d.strOp(.ref, "gemma4-dict"), d.strOp(.ref, "gemma4-array"), d.strOp(.ref, "gemma4-number"), d.strOp(.ref, "gemma4-bool"), d.strOp(.ref, "gemma4-null") });
        d.bodyEnd(body);
        _ = w;
    }
    var tool_choice = d.leaf(.choice_empty);
    for ([_][]const u8{ "get_weather", "search" }) |name| {
        const opn = d.un(.tool_open, d.bin(.op_plus, d.un(.tool_name, d.lit(name)), d.un(.peek, d.lit("{"))));
        const t = d.un(.tool, d.seq(&.{ opn, d.un(.tool_args, d.strOp(.ref, "gemma4-dict")) }));
        tool_choice = d.bin(.op_or_assign, tool_choice, d.rule(std.fmt.allocPrint(gpa, "tool-{s}", .{name}) catch oom(), t, false));
    }
    const call = d.plusLit(d.litPlus("<|tool_call>call:", tool_choice), "<tool_call|>");
    const tool_call = blk: {
        const rep = d.repeat(call, 0, if (chance(50)) -1 else 1);
        const w = d.op(.trigger_rule);
        d.str("tool-call");
        d.u32_(rep);
        break :blk w;
    };
    const content = d.rule("content", d.un(.content, blk: {
        const w = d.op(.until_one_of);
        d.strs(&.{ "<|channel>", "<|tool_call>" });
        break :blk w;
    }), false);
    const thought = d.un(.optional, d.seq(&.{ d.lit("<|channel>thought"), d.leaf(.space), d.un(.reasoning, d.strOp(.until, "<channel|>")), d.lit("<channel|>") }));
    const root = d.seq(&.{ thought, content, tool_call, d.leaf(.end) });
    d.setRoot(root);
    return d.buf.items;
}

/// Tool calls in the shape `common/parsers/minimax-m3.cpp` builds, its
/// object, array and item tags around the argument values.
fn genMinimax() []const u8 {
    var d: Desc = .{};
    const NS = "mm:";
    const close_s = NS ++ "</v>";
    const str_val = blk: {
        const v = d.un(.tool_arg_string_value, d.strOp(.until, close_s));
        const body = d.bin(.op_plus, v, d.un(.tool_arg_close, d.lit(close_s)));
        const w = d.op(.ac_one);
        d.u32_(body);
        d.str(close_s);
        break :blk w;
    };
    const num_val = d.bin(.op_plus, d.un(.tool_arg_value, d.leaf(.json_number)), d.un(.tool_arg_close, d.lit(close_s)));
    const elem = d.un(.tool_arg, d.bin(.op_plus, d.un(.tool_arg_open, d.bin(.op_plus, d.bin(.op_plus, d.lit(NS ++ "<"), d.un(.tool_arg_name, d.chars("[a-z]", 1, -1))), d.lit(">"))), d.choice(&.{ num_val, str_val })));
    const item = d.tag("tool-arg-item", d.bin(.op_plus, d.lit("<item>"), d.choice(&.{ num_val, str_val })));
    const arr = d.tag("tool-arg-array", d.un(.one_or_more, item));
    const obj = d.tag("tool-arg-object", d.un(.one_or_more, elem));
    const nested = d.un(.tool_arg, d.bin(.op_plus, d.un(.tool_arg_open, d.bin(.op_plus, d.bin(.op_plus, d.lit(NS ++ "<"), d.un(.tool_arg_name, d.lit("o"))), d.lit(">"))), d.choice(&.{ obj, arr })));
    const args = d.un(.zero_or_more, d.choice(&.{ nested, elem }));
    const t = d.un(.tool, d.seq(&.{ d.un(.tool_open, d.bin(.op_plus, d.lit("<invoke name=\""), d.un(.tool_name, d.chars("[a-z_]", 1, -1)))), d.lit("\">"), args, d.un(.tool_close, d.lit("</invoke>")) }));
    const reasoning = d.un(.optional, d.seq(&.{ d.lit("<think>"), d.un(.reasoning, d.strOp(.until, "</think>")), d.lit("</think>") }));
    const root = d.seq(&.{ reasoning, d.un(.content, d.strOp(.until, "<invoke")), d.un(.zero_or_more, t), d.leaf(.end) });
    d.setRoot(root);
    return d.buf.items;
}

// -- inputs derived from a grammar

const any_pool = [_][]const u8{ "a", "z", "Q", "0", " ", "\xc3\xa9", "\xc3\x83", "\xe6\x97\xa5", "\xf0\x9f\x98\x80", "\"", "}", "<", "\n", "\x0b" };

fn derive(arena: *const peg.Arena, id: ParserId, depth: usize, out: *std.ArrayList(u8)) void {
    if (depth > 30 or out.items.len > 300) return;
    const p = arena.get(id) catch return;
    switch (p.*) {
        .epsilon, .start, .end, .and_, .not_ => {},
        .literal => |l| out.appendSlice(gpa, l) catch oom(),
        .sequence => |c| for (c) |x| derive(arena, x, depth + 1, out),
        .choice => |c| if (c.len > 0) derive(arena, c[if (depth > 12) 0 else below(c.len)], depth + 1, out),
        .repetition => |r| {
            const min: usize = @intCast(@max(r.min_count, 0));
            const extra: usize = if (depth > 12) 0 else below(3);
            var k = min + extra;
            if (r.max_count >= 0) k = @min(k, @as(usize, @intCast(r.max_count)));
            for (0..k) |_| derive(arena, r.child, depth + 1, out);
        },
        .any => out.appendSlice(gpa, pick([]const u8, &any_pool)) catch oom(),
        .space => out.appendSlice(gpa, pick([]const u8, &.{ "", " ", "\n", " \t ", "\x0b\x0c\r" })) catch oom(),
        .chars => |c| {
            const min: usize = @intCast(@max(c.min_count, 0));
            var k = min + below(3);
            if (c.max_count >= 0) k = @min(k, @as(usize, @intCast(c.max_count)));
            for (0..k) |_| {
                var cp: u32 = 'a';
                if (!c.negated and c.ranges.len > 0) {
                    const r = c.ranges[below(c.ranges.len)];
                    if (r.end >= r.start and r.end - r.start < 0x110000) cp = r.start + @as(u32, @intCast(below(@as(usize, r.end - r.start) + 1)));
                } else {
                    cp = pick(u32, &.{ 'q', 'Z', '7', '_', 0xe9 });
                }
                var buf: [4]u8 = undefined;
                const s = json.trie.cptToUtf8(cp, &buf) catch continue;
                out.appendSlice(gpa, s) catch oom();
            }
        },
        .string => |dl| {
            const parts = [_][]const u8{ "abc", "\\n", "\\\"", "\\u00e9", "\xc3\xa9", "x y", "\\\\", "'", "\\\"{", "}\\\"" };
            for (0..below(3)) |_| {
                const s = pick([]const u8, &parts);
                if (std.mem.indexOfScalar(u8, s, dl) == null) out.appendSlice(gpa, s) catch oom();
            }
        },
        .until => |ds| {
            const parts = [_][]const u8{ "hello", "x", " ", "\xe6\x97\xa5\xe6\x9c\xac", "a\nb", "{\"k\": 1}", "{\"k\": \"\\\"{\"", "\x0b" };
            const s = pick([]const u8, &parts);
            var ok = true;
            for (ds) |dl| if (dl.len > 0 and std.mem.indexOf(u8, s, dl) != null) {
                ok = false;
            };
            if (ok) out.appendSlice(gpa, s) catch oom();
        },
        .schema => |s| derive(arena, s.child, depth + 1, out),
        .rule => |r| derive(arena, r.child, depth + 1, out),
        .ref => |name| if (arena.getRule(name)) |rid| derive(arena, rid, depth + 1, out) else |_| {},
        .atomic => |c| derive(arena, c, depth + 1, out),
        .tag => |t| derive(arena, t.child, depth + 1, out),
        .gbnf => |g| derive(arena, g.child, depth + 1, out),
        .ac => |a| derive(arena, a.child, depth + 1, out),
    }
}

fn mutateInput(s: []const u8) []const u8 {
    var t: std.ArrayList(u8) = .empty;
    t.appendSlice(gpa, s) catch oom();
    for (0..1 + below(2)) |_| {
        switch (below(5)) {
            0 => if (t.items.len > 0) {
                const i = below(t.items.len);
                const n = @min(1 + below(4), t.items.len - i);
                t.replaceRange(gpa, i, n, &.{}) catch oom();
            },
            1 => t.insertSlice(gpa, below(t.items.len + 1), randomStr()) catch oom(),
            2 => if (t.items.len > 0) {
                t.items[below(t.items.len)] = pick(u8, "\"{}[],: \\<>/a0\xff\xc3\x0b");
            },
            3 => t.shrinkRetainingCapacity(below(t.items.len + 1)),
            else => if (t.items.len > 1) {
                const i = below(t.items.len);
                const n = @min(1 + below(8), t.items.len - i);
                const span = gpa.dupe(u8, t.items[i .. i + n]) catch oom();
                t.insertSlice(gpa, i, span) catch oom();
            },
        }
    }
    return t.items;
}

fn modesFor(text: []const u8, arena_size: usize) u8 {
    var m: u8 = MODE_FULL | MODE_LENIENT | MODE_PREFIXES;
    if (text.len <= 64) m |= MODE_TAGGED;
    if (text.len <= 40 and arena_size <= 150) m |= MODE_DEBUG;
    return m;
}

/// The inputs for a description: the given ones, strings derived from the
/// port's build of it, and both mutated.
fn inputsFor(ops: []const u8, given: []const []const u8, n_derived: usize, n_mutated: usize) []Input {
    var list: std.ArrayList(Input) = .empty;
    var arena_opt: ?peg.Arena = portBuild(ops) catch blk: {
        if (harness_msg) |m| {
            gpa.free(m);
            harness_msg = null;
        }
        break :blk null;
    };
    defer if (arena_opt) |*a| a.deinit();
    const size = if (arena_opt) |a| a.size() else 0;
    var texts: std.ArrayList([]const u8) = .empty;
    for (given) |g| texts.append(gpa, g) catch oom();
    if (arena_opt) |*a| {
        if (a.root != peg.invalid_parser_id) {
            for (0..n_derived) |_| {
                var out: std.ArrayList(u8) = .empty;
                derive(a, a.root, 0, &out);
                texts.append(gpa, out.items) catch oom();
            }
        }
    }
    const base = texts.items.len;
    for (0..n_mutated) |_| {
        const src = if (base > 0) texts.items[below(base)] else randomStr();
        texts.append(gpa, mutateInput(src)) catch oom();
    }
    if (texts.items.len == 0) texts.append(gpa, randomStr()) catch oom();
    for (texts.items) |t| list.append(gpa, .{ .text = t, .modes = modesFor(t, size) }) catch oom();
    return list.items;
}

// -----------------------------------------------------------------------------
// Coverage: every arm must appear in some reference record

const required_phrases = [_][]const u8{
    "input is nested more than 64 levels deep",
    "B:E:RE:Rule not found: ",
    "RE:ac parser requires at least one delimiter",
    "RE:No root parser set",
    "OR:vector",
    "JE:[json.exception.type_error.316]",
    "Epsilon",
    "Start",
    "End",
    "Literal(",
    "Sequence(",
    "Choice(",
    "Repetition(",
    ", unbounded)",
    "And(",
    "Not(",
    "Atomic(",
    "Gbnf(",
    "Ac(",
    "Any",
    "Space",
    "CharRepeat(",
    "String(",
    "Until(",
    "Schema(",
    "Rule(",
    "Tag(",
    "[cycle]",
    "until-",
    "ac-",
    "need_more_input",
    "SEQ -> NEED_MORE",
    "REPEAT -> NEED_MORE (not enough",
    "CHOICE -> FAIL",
    "REPEAT: empty match",
    "TAG: ",
    "REPEAT: at end of input",
    "M0:G:",
    "\x1eget_weather\x1d{",
    "M1:G:",
    "M2:G:",
    "\x1d\"",
    "T2:G:success",
    "\\U0001F600",
    "{2,4}",
};

fn envUsize(name: [*:0]const u8, def: usize) usize {
    const v = getenv(name) orelse return def;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch def;
}

fn printEscaped(s: []const u8) void {
    const max = 600;
    for (s[0..@min(s.len, max)]) |c| {
        if (c == '\n') std.debug.print("\\n", .{}) else if (c < 0x20 or c >= 0x7f) std.debug.print("\\x{x:0>2}", .{c}) else std.debug.print("{c}", .{c});
    }
    if (s.len > max) std.debug.print("...({d} bytes)", .{s.len});
}

fn dropWarning(_: []const u8) void {}

fn addCase(label: []const u8, kind: u8, text: []const u8, inputs: []const Input) void {
    cases.append(gpa, .{ .label = label, .kind = kind, .text = text, .inputs = inputs }) catch oom();
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    // Windows: this process may be the reference's child (win_child.zig),
    // with stderr dropped as `runForked`'s fork drops it
    if (comptime builtin.os.tag == .windows) if (win_child.childArgs(argv)) |c| {
        const devnull = open("NUL", 1);
        if (devnull >= 0) _ = dup2(devnull, 2);
        var len: usize = 0;
        const p = if (std.mem.eql(u8, c.mode, "upstream")) ref_peg_upstream(&len) else ref_peg_run(c.input.ptr, c.input.len, &len);
        std.process.exit(if (win_child.writeFile(c.out.ptr, p[0..len])) 0 else 3);
    };
    const filter: ?[]const u8 = if (argv.len > 1) argv[1] else null;

    const seed = envUsize("PEG_DIFF_SEED", 1);
    const scale = envUsize("PEG_DIFF_SCALE", 1);
    prng = std.Random.DefaultPrng.init(seed);
    json.schema_to_grammar.warning_sink = dropWarning;

    // -------------------------------------------------------------------------
    // upstream's parsers and inputs, from its own tests

    {
        var buf: std.ArrayList(u8) = .empty;
        if (!runForked(null, &buf, 120)) {
            std.debug.print("FAIL: upstream's PEG tests crashed under the recorder\n", .{});
            std.process.exit(1);
        }
        var r: Reader = .{ .buf = buf.items };
        const fails = r.str();
        const mismatch = r.str();
        if (!std.mem.eql(u8, fails, "0 0")) {
            std.debug.print("FAIL: upstream's PEG tests fail under the recorder (failures: {s})\n", .{fails});
            std.process.exit(1);
        }
        if (mismatch.len != 0) {
            std.debug.print("FAIL: a recorded description does not rebuild the test's arena: {s}\n", .{mismatch});
            std.process.exit(1);
        }
        const n = r.u32_();
        var n_inputs: usize = 0;
        for (0..n) |_| {
            const kind = r.str()[0];
            const ops = r.str();
            const k = r.u32_();
            const given = gpa.alloc([]const u8, k) catch oom();
            for (given) |*s| s.* = r.str();
            n_inputs += k;
            addCase("upstream", kind, ops, inputsFor(ops, given, 4 * scale, 6 * scale));
        }
        std.debug.print("upstream: {d} parsers and {d} inputs recorded from test-peg-parser and test-chat-peg-parser, both passing\n", .{ n, n_inputs });
    }
    const n_upstream = cases.items.len;

    // -------------------------------------------------------------------------
    // generated

    for (0..3 * scale) |_| {
        const ops = genGemma4();
        addCase("gemma4", 0, ops, inputsFor(ops, &.{}, 40, 40));
        const mm = genMinimax();
        addCase("minimax", 0, mm, inputsFor(mm, &.{}, 40, 40));
    }
    // the base mapper's state machine, node orders a grammar rarely
    // produces: a close before any name (and a tool node after it,
    // undefined in the C++ until v0.6.0), arguments before the name, a string value left open
    {
        const orders = [_][]const Op{
            &.{ .tool_open, .tool_close, .tool_name, .tool_arg_name, .tool_arg_string_value },
            &.{ .tool_open, .tool_args, .tool_name, .tool_close },
            &.{ .tool_open, .tool_arg_open, .tool_arg_name, .tool_arg_string_value, .tool_close },
            &.{ .tool_open, .tool_arg_name, .tool_arg_value, .tool_arg_close, .tool_name, .tool_close, .tool_id },
            &.{ .tool_open, .tool_id, .tool_open, .tool_name, .tool_arg_string_value, .tool_arg_string_value, .tool_arg_close },
            &.{ .tool_name, .tool_open, .tool_arg_value, .tool_close, .tool_close },
            // a named call kept, then a nameless one closed and a tool node
            // after it: v0.6.0 clears current_tool, so nothing reaches the first
            &.{ .tool_open, .tool_name, .tool_close, .tool_open, .tool_close, .tool_name, .tool_arg_name, .tool_arg_string_value },
            &.{ .tool_open, .tool_name, .tool_args, .tool_close, .tool_open, .tool_close, .tool_args, .tool_id },
        };
        const texts = [_][]const u8{ "{\"a\": 1}", "x", "\"q\" ", "", "'s'", "{'k': True}", "N", "\xc3\xa9", "{\"a\": \"\\\"{\"", "\"\\\"{" };
        for (orders) |order| {
            for (0..8) |_| {
                var d: Desc = .{};
                var parts: [8]u32 = undefined;
                for (order, 0..) |code, i| {
                    const t = texts[below(texts.len)];
                    parts[i] = d.un(code, d.lit(t));
                    _ = &parts;
                }
                d.setRoot(d.seq(parts[0..order.len]));
                // the input is the literals in order: derived from the grammar
                addCase("mapper", 1, d.buf.items, inputsFor(d.buf.items, &.{}, 2, 2));
            }
        }
    }
    for (0..600 * scale) |_| {
        const ops = genChat();
        addCase("chat", 1, ops, inputsFor(ops, &.{}, 6, 6));
    }
    for (0..3000 * scale) |_| {
        const ops = genRandom();
        addCase("random", 0, ops, inputsFor(ops, &.{}, 3, 3));
    }

    std.debug.print("cases: {d} upstream, {d} generated\n", .{ n_upstream, cases.items.len - n_upstream });

    // -------------------------------------------------------------------------
    // run both sides

    var checks: usize = 0;
    var failures: usize = 0;
    var crashed: usize = 0;
    var n_errors: usize = 0;
    var n_parses: usize = 0;
    var n_grammars: usize = 0;
    var n_tool_calls: usize = 0;
    var per_mapper: [3]usize = @splat(0);
    var phrase_seen: [required_phrases.len]bool = @splat(false);
    var stats: std.StringHashMapUnmanaged(usize) = .empty;
    const want_stats = getenv("PEG_DIFF_STATS") != null;

    const batch = 64;
    var start: usize = 0;
    while (start < cases.items.len) : (start += batch) {
        const end = @min(start + batch, cases.items.len);
        const ref_by_case = gpa.alloc(?Records, end - start) catch oom();
        defer gpa.free(ref_by_case);
        @memset(ref_by_case, null);

        var prog: std.ArrayList(u8) = .empty;
        defer prog.deinit(gpa);
        for (cases.items[start..end], start..) |c, id| {
            if (filter) |f| if (std.mem.indexOf(u8, c.label, f) == null) continue;
            caseProgram(c, @intCast(id), &prog);
        }
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        if (runForked(prog.items, &buf, 60)) {
            parseRefBatch(buf.items, ref_by_case, start);
        } else {
            for (cases.items[start..end], start..) |c, id| {
                if (filter) |f| if (std.mem.indexOf(u8, c.label, f) == null) continue;
                var one: std.ArrayList(u8) = .empty;
                defer one.deinit(gpa);
                caseProgram(c, @intCast(id), &one);
                var out: std.ArrayList(u8) = .empty;
                defer out.deinit(gpa);
                if (runForked(one.items, &out, 10)) {
                    parseRefBatch(out.items, ref_by_case, start);
                } else {
                    crashed += 1;
                    if (crashed <= 3) std.debug.print("note: the reference crashed or ran past 10 s on case {d} ({s}), skipped\n", .{ id, c.label });
                }
            }
        }

        for (cases.items[start..end], 0..) |c, k| {
            const ref = ref_by_case[k] orelse continue;
            var port: Records = .empty;
            runPortCase(c.kind, c.text, c.inputs, &port);
            const n = @max(ref.items.len, port.items.len);
            var input: []const u8 = "";
            for (0..n) |j| {
                const a: []const u8 = if (j < ref.items.len) ref.items[j] else "<missing>";
                const b: []const u8 = if (j < port.items.len) port.items[j] else "<missing>";
                if (std.mem.startsWith(u8, a, "I:")) input = a[2..];
                checks += 1;
                if (std.mem.startsWith(u8, a, "P") or std.mem.startsWith(u8, a, "Q")) n_parses += 1;
                if (std.mem.startsWith(u8, a, "G:G:") or std.mem.startsWith(u8, a, "L:G:")) n_grammars += 1;
                if (a.len > 1 and a[0] == 'M' and std.mem.indexOfScalar(u8, a, 0x1e) != null) {
                    n_tool_calls += 1;
                    if (a[1] >= '0' and a[1] <= '2') per_mapper[a[1] - '0'] += 1;
                }
                if (std.mem.indexOf(u8, a, "E:IA:") != null or std.mem.indexOf(u8, a, "E:RE:") != null or std.mem.indexOf(u8, a, "E:JE:") != null or std.mem.indexOf(u8, a, "E:OR:") != null) {
                    n_errors += 1;
                    if (want_stats) {
                        const at = std.mem.indexOf(u8, a, "E:").?;
                        const key = a[at..@min(a.len, at + 90)];
                        const gop = stats.getOrPut(gpa, gpa.dupe(u8, key) catch oom()) catch oom();
                        if (!gop.found_existing) gop.value_ptr.* = 0;
                        gop.value_ptr.* += 1;
                    }
                }
                for (required_phrases, 0..) |ph, pi| {
                    if (!phrase_seen[pi] and std.mem.indexOf(u8, a, ph) != null) phrase_seen[pi] = true;
                }
                if (!std.mem.eql(u8, a, b)) {
                    failures += 1;
                    if (failures <= 6) {
                        std.debug.print("FAIL case {d} ({s}) record {d}\n  input: ", .{ start + k, c.label, j });
                        printEscaped(input);
                        std.debug.print("\n  ref:   ", .{});
                        printEscaped(a);
                        std.debug.print("\n  port:  ", .{});
                        printEscaped(b);
                        std.debug.print("\n", .{});
                    }
                }
            }
            for (port.items) |x| gpa.free(x);
            port.deinit(gpa);
            for (ref.items) |x| gpa.free(x);
            var refm = ref;
            refm.deinit(gpa);
        }
    }

    if (want_stats) {
        var it = stats.iterator();
        while (it.next()) |e| {
            std.debug.print("{d: >7}  ", .{e.value_ptr.*});
            printEscaped(e.key_ptr.*);
            std.debug.print("\n", .{});
        }
    }

    var missing: usize = 0;
    if (filter == null) {
        for (required_phrases, 0..) |ph, pi| {
            if (!phrase_seen[pi]) {
                std.debug.print("FAIL: no reference record contains \"", .{});
                printEscaped(ph);
                std.debug.print("\": a generator has drifted\n", .{});
                missing += 1;
            }
        }
    }

    for (per_mapper, 0..) |n, k| {
        if (filter == null and n == 0) {
            std.debug.print("FAIL: no reference record of mapper {d} carries a tool call\n", .{k});
            missing += 1;
        }
    }
    if (filter == null and after_close == 0) {
        std.debug.print("FAIL: no mapping saw a tool node after a tool-close\n", .{});
        missing += 1;
    }

    if (failures != 0 or missing != 0) {
        std.debug.print("FAIL: {d} of {d} checks differ over {d} cases\n", .{ failures, checks, cases.items.len });
        std.process.exit(1);
    }
    std.debug.print("PASS: {d} checks over {d} cases ({d} parses, {d} grammars, {d} mapped messages with tool calls -- {d}, {d}, {d} by mapper -- {d} errors; {d} reference crashes skipped; {d} mappings past a tool-close), cli/peg and chat_peg.zig identical to peg-parser.cpp and chat-peg-parser.cpp\n", .{ checks, cases.items.len, n_parses, n_grammars, n_tool_calls, per_mapper[0], per_mapper[1], per_mapper[2], n_errors, crashed, after_close });
}
