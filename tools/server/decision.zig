//! Typed decision models: the `/v1/systemone` API. The model answers each
//! question of a request in one forward pass, no token is generated; this
//! file turns a request into the prompts (one task per question variant, or
//! one for all the questions) and the model outputs back into answers.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/server/server-decision.cpp` and
//! `llama.cpp/tools/server/server-decision.h` (llama.cpp v0.6.0,
//! `d81235049`) whole: `server_decision_question_type`,
//! `server_decision_option`, `server_decision_question`,
//! `server_decision_context` and `server_decision_group_tasks`, with the
//! file's static helpers. Each declaration names what it ports and the line
//! it began at.
//!
//! # Templates
//!
//! The model's `systemone` template renders through
//! `cli/common_chat/template.zig`'s `Render`, which is upstream's
//! `jinja::context` filled by `global_from_json(ctx, inp, false)` and run.
//!
//! # Floating point
//!
//! `format_answer`'s softmax calls the C library's `exp`, as the C++'s
//! `std::exp(double)` does: Zig's own differs in the last bit. Two sums are
//! fused as clang compiles them, both read from the reference's disassembly
//! and held by `server-diff decision`: a score's expected value (scalar,
//! fused) and `decision_confidence_score`'s distance (vectorized eight at a
//! time with rounded products, the tail fused).

const std = @import("std");
const json = @import("json");
const chat = @import("common_chat");
const tc = @import("tools_common");

const common = @import("common.zig");
const tokens_mod = @import("tokens.zig");
const task_mod = @import("task.zig");
const oaicompat = @import("oaicompat.zig");

const c = common.c;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Error = common.Error;
const Obj = common.Obj;
const throw = common.throw;
const jsonValue = common.jsonValue;
const Token = common.Token;
const token_null = common.token_null;
const ServerTokens = tokens_mod.ServerTokens;
const Task = task_mod.Task;
const Template = chat.template.Template;

pub const DecisionType = tc.common.DecisionType;

/// `llama-ext.h`'s `llama_decision_order` (llama-ext.h:107 @d81235049),
/// as the `int32_t` values `server_task::decision::order` holds.
const DecisionOrder = @import("llama").api.batch.llama_decision_order;
fn orderValue(o: DecisionOrder) i32 {
    return @intCast(@backingInt(o));
}

extern "c" fn exp(x: f64) f64;
extern "c" fn strtof(s: [*:0]const u8, end: ?*[*:0]u8) f32;
extern "c" fn strtoul(s: [*:0]const u8, end: ?*[*:0]u8, base: c_int) c_ulong;

/// Ports `server_decision_question_type` (server-decision.h:14 @d81235049).
pub const QuestionType = enum(c_int) {
    choice,
    score,
    noul,
};

/// Ports `decision_question_type_name` (server-decision.cpp:10 @d81235049).
fn questionTypeName(ty: QuestionType) []const u8 {
    return switch (ty) {
        .choice => "choice",
        .score => "score",
        .noul => "noul",
    };
}

// lev reads noul from a rating scale: 0 = certainly no, 8 = certainly yes
const lev_n_ratings: usize = 9;

/// Ports `server_decision_option` (server-decision.h:20 @d81235049).
pub const Option = struct {
    key: []const u8,
    description: Value, // null if not provided
};

/// Ports `server_decision_question` (server-decision.h:25 @d81235049).
pub const Question = struct {
    id: []const u8,
    type: QuestionType,
    instructions: Value,
    options: []Option, // in the order of the model outputs
};

/// Ports `decision_meta_str` (server-decision.cpp:22 @d81235049).
fn metaStr(a: Allocator, model: ?*const c.llama_model, key: []const u8) Error![]u8 {
    var buf: [256]u8 = undefined;
    const kz = try a.dupeSentinel(u8, key, 0);
    defer a.free(kz);
    const n = c.llama_model_meta_val_str(model, kz.ptr, &buf, buf.len);
    if (n < 0) return a.dupe(u8, "");
    return a.dupe(u8, std.mem.sliceTo(&buf, 0));
}

/// Ports `server_decision_context` (server-decision.h:32 @d81235049).
pub const DecisionContext = struct {
    gpa: Allocator = std.heap.c_allocator,

    type: DecisionType = .none,

    vocab: ?*const c.llama_vocab = null,
    tmpl: ?*Template = null, // the "systemone" template

    temperatures: std.StringArrayHashMapUnmanaged(f32) = .empty, // "<type>" or "<type>.<n_options bucket>"
    n_options_max: usize = 0,
    noul_true_first: bool = false, // noul options are [true, false] instead of [false, true]
    choice_sorted: bool = false, // choice options are in the order of their keys

    // OPENJEV, LEV, NIMBLE
    labels: std.ArrayList(Token) = .empty,
    label_texts: std.ArrayList([]u8) = .empty, // only if the label of an option is given to the template

    // LAYA, KEV
    token_marker: Token = token_null,
    token_sep: Token = token_null,
    text_marker: []u8 = &.{},
    max_head_tokens: usize = 0, // question + options
    max_option_tokens: usize = 48,

    pub fn deinit(self: *DecisionContext) void {
        const gpa = self.gpa;
        if (self.tmpl) |t| {
            t.deinit();
            gpa.destroy(t);
        }
        for (self.temperatures.keys()) |k| gpa.free(k);
        self.temperatures.deinit(gpa);
        self.labels.deinit(gpa);
        for (self.label_texts.items) |t| gpa.free(t);
        self.label_texts.deinit(gpa);
        gpa.free(self.text_marker);
        self.* = .{ .gpa = gpa };
    }

    /// Ports `init` (server-decision.cpp:32 @d81235049): read the
    /// "<arch>.decision.*" metadata, type stays NONE if the model has none.
    pub fn init(self: *DecisionContext, gpa: Allocator, model: ?*const c.llama_model) Error!void {
        self.deinit(); // the model can be reloaded
        self.gpa = gpa;
        errdefer self.deinit();

        const model_type = tc.common.getDecisionType(model);
        if (model_type == .none) {
            return;
        }

        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const ar = arena_state.allocator();

        const prefix = try std.mem.concat(ar, u8, &.{ try metaStr(ar, model, "general.architecture"), ".decision." });
        const type_name = try metaStr(ar, model, try std.mem.concat(ar, u8, &.{ prefix, "type" }));

        self.vocab = c.llama_model_get_vocab(model);

        const tmpl_src = c.llama_model_chat_template(model, "systemone");
        if (tmpl_src == null) {
            return throw(error.RuntimeError, "decision model has no \"systemone\" template", .{});
        }
        const t = try gpa.create(Template);
        t.* = Template.init(gpa, std.mem.span(tmpl_src), "", "") catch |e| {
            gpa.destroy(t);
            return e;
        };
        self.tmpl = t;

        const prefix_temp = try std.mem.concat(ar, u8, &.{ prefix, "temperature." });
        const n_meta = c.llama_model_meta_count(model);
        var i: i32 = 0;
        while (i < n_meta) : (i += 1) {
            var key: [256]u8 = undefined;
            var val: [64]u8 = undefined;
            if (c.llama_model_meta_key_by_index(model, i, &key, key.len) < 0 or !std.mem.startsWith(u8, std.mem.sliceTo(&key, 0), prefix_temp)) {
                continue;
            }
            if (c.llama_model_meta_val_str_by_index(model, i, &val, val.len) < 0) {
                continue;
            }
            const temp = strtof(@ptrCast(&val), null);
            const key_s = std.mem.sliceTo(&key, 0);
            if (temp <= 0.0) {
                return throw(error.RuntimeError, "invalid decision temperature: {s} = {s}", .{ key_s, std.mem.sliceTo(&val, 0) });
            }
            // std::map's operator[]: a key seen twice keeps the last value
            const name = key_s[prefix_temp.len..];
            const gop = try self.temperatures.getOrPut(gpa, name);
            if (!gop.found_existing) {
                gop.key_ptr.* = gpa.dupe(u8, name) catch |e| {
                    _ = self.temperatures.pop();
                    return e;
                };
            }
            gop.value_ptr.* = temp;
        }

        switch (model_type) {
            .openjev => {
                // one letter per option, each must be a single token
                const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
                for (letters) |ch| {
                    const toks = try common.tokenize(ar, self.vocab, &.{ch}, false, false);
                    if (toks.len != 1) {
                        return throw(error.RuntimeError, "decision label '{c}' is not a single token", .{ch});
                    }
                    try self.labels.append(gpa, toks[0]);
                }
                self.n_options_max = self.labels.items.len;
                self.noul_true_first = true;
            },
            .lev, .nimble => {
                // label codes are A..Z then AA..ZZ, only the ones that are a single token are used
                var codes: std.ArrayList([]const u8) = .empty;
                var ch: u8 = 'A';
                while (ch <= 'Z') : (ch += 1) try codes.append(ar, try ar.dupe(u8, &.{ch}));
                var x: u8 = 'A';
                while (x <= 'Z') : (x += 1) {
                    var y: u8 = 'A';
                    while (y <= 'Z') : (y += 1) try codes.append(ar, try ar.dupe(u8, &.{ x, y }));
                }
                for (codes.items) |code| {
                    const toks = try common.tokenize(ar, self.vocab, code, false, false);
                    if (toks.len == 1 and self.labels.items.len < 255) {
                        try self.labels.append(gpa, toks[0]);
                        const dup = try gpa.dupe(u8, code);
                        self.label_texts.append(gpa, dup) catch |e| {
                            gpa.free(dup);
                            return e;
                        };
                    }
                }
                self.n_options_max = self.labels.items.len;
            },
            .kev => {
                // the hidden state of an option is read at the token that ends it
                const toks = try common.tokenize(ar, self.vocab, "<|box_end|>", false, true);
                if (toks.len != 1) {
                    return throw(error.RuntimeError, "decision model has no <|box_end|> token", .{});
                }
                self.token_marker = toks[0];
                self.n_options_max = 255;
            },
            .laya => {
                self.token_marker = c.llama_vocab_mask(self.vocab);
                self.token_sep = c.llama_vocab_sep(self.vocab);
                if (self.token_marker == token_null or self.token_sep == token_null) {
                    return throw(error.RuntimeError, "decision model has no mask or sep token", .{});
                }
                self.text_marker = try gpa.dupe(u8, try tc.common.tokenToPiece(ar, self.vocab, self.token_marker, true));

                const val = try metaStr(ar, model, try std.mem.concat(ar, u8, &.{ prefix, "max_head_tokens" }));
                const valz = try ar.dupeSentinel(u8, val, 0);
                self.max_head_tokens = strtoul(valz.ptr, null, 10);
                if (self.max_head_tokens == 0) {
                    return throw(error.RuntimeError, "decision model has no valid max_head_tokens", .{});
                }
                self.n_options_max = 255;
            },
            .clef => {
                self.n_options_max = 255;
                self.noul_true_first = true;
                self.choice_sorted = true;
            },
            else => {
                return throw(error.RuntimeError, "unsupported decision model type: {s}", .{type_name});
            },
        }
        self.type = model_type;

        const z = try ar.dupeSentinel(u8, type_name, 0);
        common.srv(.inf, "init", "decision model type: %s\n", .{z.ptr});
    }

    /// Ports `can_share_prompt` (server-decision.h:39 @d81235049): true if
    /// the questions of a request start with the same tokens, and the model
    /// can continue from them.
    pub fn canSharePrompt(self: *const DecisionContext) bool {
        return switch (self.type) {
            .openjev, .lev, .kev, .nimble => true,
            else => false,
        };
    }

    /// Ports `is_joint` (server-decision.h:52 @d81235049): true if all the
    /// questions of a request go in one prompt, see `fillTaskJoint`.
    pub fn isJoint(self: *const DecisionContext) bool {
        return self.type == .clef;
    }

    /// Ports `can_use_images` (server-decision.h:57 @d81235049): true if the
    /// prompt of the model has a place for images.
    pub fn canUseImages(self: *const DecisionContext) bool {
        return switch (self.type) {
            .openjev, .clef => true,
            else => false,
        };
    }

    // -------------------------------------------------------------------------
    // request parsing

    /// Ports `parse_questions` (server-decision.cpp:137 @d81235049).
    /// `error.InvalidArgument` on bad input.
    ///
    /// Return: owned by `a` (an arena: nothing is freed piecewise).
    pub fn parseQuestions(self: *const DecisionContext, a: Allocator, body: *const Value) Error![]Question {
        if (!body.contains("state") or (try body.at("state")).isNull()) {
            return throw(error.InvalidArgument, "\"state\" must be provided", .{});
        }
        if (!body.contains("questions") or !(try body.at("questions")).isObject() or (try body.at("questions")).empty()) {
            return throw(error.InvalidArgument, "\"questions\" must be a non-empty object", .{});
        }

        var questions: std.ArrayList(Question) = .empty;
        const qs = try body.at("questions");
        for (qs.object.members.items) |*member| {
            const id = member.key;
            const q = &member.value;
            if (!q.isObject()) {
                return qerr(id, "must be an object", .{});
            }
            if (!q.contains("instructions") or (try q.at("instructions")).isNull()) {
                return qerr(id, "\"instructions\" must be provided", .{});
            }

            var question: Question = .{
                .id = id,
                .type = .choice,
                .instructions = try (try q.at("instructions")).clone(a),
                .options = &.{},
            };
            var options: std.ArrayList(Option) = .empty;

            const type_name = jsonValue(q, "type", []const u8, "");
            const criteria: Value = if (q.contains("criteria")) (try q.at("criteria")).* else .null;

            if (std.mem.eql(u8, type_name, "choice")) {
                question.type = .choice;
                if (!criteria.isObject() or criteria.empty()) {
                    return qerr(id, "\"criteria\" must be a non-empty object", .{});
                }
                for (criteria.object.members.items) |*cm| {
                    try options.append(a, .{ .key = cm.key, .description = try cm.value.clone(a) });
                }
                if (self.choice_sorted) {
                    std.mem.sort(Option, options.items, {}, struct {
                        fn lt(_: void, x: Option, y: Option) bool {
                            return std.mem.lessThan(u8, x.key, y.key);
                        }
                    }.lt);
                }
            } else if (std.mem.eql(u8, type_name, "score")) {
                question.type = .score;
                if (!criteria.isArray() or criteria.size() < 2 or criteria.size() > 10) {
                    return qerr(id, "\"criteria\" must be an array of 2 to 10 levels", .{});
                }
                for (criteria.array.items, 0..) |*level, i| {
                    try options.append(a, .{ .key = try std.fmt.allocPrint(a, "{d}", .{i}), .description = try level.clone(a) });
                }
            } else if (std.mem.eql(u8, type_name, "noul")) {
                question.type = .noul;
                if (!criteria.isNull() and !criteria.isObject()) {
                    return qerr(id, "\"criteria\" must be an object", .{});
                }
                for ([_][]const u8{ "false", "true" }) |key| {
                    const desc: Value = if (criteria.isObject() and criteria.contains(key)) try (try criteria.at(key)).clone(a) else .null;
                    try options.append(a, .{ .key = key, .description = desc });
                }
                if (self.noul_true_first) {
                    std.mem.swap(Option, &options.items[0], &options.items[1]);
                }
            } else {
                return qerr(id, "\"type\" must be one of: choice, score, noul", .{});
            }

            if (options.items.len > self.n_options_max) {
                return qerr(id, "too many options ({d}), this model supports at most {d}", .{ options.items.len, self.n_options_max });
            }

            question.options = options.items;
            try questions.append(a, question);
        }
        return questions.items;
    }

    /// The `err` lambda of `parse_questions`: `questions.<id>: <msg>`.
    fn qerr(id: []const u8, comptime fmt: []const u8, args: anytype) Error {
        var buf: [512]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, fmt, args) catch fmt;
        return throw(error.InvalidArgument, "questions.{s}: {s}", .{ id, msg });
    }

    /// Ports `parse_state` (server-decision.cpp:225 @d81235049): the state
    /// without its images, which are appended to `files` in order. Images
    /// come from "images" and from the image_url parts of a state made of
    /// chat messages.
    ///
    /// Return: owned by `a`; each file too.
    pub fn parseState(self: *const DecisionContext, a: Allocator, body: *const Value, files: *std.ArrayList([]u8)) Error!Value {
        _ = self;
        if (body.contains("videos") and !(try body.at("videos")).isNull() and !(try body.at("videos")).empty()) {
            return throw(error.InvalidArgument, "\"videos\" is not supported", .{});
        }
        if (body.contains("images") and !(try body.at("images")).isNull()) {
            const images = try body.at("images");
            if (!images.isArray()) {
                return throw(error.InvalidArgument, "\"images\" must be an array", .{});
            }
            for (images.array.items) |*url| {
                try loadImage(a, url, files);
            }
        }

        const state = try body.at("state");
        const is_wrapped = state.isObject() and state.contains("messages");
        const messages = if (is_wrapped) try state.at("messages") else state;
        if (!messages.isArray()) {
            return state.clone(a);
        }

        // chat messages: take the image parts out of the content
        var messages_out = Value.emptyArray();
        for (messages.array.items) |*msg| {
            if (!msg.isObject() or !msg.contains("content") or !(try msg.at("content")).isArray()) {
                try messages_out.pushBack(a, try msg.clone(a));
                continue;
            }
            var content = Value.emptyArray();
            for ((try msg.at("content")).array.items) |*part| {
                if (part.isObject() and std.mem.eql(u8, jsonValue(part, "type", []const u8, ""), "image_url") and part.contains("image_url")) {
                    const image_url = try part.at("image_url");
                    try loadImage(a, if (image_url.isObject() and image_url.contains("url")) try image_url.at("url") else image_url, files);
                } else {
                    try content.pushBack(a, try part.clone(a));
                }
            }
            var msg_out = try msg.clone(a);
            try msg_out.set(a, "content", content);
            try messages_out.pushBack(a, msg_out);
        }

        if (!is_wrapped) {
            return messages_out;
        }
        var state_out = try state.clone(a);
        try state_out.set(a, "messages", messages_out);
        return state_out;
    }

    // -------------------------------------------------------------------------
    // prompt

    /// Ports `n_variants` (server-decision.cpp:362 @d81235049): the number of
    /// prompts that are evaluated to answer this question, each one shows the
    /// options in a different order.
    pub fn nVariants(self: *const DecisionContext, question: *const Question) usize {
        // lev shows the options of a choice in 2 orders, to cancel the preference for the first label
        if (self.type == .lev and question.type == .choice and question.options.len > 1) {
            return 2;
        }
        return 1;
    }

    /// Ports `n_outputs` (server-decision.cpp:370 @d81235049).
    fn nOutputs(self: *const DecisionContext, question: *const Question) usize {
        if (self.type == .lev and question.type == .noul) {
            return lev_n_ratings;
        }
        return question.options.len;
    }

    /// Ports `render_options` (server-decision.cpp:377 @d81235049).
    fn renderOptions(self: *const DecisionContext, a: Allocator, question: *const Question, variant: usize) Error!Value {
        const n_options = question.options.len;

        // the second variant shows the options in the reverse order
        var options = Value.emptyArray();
        for (0..n_options) |i| {
            const opt = &question.options[if (variant == 0) i else n_options - 1 - i];
            var option = Obj.init(a);
            try option.put("key", opt.key);
            try option.put("description", try opt.description.clone(a));
            if (self.type == .kev) {
                try option.put("key", try kevText(a, &.{ .string = @constCast(opt.key) }));
                if (!opt.description.isNull()) {
                    try option.put("description", try kevText(a, &opt.description));
                }
            }
            if (self.label_texts.items.len != 0) {
                try option.put("label", self.label_texts.items[i]);
            }
            try options.pushBack(a, option.done());
        }
        return options;
    }

    /// Ports `render` (server-decision.cpp:402 @d81235049).
    ///
    /// Return: owned by `a`.
    fn render(
        self: *const DecisionContext,
        a: Allocator,
        state: *const Value,
        questions: []const Question,
        question: *const Question,
        variant: usize,
        n_images: usize,
    ) Error![]u8 {
        // the template is given raw JSON values, it serializes the ones that are not strings
        var o = Obj.init(a);
        try o.put("id", question.id);
        try o.put("type", questionTypeName(question.type));
        try o.put("instructions", try question.instructions.clone(a));
        try o.put("state", try state.clone(a));
        try o.put("options", try self.renderOptions(a, question, variant));
        var inp = o.done();

        // the nimble prompt lists all the questions of the request
        if (self.type == .nimble) {
            var qs = Value.emptyArray();
            for (questions) |*q| {
                var qo = Obj.init(a);
                try qo.put("id", q.id);
                try qo.put("type", questionTypeName(q.type));
                try qo.put("instructions", try q.instructions.clone(a));
                try qo.put("options", try self.renderOptions(a, q, 0));
                try qs.pushBack(a, qo.done());
            }
            try inp.set(a, "questions", qs);
        }

        // lev was trained with sorted keys
        if (self.type == .lev) {
            inp = try sortKeys(a, &inp);
        }

        // the kev template only takes text
        if (self.type == .kev) {
            try inp.set(a, "state", try kevText(a, state));
            try inp.set(a, "instructions", try kevText(a, &question.instructions));
        }

        // the input must not contain the marker of the options
        if (self.text_marker.len != 0) {
            inp = try replaceText(a, &inp, self.text_marker, " ");
        }

        // the template puts one media marker per image
        var images = Value.emptyArray();
        if (n_images > 0) {
            inp = try replaceText(a, &inp, common.getMediaMarker(), " ");
            for (0..n_images) |_| {
                try images.pushBack(a, try Value.initString(a, common.getMediaMarker()));
            }
        }
        try inp.set(a, "images", images);

        return self.runTemplate(a, &inp);
    }

    /// `jinja::context` + `global_from_json(ctx, inp, false)` +
    /// `runtime.execute` + `gather_string_parts`.
    fn runTemplate(self: *const DecisionContext, a: Allocator, inp: *const Value) Error![]u8 {
        var r: chat.template.Render = undefined;
        r.init(a, false);
        defer r.deinit();
        try r.bindJson(inp, false);
        return r.run(self.tmpl.?, a);
    }

    /// Ports `fill_task` (server-decision.cpp:463 @d81235049): set the prompt
    /// of one variant of this question, and where to read its result.
    /// `mctx` is only used if there are files.
    pub fn fillTask(
        self: *const DecisionContext,
        a: Allocator,
        gpa: Allocator,
        state: *const Value,
        questions: []const Question,
        question: *const Question,
        variant: usize,
        files: []const []const u8,
        mctx: ?*c.mtmd_context,
        init_opt: c.mtmd_helper_init_opt,
        task: *Task,
    ) Error!void {
        const prompt = try self.render(a, state, questions, question, variant, files.len);

        if (self.type == .openjev or self.type == .lev or self.type == .nimble) {
            // lev reads the ratings of a noul question at its first labels, not at the digits
            task.decision.labels.clearRetainingCapacity();
            try task.decision.labels.appendSlice(gpa, self.labels.items[0..self.nOutputs(question)]);
            if (files.len != 0) {
                task.tokens.deinit();
                task.tokens = try tokens_mod.processMtmdPrompt(gpa, mctx, prompt, files, init_opt, false);
                return;
            }
        }

        var tokens: std.ArrayList(Token) = .fromOwnedSlice(try common.tokenize(a, self.vocab, prompt, false, true));
        if (self.type == .laya) {
            try self.fillTaskLaya(a, gpa, &tokens, question, task);
        }
        if (self.type == .kev) {
            // an option is read at its end token, the question at the last token
            for (tokens.items, 0..) |tok, i| {
                if (tok == self.token_marker) {
                    try task.decision.markers.append(gpa, @intCast(i));
                }
            }
            if (task.decision.markers.items.len != question.options.len) {
                return throw(error.RuntimeError, "unexpected layout of the decision prompt", .{});
            }
            task.decision.pointer = @as(i32, @truncate(@as(i64, @intCast(tokens.items.len)))) -% 1;
        }
        task.tokens.deinit();
        task.tokens = try ServerTokens.fromTokens(gpa, tokens.items, false);
    }

    /// Ports `fill_task_laya` (server-decision.cpp:504 @d81235049): the
    /// prompt is `[cls] question [sep] ([marker] option)* [sep] state [sep]`;
    /// options and question are cut to fit max_head_tokens, the same way the
    /// model was trained.
    fn fillTaskLaya(self: *const DecisionContext, a: Allocator, gpa: Allocator, tokens: *std.ArrayList(Token), question: *const Question, task: *Task) Error!void {
        const n_options = question.options.len;
        const toks = tokens.items;

        var markers: std.ArrayList(usize) = .empty;
        for (toks, 0..) |tok, i| {
            if (tok == self.token_marker) {
                try markers.append(a, i);
            }
        }
        const invalid = "unexpected layout of the decision prompt";
        if (markers.items.len != n_options or markers.items[0] < 2 or toks[markers.items[0] - 1] != self.token_sep or toks[toks.len - 1] != self.token_sep) {
            return throw(error.RuntimeError, invalid, .{});
        }
        const head_end = markers.items[0] - 1;
        const last = markers.items[markers.items.len - 1];
        const opts_end = if (std.mem.indexOfScalar(Token, toks[last..], self.token_sep)) |p| last + p else toks.len;
        if (opts_end + 1 >= toks.len) {
            return throw(error.RuntimeError, invalid, .{});
        }

        // marker + text of each option
        var options: std.ArrayList(std.ArrayList(Token)) = .empty;
        var n_options_tokens: usize = 0;
        for (0..n_options) |i| {
            const end = if (i + 1 < n_options) markers.items[i + 1] else opts_end;
            var opt: std.ArrayList(Token) = .empty;
            try opt.appendSlice(a, toks[markers.items[i]..end]);
            try options.append(a, opt);
        }
        const SetMax = struct {
            fn run(opts: []std.ArrayList(Token), n_max: usize) usize {
                var n: usize = 0;
                for (opts) |*opt| {
                    opt.shrinkRetainingCapacity(@min(opt.items.len, n_max));
                    n += opt.items.len;
                }
                return n;
            }
        };
        n_options_tokens = SetMax.run(options.items, self.max_option_tokens + 1);
        if (n_options_tokens + 16 > self.max_head_tokens) {
            // too many or too long options, shrink them evenly
            n_options_tokens = SetMax.run(options.items, @max(@as(usize, 4), (self.max_head_tokens - @min(self.max_head_tokens, 16)) / n_options));
        }
        const n_question_max = @max(@as(usize, 8), self.max_head_tokens - @min(self.max_head_tokens, n_options_tokens));

        var out: std.ArrayList(Token) = .empty;
        try out.append(a, toks[0]);
        try out.appendSlice(a, toks[1..@min(head_end, 1 + n_question_max)]);
        try out.append(a, self.token_sep);
        for (options.items) |*opt| {
            try task.decision.markers.append(gpa, @intCast(out.items.len));
            try out.appendSlice(a, opt.items);
        }
        try out.appendSlice(a, toks[opts_end..]);
        tokens.* = out;

        // the output has one score per question type
        task.decision.column = @backingInt(question.type);
    }

    /// Ports `fill_task_joint` (server-decision.cpp:569 @d81235049): set the
    /// prompt of all the questions, the result has the scores of all their
    /// options, in order. `mctx` is only used if there are files.
    pub fn fillTaskJoint(
        self: *const DecisionContext,
        a: Allocator,
        gpa: Allocator,
        state: *const Value,
        questions: []const Question,
        files: []const []const u8,
        mctx: ?*c.mtmd_context,
        init_opt: c.mtmd_helper_init_opt,
        task: *Task,
    ) Error!void {
        var inp_questions = Value.emptyArray();
        for (questions) |*question| {
            var options = Value.emptyArray();
            for (question.options) |*opt| {
                var oo = Obj.init(a);
                try oo.put("key", opt.key);
                try oo.put("description", try opt.description.clone(a));
                try options.pushBack(a, oo.done());
            }
            var qo = Obj.init(a);
            try qo.put("id", question.id);
            try qo.put("type", questionTypeName(question.type));
            try qo.put("instructions", try question.instructions.clone(a));
            try qo.put("options", options);
            try inp_questions.pushBack(a, qo.done());
        }

        // the template is given raw JSON values with sorted keys, and no marker in the input
        var io = Obj.init(a);
        try io.put("state", try state.clone(a));
        try io.put("questions", inp_questions);
        var inp0 = io.done();
        const sorted = try sortKeys(a, &inp0);
        var inp = try replaceText(a, &sorted, clef_marker, "<<clef ");

        // the template puts one media marker per image
        var images = Value.emptyArray();
        if (files.len != 0) {
            inp = try replaceText(a, &inp, common.getMediaMarker(), " ");
            for (0..files.len) |_| {
                try images.pushBack(a, try Value.initString(a, common.getMediaMarker()));
            }
        }
        try inp.set(a, "images", images);
        try inp.set(a, "sep", try Value.initString(a, clef_sep));
        try inp.set(a, "mark_question", try Value.initString(a, clef_mark_question));
        try inp.set(a, "mark_option", try Value.initString(a, clef_mark_option));

        const prompt = try self.runTemplate(a, &inp);

        const invalid = "unexpected layout of the decision prompt";

        // the model was trained with the pieces tokenized one by one
        var pieces: std.ArrayList([]const u8) = .empty;
        {
            var it = std.mem.splitSequence(u8, prompt, clef_sep);
            while (it.next()) |p| try pieces.append(a, p);
        }
        var i_piece: usize = 0;

        task.tokens.deinit();
        task.tokens = try ServerTokens.fromTokens(gpa, &.{}, false);
        const marker = common.getMediaMarker();
        if (files.len != 0) {
            // the text before the images is tokenized with them, a vision token separates the pieces anyway
            var head: std.ArrayList(u8) = .empty;
            while (i_piece < pieces.items.len and std.mem.indexOf(u8, head.items, marker) == null) {
                try head.appendSlice(a, pieces.items[i_piece]);
                i_piece += 1;
            }
            if (std.mem.indexOf(u8, head.items, marker) == null or std.mem.indexOf(u8, head.items, clef_marker) != null) {
                return throw(error.RuntimeError, invalid, .{});
            }
            task.tokens.deinit();
            task.tokens = try tokens_mod.processMtmdPrompt(gpa, mctx, head.items, files, init_opt, false);
            try task.decision.order.appendNTimes(gpa, orderValue(.none), task.tokens.size() -| task.decision.order.items.len);
        }

        var i_question: usize = 0;
        while (i_piece < pieces.items.len) : (i_piece += 1) {
            var piece = pieces.items[i_piece];
            var order: i32 = orderValue(.none);
            if (std.mem.startsWith(u8, piece, clef_mark_question)) {
                piece = piece[clef_mark_question.len..];
                if (i_question >= questions.len) {
                    return throw(error.RuntimeError, invalid, .{});
                }
                order = switch (questions[i_question].type) {
                    .noul => orderValue(.question_noul),
                    .choice => orderValue(.question_choice),
                    .score => orderValue(.question_score),
                };
                i_question += 1;
            } else if (std.mem.startsWith(u8, piece, clef_mark_option)) {
                piece = piece[clef_mark_option.len..];
                order = orderValue(.option);
                task.decision.n_scores += 1;
            }

            const piece_tokens = try common.tokenize(a, self.vocab, piece, false, true);
            if (order != orderValue(.none) and piece_tokens.len == 0) {
                return throw(error.InvalidArgument, "the instructions and the options of a question must not be empty", .{});
            }
            for (piece_tokens) |token| {
                try task.tokens.pushBackToken(token);
            }
            // std::vector::resize: grows with order, never shrinks here
            const n = task.tokens.size();
            if (n > task.decision.order.items.len) {
                try task.decision.order.appendNTimes(gpa, order, n - task.decision.order.items.len);
            }
        }

        var n_options: usize = 0;
        for (questions) |*question| {
            n_options += question.options.len;
        }
        if (i_question != questions.len or @as(usize, @intCast(task.decision.n_scores)) != n_options) {
            return throw(error.RuntimeError, invalid, .{});
        }
    }

    // -------------------------------------------------------------------------
    // answer

    /// Ports `get_temperature` (server-decision.cpp:682 @d81235049).
    fn getTemperature(self: *const DecisionContext, question: *const Question) f32 {
        const n = question.options.len;
        const type_name = questionTypeName(question.type);

        // the temperature can depend on the number of options, the buckets are the ones used to fit it
        const bucket: []const u8 = if (self.type == .lev)
            (if (n <= 8) "small" else if (n <= 26) "mid" else "large")
        else
            (if (n <= 2) "2" else if (n <= 5) "3_5" else if (n <= 10) "6_10" else "11");

        var buf: [64]u8 = undefined;
        const with_bucket = std.fmt.bufPrint(&buf, "{s}.{s}", .{ type_name, bucket }) catch |err| switch (err) {
            error.NoSpaceLeft => unreachable,
        };
        for ([_][]const u8{ with_bucket, type_name }) |name| {
            if (self.temperatures.get(name)) |t| {
                return t;
            }
        }
        return 1.0;
    }

    /// Ports `format_answer` (server-decision.cpp:731 @d81235049): `scores`
    /// holds the raw model outputs of each variant.
    ///
    /// Return: owned by `a`.
    pub fn formatAnswer(self: *const DecisionContext, a: Allocator, question: *const Question, scores: []const []const f32) Error!Value {
        const n = self.nOutputs(question);
        if (scores.len != self.nVariants(question)) {
            return throw(error.RuntimeError, "decision result does not match the number of variants", .{});
        }

        // softmax over the outputs of each variant, then the average of the variants
        const temperature = self.getTemperature(question);
        const probs = try a.alloc(f64, n);
        @memset(probs, 0.0);
        const p = try a.alloc(f64, n);
        for (scores, 0..) |s, v| {
            if (s.len != n) {
                return throw(error.RuntimeError, "decision result does not match the number of options", .{});
            }
            // a joint head returns NaN if it could not use the decision order
            for (s) |x| {
                if (std.math.isNan(x)) {
                    return throw(error.RuntimeError, "the model could not evaluate the decision", .{});
                }
            }
            // std::max_element: the first of the greatest
            var score_max = s[0];
            for (s[1..]) |x| {
                if (score_max < x) score_max = x;
            }
            var sum: f64 = 0.0;
            for (0..n) |i| {
                p[i] = exp(@as(f64, s[i] - score_max) / @as(f64, temperature));
                sum += p[i];
            }
            for (0..n) |i| {
                // the second variant is in the reverse order
                probs[if (v == 0) i else n - 1 - i] += p[i] / sum / @as(f64, @floatFromInt(scores.len));
            }
        }

        var answer = Obj.init(a);
        try answer.put("type", questionTypeName(question.type));

        if (question.type == .noul) {
            if (self.type == .lev) {
                var expected: f64 = 0.0;
                for (0..n) |i| {
                    expected += probs[i] * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n - 1));
                }
                try answer.put("noul", expected);
                return answer.done();
            }
            for (0..n) |i| {
                if (std.mem.eql(u8, question.options[i].key, "true")) {
                    try answer.put("noul", probs[i]);
                }
            }
            return answer.done();
        }

        var probabilities = Obj.init(a);
        for (0..n) |i| {
            try probabilities.put(question.options[i].key, probs[i]);
        }

        if (question.type == .choice) {
            const best = maxIndex(probs);
            try answer.put("choice", question.options[best].key);
            try answer.put("probabilities", probabilities.done());
            try answer.put("confidence", confidenceChoice(probs));
        } else {
            var expected: f64 = 0.0;
            var legend = Obj.init(a);
            for (0..n) |i| {
                // clang contracts this one: the loop builds JSON, so it stays scalar and fused
                expected = @mulAdd(f64, @floatFromInt(i), probs[i], expected);
                try legend.put(question.options[i].key, try question.options[i].description.clone(a));
            }
            try answer.put("score", expected);
            try answer.put("legend", legend.done());
            try answer.put("probabilities", probabilities.done());
            try answer.put("confidence", confidenceScore(probs));
        }
        return answer.done();
    }
};

/// `std::max_element`'s index: the first of the greatest.
fn maxIndex(xs: []const f64) usize {
    var best: usize = 0;
    for (xs, 0..) |x, i| {
        if (xs[best] < x) best = i;
    }
    return best;
}

// confidence formulas are the ones published by TypeSafe

/// Ports `decision_confidence_choice` (server-decision.cpp:705 @d81235049).
fn confidenceChoice(probs: []const f64) f64 {
    if (probs.len < 2) {
        return 1.0;
    }
    const uniform = 1.0 / @as(f64, @floatFromInt(probs.len));
    const p_max = probs[maxIndex(probs)];
    return @max(0.0, (p_max - uniform) / (1.0 - uniform));
}

/// Ports `decision_confidence_score` (server-decision.cpp:714 @d81235049).
fn confidenceScore(probs: []const f64) f64 {
    if (probs.len < 2) {
        return 1.0;
    }
    const n = probs.len;
    const mode = maxIndex(probs);
    const nf: f64 = @floatFromInt(n);

    // mean distance to the mode, relative to the one of a uniform distribution around its center
    //
    // As the reference compiles it: from 8 elements the loop is vectorized
    // eight at a time, those products rounded and added in order, and only
    // the `n % 8` left over are fused (read from the disassembly; the
    // division in the second sum keeps it plain either way).
    const n_vec = if (n >= 8) n - n % 8 else 0;
    var dist: f64 = 0.0;
    var dist_uniform: f64 = 0.0;
    for (0..n) |i| {
        const fi: f64 = @floatFromInt(i);
        const d = @abs(fi - @as(f64, @floatFromInt(mode)));
        if (i < n_vec) {
            dist += probs[i] * d;
        } else {
            dist = @mulAdd(f64, probs[i], d, dist);
        }
        dist_uniform += @abs(fi - (nf - 1) / 2.0) / nf;
    }
    return @max(0.0, 1.0 - dist / dist_uniform);
}

// -----------------------------------------------------------------------------
// images

const max_images: usize = 8;

/// Ports `decision_load_image` (server-decision.cpp:215 @d81235049).
fn loadImage(a: Allocator, url: *const Value, files: *std.ArrayList([]u8)) Error!void {
    if (!url.isString() or !std.mem.startsWith(u8, url.string, "data:image/")) {
        return throw(error.InvalidArgument, "images must be data URLs (data:image/...;base64,...)", .{});
    }
    if (files.items.len >= max_images) {
        return throw(error.InvalidArgument, "too many images, the maximum is {d}", .{max_images});
    }
    try oaicompat.handleMedia(a, files, url.string, "");
}

// -----------------------------------------------------------------------------
// JSON helpers

/// Ports `decision_replace_text` (server-decision.cpp:279 @d81235049):
/// replace text in all strings of a JSON value.
///
/// Return: owned by `a`.
fn replaceText(a: Allocator, val: *const Value, search: []const u8, replace: []const u8) Error!Value {
    switch (val.*) {
        .string => |s| {
            return Value.initString(a, try stringReplaceAll(a, s, search, replace));
        },
        .array => |arr| {
            var out = Value.emptyArray();
            for (arr.items) |*item| try out.pushBack(a, try replaceText(a, item, search, replace));
            return out;
        },
        .object => |m| {
            var out = Value.emptyObject();
            for (m.members.items) |*member| try out.set(a, member.key, try replaceText(a, &member.value, search, replace));
            return out;
        },
        else => return val.clone(a),
    }
}

/// Ports `string_replace_all` (common/common.cpp:510 @d81235049).
fn stringReplaceAll(a: Allocator, s: []const u8, search: []const u8, replace: []const u8) Error![]u8 {
    if (search.len == 0) {
        return a.dupe(u8, s);
    }
    var builder: std.ArrayList(u8) = .empty;
    var last_pos: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, s, pos, search)) |p| {
        try builder.appendSlice(a, s[last_pos..p]);
        try builder.appendSlice(a, replace);
        last_pos = p + search.len;
        pos = last_pos;
    }
    try builder.appendSlice(a, s[last_pos..]);
    return builder.items;
}

/// Ports `decision_sort_keys` (server-decision.cpp:303 @d81235049): sort the
/// keys of all objects of a JSON value.
///
/// Return: owned by `a`.
fn sortKeys(a: Allocator, val: *const Value) Error!Value {
    switch (val.*) {
        .array => |arr| {
            var out = Value.emptyArray();
            for (arr.items) |*item| try out.pushBack(a, try sortKeys(a, item));
            return out;
        },
        .object => |m| {
            // std::map<std::string, json>: one entry per key, in byte order
            const Entry = struct { key: []const u8, val: *const Value };
            var entries: std.ArrayList(Entry) = .empty;
            for (m.members.items) |*member| try entries.append(a, .{ .key = member.key, .val = &member.value });
            std.mem.sort(Entry, entries.items, {}, struct {
                fn lt(_: void, x: Entry, y: Entry) bool {
                    return std.mem.lessThan(u8, x.key, y.key);
                }
            }.lt);
            var out = Value.emptyObject();
            for (entries.items) |e| try out.set(a, e.key, try sortKeys(a, e.val));
            return out;
        },
        else => return val.clone(a),
    }
}

/// Ports `decision_kev_render` (server-decision.cpp:326 @d81235049): kev
/// flattens a JSON value into text, the keys of an object are kept as labels.
///
/// Return: owned by `a`.
fn kevRender(a: Allocator, val: *const Value, indent: usize) Error![]u8 {
    const pad = try a.alloc(u8, 2 * indent);
    @memset(pad, ' ');
    switch (val.*) {
        .null => return a.dupe(u8, ""),
        .string => |s| return a.dupe(u8, s),
        .boolean => |b| return a.dupe(u8, if (b) "True" else "False"),
        .array => |arr| {
            var out: std.ArrayList(u8) = .empty;
            for (arr.items) |*item| {
                const text = try kevRender(a, item, indent + 1);
                const first = std.mem.indexOfNone(u8, text, " \t\n\r") orelse text.len;
                if (out.items.len != 0) try out.append(a, '\n');
                try out.appendSlice(a, pad);
                try out.appendSlice(a, "- ");
                try out.appendSlice(a, text[first..]);
            }
            return out.items;
        },
        .object => |m| {
            var out: std.ArrayList(u8) = .empty;
            for (m.members.items) |*member| {
                const item = &member.value;
                const is_nested = item.isObject() or item.isArray();
                if (out.items.len != 0) try out.append(a, '\n');
                try out.appendSlice(a, pad);
                try out.appendSlice(a, member.key);
                try out.appendSlice(a, if (is_nested) ":\n" else ": ");
                try out.appendSlice(a, try kevRender(a, item, if (is_nested) indent + 1 else 0));
            }
            return out.items;
        },
        else => return json.dump(a, val, -1),
    }
}

/// Ports `decision_kev_text` (server-decision.cpp:357 @d81235049): special
/// tokens written in the text must not be parsed as such --
/// `std::regex_replace` of `<\|([A-Za-z0-9_]+)\|>` by `<¦$1¦>`, scanned by
/// hand: the leftmost match at each position, none overlapping.
///
/// Return: a string value owned by `a`.
fn kevText(a: Allocator, val: *const Value) Error!Value {
    const text = try kevRender(a, val, 0);
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < text.len) {
        if (std.mem.startsWith(u8, text[i..], "<|")) {
            var j = i + 2;
            while (j < text.len and (std.ascii.isAlphanumeric(text[j]) or text[j] == '_')) j += 1;
            if (j > i + 2 and std.mem.startsWith(u8, text[j..], "|>")) {
                try out.appendSlice(a, "<\xC2\xA6");
                try out.appendSlice(a, text[i + 2 .. j]);
                try out.appendSlice(a, "\xC2\xA6>");
                i = j + 2;
                continue;
            }
        }
        try out.append(a, text[i]);
        i += 1;
    }
    return Value.initString(a, out.items);
}

/// Ports `CLEF_MARKER`, `CLEF_SEP`, `CLEF_MARK_QUESTION` and
/// `CLEF_MARK_OPTION` (server-decision.cpp:564, 565, 566, 567 @d81235049):
/// the joint prompt's (clef) text given to the template, between the pieces
/// of the prompt, and at the start of the span of a question or of an option.
const clef_marker = "<<clef:";
const clef_sep = "<<clef:sep>>";
const clef_mark_question = "<<clef:question>>";
const clef_mark_option = "<<clef:option>>";

// -----------------------------------------------------------------------------
// shared prompt prefix

/// Ports `server_decision_group_tasks` (server-decision.cpp:810
/// @d81235049): group the tasks so that the common prefix of their prompts
/// is evaluated only once; each group is one parent and its children, it
/// takes at most `n_slots` slots. The order of the tasks is preserved.
///
/// Parameters:
/// - `tasks`: moved from; every element is left empty.
///
/// Return: owned by `gpa`.
pub fn groupTasks(gpa: Allocator, tasks: []Task, n_slots_in: usize) Error!std.ArrayList(Task) {
    const n_slots = @max(n_slots_in, 1);

    var groups: std.ArrayList(Task) = .empty;
    errdefer {
        for (groups.items) |*t| t.deinit(gpa);
        groups.deinit(gpa);
    }
    var i: usize = 0;
    while (i < tasks.len) : (i += n_slots) {
        const end = @min(tasks.len, i + n_slots);
        const parent = &tasks[i];

        // every task must have at least one token of its own to evaluate
        var n_shared: usize = parent.tokens.size() -% 1;
        for (i + 1..end) |j| {
            n_shared = @min(n_shared, try parent.tokens.getCommonPrefix(&tasks[j].tokens));
            n_shared = @min(n_shared, tasks[j].tokens.size() -% 1);
        }

        if (end - i < 2 or n_shared == 0) {
            for (i..end) |j| {
                try groups.append(gpa, tasks[j]);
                tasks[j] = .{};
            }
            continue;
        }

        parent.n_tokens_shared = @truncate(@as(i64, @intCast(n_shared)));
        for (i + 1..end) |j| {
            tasks[j].id_parent = parent.id;
            try parent.child_tasks.append(gpa, tasks[j]);
            tasks[j] = .{};
        }
        try groups.append(gpa, parent.*);
        parent.* = .{};
    }
    return groups;
}
