//! `llama-tokenize`: print a prompt's tokens.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/tokenize/tokenize.cpp` at v1.0.0
//! (`b1b770d90`). Each declaration names the C++ it replaces and the line it
//! began at.

const std = @import("std");
const stdlib = @import("stdlib");
const builtin = @import("builtin");
const tc = @import("tools_common");

comptime {
    // libllama, compiled in (see `build/tools.zig`).
    _ = @import("llama").lib;
}

/// Ports `print_usage` (tokenize.cpp:20 @b1b770d90).
fn printUsage(argv: []const [:0]const u8) void {
    tc.log.out("\nexample usage:\n", .{});
    tc.log.out("\n  %s -m your_model.gguf -p \"Hello world\"\n", .{argv[0].ptr});
    tc.log.out("\n  %s -m your_model.gguf -f prompt.txt --ids\n", .{argv[0].ptr});
    tc.log.out("\n  cat prompt.txt | %s -m your_model.gguf --stdin --show-count\n", .{argv[0].ptr});
    tc.log.out("\n", .{});
}

const win = struct {
    extern "kernel32" fn GetStdHandle(n: u32) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn GetConsoleMode(h: ?*anyopaque, mode: *u32) callconv(.winapi) c_int;
    extern "kernel32" fn MultiByteToWideChar(cp: c_uint, flags: u32, s: [*]const u8, n: c_int, w: ?[*]u16, wn: c_int) callconv(.winapi) c_int;
    extern "kernel32" fn WriteConsoleW(h: ?*anyopaque, s: [*]const u16, n: u32, written: ?*u32, reserved: ?*anyopaque) callconv(.winapi) c_int;
    extern "kernel32" fn GetLastError() callconv(.winapi) u32;
};

/// Ports `write_utf8_cstr_to_stdout` (tokenize.cpp:41 @b1b770d90): `printf`,
/// or on a Windows console the UTF-16 it needs, with invalid UTF-8 written
/// as its bytes in hex.
fn writeUtf8ToStdout(a: std.mem.Allocator, str: [:0]const u8, invalid_utf8: *bool) !void {
    invalid_utf8.* = false;
    if (comptime builtin.os.tag == .windows) {
        // Are we in a console?
        const STD_OUTPUT_HANDLE: u32 = @bitCast(@as(i32, -11));
        const h = win.GetStdHandle(STD_OUTPUT_HANDLE);
        var mode: u32 = 0;
        if (h == null or @intFromPtr(h) == std.math.maxInt(usize) or win.GetConsoleMode(h, &mode) == 0) {
            tc.io.printf("%s", .{str.ptr});
            return;
        }
        // MultiByteToWideChar reports an error if str is empty, don't report
        // them as invalid_utf8.
        if (str.len == 0) return;
        const CP_UTF8 = 65001;
        const MB_ERR_INVALID_CHARS = 0x8;
        const n = win.MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, str.ptr, @intCast(str.len), null, 0);
        if (n == 0) {
            const ERROR_NO_UNICODE_TRANSLATION = 1113;
            if (win.GetLastError() == ERROR_NO_UNICODE_TRANSLATION) {
                invalid_utf8.* = true;
                tc.io.printf("<", .{});
                for (str, 0..) |b, i| {
                    if (i > 0) tc.io.printf(" ", .{});
                    tc.io.printf("%02x", .{@as(c_uint, b)});
                }
                tc.io.printf(">", .{});
                return;
            }
            _ = tc.c.ggml_abort("tokenize.cpp", 78, "MultiByteToWideChar() failed in an unexpected way.");
        }
        const wstr = try a.alloc(u16, @intCast(n + 1));
        _ = win.MultiByteToWideChar(CP_UTF8, 0, str.ptr, @intCast(str.len), wstr.ptr, n);
        stdlib.stdio.stdout().flush();
        _ = win.WriteConsoleW(h, wstr.ptr, @intCast(n), null, null);
        return;
    }
    // TODO: reporting invalid_utf8 would be useful on non-Windows too.
    // printf will silently just write bad unicode.
    tc.io.printf("%s", .{str.ptr});
}

/// Reads all of standard input, as `ss << std::cin.rdbuf()` does: its
/// bytes as they are, no newline translation on Windows.
///
/// Return: the bytes, owned by `a`; an error when they cannot be read.
fn readStdin(a: std.mem.Allocator, std_io: std.Io) ![]const u8 {
    var buf: [65536]u8 = undefined;
    var r = std.Io.File.stdin().readerStreaming(std_io, &buf);
    return r.interface.allocRemaining(a, .unlimited);
}

/// Ports `main` (tokenize.cpp:95 @b1b770d90).
fn run(a: std.mem.Allocator, std_io: std.Io, argv: []const [:0]const u8) !u8 {
    var params = try tc.params.Params.init(a);

    tc.common.init();

    if (!try tc.arg.parse(argv, &params, .tokenize, printUsage)) return 1;

    // -f and -p both land in params.prompt; -f also sets prompt_file. -f and -p
    // resolve like the other tools (no mutual exclusion), --stdin takes precedence.
    const use_stdin = params.tokenize_stdin;
    const use_file = params.prompt_file.len != 0;

    // must have some prompt
    if (!use_stdin and !use_file and params.prompt.len == 0) {
        tc.log.err("error: must specify one of: --stdin, --file or --prompt\n", .{});
        return 1;
    }

    var prompt: []const u8 = "";
    if (use_file) {
        // read the file verbatim: common's -f handler strips a single trailing
        // newline, but for a tokenizer the input bytes must be preserved exactly
        // (a trailing newline is itself a token). escapes are applied locally
        // to match the behavior of -p/--prompt and --stdin.
        const pf = try a.dupeSentinel(u8, params.prompt_file, 0);
        prompt = std.Io.Dir.cwd().readFileAlloc(std_io, pf, a, .unlimited) catch {
            tc.log.err("error: could not open file '%s' for reading\n", .{pf.ptr});
            return 1;
        };
        if (params.escape) prompt = try tc.arg.processEscapes(a, prompt);
    } else if (!use_stdin) {
        // -p/--prompt is already escape-processed by common_params_parse()
        // (controlled by --escape/--no-escape), so use it verbatim here.
        prompt = params.prompt;
    }
    // else: we read stdin *after* loading the model (early exit if the
    // model cannot be loaded, which is a nicer user experience)

    tc.c.llama_backend_init();

    // load only the vocabulary (no weights), since tokenizing does not need them
    var model_params = tc.c.llama_model_default_params();
    model_params.vocab_only = true;
    const model_path = try a.dupeSentinel(u8, params.model.path, 0);
    const model = tc.c.llama_model_load_from_file(model_path.ptr, model_params) orelse {
        tc.log.err("error: could not load model from file '%s'.\n", .{model_path.ptr});
        return 1;
    };

    const vocab = tc.c.llama_model_get_vocab(model);

    const ctx_params = tc.c.llama_context_default_params();
    const ctx = tc.c.llama_init_from_model(model, ctx_params) orelse {
        tc.log.err("error: could not create context.\n", .{});
        return 1;
    };

    // read entire prompt from stdin?
    if (params.tokenize_stdin) {
        prompt = readStdin(a, std_io) catch {
            tc.log.err("error: could not read the entire standard input.\n", .{});
            return 1;
        };

        // stdin is not seen by common_params_parse(), so apply escape handling
        // here to match the behavior of -p/--prompt and -f/--file.
        if (params.escape) prompt = try tc.arg.processEscapes(a, prompt);
    }

    const model_wants_add_bos = tc.c.llama_vocab_get_add_bos(vocab);
    const add_bos = model_wants_add_bos and !params.tokenize_no_bos;
    const parse_special = params.parse_special;

    const tokens = tc.common.tokenize(a, vocab, prompt, add_bos, parse_special) catch |e| {
        if (e == error.Throw) {
            tc.io.eprintf("terminate called after throwing an instance of 'std::runtime_error'\n  what():  %.*s\n", .{ @as(c_int, @intCast(tc.arg.what().len)), tc.arg.what().ptr });
            std.process.abort();
        }
        return e;
    };

    if (params.tokenize_ids) tc.io.printf("[", .{});

    for (tokens, 0..) |t, i| {
        if (params.tokenize_ids) {
            if (i > 0) tc.io.printf(", ", .{});
            tc.io.printf("%d", .{t});
        } else {
            var invalid_utf8 = false;
            tc.io.printf("%6d -> '", .{t});
            try writeUtf8ToStdout(a, try tc.common.tokenToPiece(a, vocab, t, true), &invalid_utf8);
            if (invalid_utf8) {
                tc.io.printf("' (utf-8 decode failure)\n", .{});
            } else {
                tc.io.printf("'\n", .{});
            }
        }
    }

    if (params.tokenize_ids) tc.io.printf("]\n", .{});

    if (params.tokenize_show_count) tc.io.printf("Total number of tokens: %zu\n", .{tokens.len});

    // silence valgrind
    tc.c.llama_free(ctx);
    tc.c.llama_model_free(model);

    return 0;
}

pub fn main(init: std.process.Init) !u8 {
    stdlib.stdio.bufferAsC();
    stdlib.env.set(init.environ_map);
    defer stdlib.stdio.flushAll();
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return run(arena, init.io, argv);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
