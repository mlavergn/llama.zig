//! `console`: the interactive line reader a chat tool reads its user from,
//! and the display colours around it.
//!
//! # Provenance
//!
//! Ports the POSIX arms of `llama.cpp/common/console.cpp` at v0.3.0
//! (`c1d0e7a00`) as far as `llama-mtmd-cli` reaches them: `init`,
//! `cleanup`, `set_display`, `readline` with both its readers, and every
//! helper `readline_advanced` calls. Each declaration names the C++ it
//! replaces and the line it began at.
//!
//! # Not ported
//!
//! The completion callback (no ported tool sets one), the spinner and
//! `console::log`/`error` (`llama-mtmd-cli` calls none of them), and the
//! Windows arms: there `readline` falls back to reading UTF-8 lines from the
//! C library's `stdin`, which is `readline_simple`'s POSIX arm. `PLAN.md`'s
//! Windows pickup records the console's `ReadConsoleInputW` reader.

const std = @import("std");
const builtin = @import("builtin");
const c = @import("tools_h");
const io = @import("io.zig");

const is_windows = builtin.os.tag == .windows;

/// Ports `display_type` (console.h:11 @c1d0e7a00).
pub const Display = enum(c_int) { reset = 0, info, prompt, reasoning, user_input, err };

const ansi_color_red = "\x1b[31m";
const ansi_color_green = "\x1b[32m";
const ansi_color_yellow = "\x1b[33m";
const ansi_color_magenta = "\x1b[35m";
const ansi_color_gray = "\x1b[90m";
const ansi_color_reset = "\x1b[0m";
const ansi_bold = "\x1b[1m";

const weof: u32 = std.math.maxInt(u32); // WEOF, as `char32_t`

extern fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern fn fclose(f: ?*anyopaque) c_int;
extern fn fflush(f: ?*anyopaque) c_int;
extern fn fputc(ch: c_int, f: ?*anyopaque) c_int;
extern fn fputs(s: [*:0]const u8, f: ?*anyopaque) c_int;
extern fn fwrite(p: ?*const anyopaque, size: usize, n: usize, f: ?*anyopaque) usize;
extern fn fprintf(f: ?*anyopaque, fmt: [*:0]const u8, ...) c_int;
extern fn fscanf(f: ?*anyopaque, fmt: [*:0]const u8, ...) c_int;
extern fn getwchar() u32;
extern fn wcwidth(wc: u32) c_int;
extern fn iswspace(wc: u32) c_int;
extern fn setlocale(cat: c_int, locale: [*:0]const u8) ?[*:0]const u8;
extern fn ioctl(fd: c_int, req: c_ulong, ...) c_int;
extern fn getline(lineptr: *?[*]u8, n: *usize, f: ?*anyopaque) isize;
extern fn free(p: ?*anyopaque) void;

// Ports `advanced_display` and the console state (console.cpp:70 @c1d0e7a00)
var advanced_display: bool = false;
var simple_io: bool = true;
var current_display: Display = .reset;
var out_stream: ?*anyopaque = null; // `stdout` until `init`
var tty: ?*anyopaque = null;
var initial_state: std.c.termios = undefined;

fn out() ?*anyopaque {
    return out_stream orelse io.cStdout();
}

const lc_all: c_int = if (builtin.os.tag.isDarwin()) 0 else 6;

/// Ports `init` (console.cpp:89 @c1d0e7a00), the POSIX arm.
pub fn init(use_simple_io: bool, use_advanced_display: bool) void {
    advanced_display = use_advanced_display;
    simple_io = use_simple_io;
    if (comptime is_windows) {
        simple_io = true;
        return;
    }
    // POSIX-specific console initialization
    if (!simple_io) {
        _ = std.c.tcgetattr(std.posix.STDIN_FILENO, &initial_state);
        var new_termios = initial_state;
        new_termios.lflag.ICANON = false;
        new_termios.lflag.ECHO = false;
        new_termios.cc[@backingInt(std.c.V.MIN)] = 1;
        new_termios.cc[@backingInt(std.c.V.TIME)] = 0;
        _ = std.c.tcsetattr(std.posix.STDIN_FILENO, .NOW, &new_termios);

        tty = fopen("/dev/tty", "w+");
        if (tty != null) out_stream = tty;
    }

    _ = setlocale(lc_all, "");
}

/// Ports `cleanup` (console.cpp:151 @c1d0e7a00).
pub fn cleanup() void {
    // Reset console display
    setDisplay(.reset);

    if (comptime is_windows) return;
    // Restore settings on POSIX systems
    if (!simple_io) {
        if (tty != null) {
            out_stream = null;
            _ = fclose(tty);
            tty = null;
        }
        _ = std.c.tcsetattr(std.posix.STDIN_FILENO, .NOW, &initial_state);
    }
}

/// Ports `set_display` (console.cpp:173 @c1d0e7a00): emit an ANSI code
/// only when the display changes. The log is synchronous here, so
/// `common_log_flush` has nothing to flush.
pub fn setDisplay(display: Display) void {
    if (advanced_display and current_display != display) {
        const code: [*:0]const u8 = switch (display) {
            .reset => ansi_color_reset,
            .info => ansi_color_magenta,
            .prompt => ansi_color_yellow,
            .reasoning => ansi_color_gray,
            .user_input => ansi_bold ++ ansi_color_green,
            .err => ansi_bold ++ ansi_color_red,
        };
        _ = fprintf(out(), "%s", code);
        current_display = display;
        _ = fflush(out());
    }
}

/// Ports `getchar32` (console.cpp:200 @c1d0e7a00), the POSIX arm: a 32-bit
/// `wchar_t`, so no surrogate handling.
fn getchar32() u32 {
    const wc = getwchar();
    if (wc == weof) return weof;
    return wc;
}

/// Ports `pop_cursor` (console.cpp:265 @c1d0e7a00).
fn popCursor() void {
    _ = fputc('\x08', out());
}

/// Ports `estimateWidth` (console.cpp:286 @c1d0e7a00).
fn estimateWidth(codepoint: u32) c_int {
    return wcwidth(codepoint);
}

/// Ports `put_codepoint` (console.cpp:295 @c1d0e7a00), the POSIX arm.
fn putCodepoint(utf8: []const u8, expected_width: c_int) c_int {
    // We can trust expectedWidth if we've got one
    if (expected_width >= 0 or tty == null) {
        _ = fwrite(utf8.ptr, utf8.len, 1, out());
        return expected_width;
    }

    _ = fputs("\x1b[6n", tty); // Query cursor position
    var x1: c_int = 0;
    var y1: c_int = 0;
    var x2: c_int = 0;
    var y2: c_int = 0;
    var results: c_int = 0;
    results = fscanf(tty, "\x1b[%d;%dR", &y1, &x1);

    _ = fwrite(utf8.ptr, utf8.len, 1, tty);

    _ = fputs("\x1b[6n", tty); // Query cursor position
    results += fscanf(tty, "\x1b[%d;%dR", &y2, &x2);

    if (results != 4) return expected_width;

    var width = x2 - x1;
    if (width < 0) {
        // Calculate the width considering text wrapping
        var w: extern struct { ws_row: u16, ws_col: u16, ws_xpixel: u16, ws_ypixel: u16 } = undefined;
        const tiocgwinsz: c_ulong = if (builtin.os.tag.isDarwin()) 0x40087468 else 0x5413;
        _ = ioctl(std.posix.STDOUT_FILENO, tiocgwinsz, &w);
        width += w.ws_col;
    }
    return width;
}

/// Ports `replace_last` (console.cpp:356 @c1d0e7a00), the POSIX arm.
fn replaceLast(ch: u8) void {
    _ = fprintf(out(), "\x08%c", @as(c_int, ch));
}

/// Ports `decode_utf8` (console.cpp:365 @c1d0e7a00).
fn decodeUtf8(input: []const u8, pos: usize, advance: *usize) u32 {
    const ch: u32 = input[pos];
    if ((ch & 0x80) == 0) {
        advance.* = 1;
        return ch;
    }
    if ((ch & 0xE0) == 0xC0 and pos + 1 < input.len) {
        const c1: u32 = input[pos + 1];
        if ((c1 & 0xC0) != 0x80) {
            advance.* = 1;
            return 0xFFFD;
        }
        advance.* = 2;
        return ((ch & 0x1F) << 6) | (c1 & 0x3F);
    }
    if ((ch & 0xF0) == 0xE0 and pos + 2 < input.len) {
        const c1: u32 = input[pos + 1];
        const c2: u32 = input[pos + 2];
        if ((c1 & 0xC0) != 0x80 or (c2 & 0xC0) != 0x80) {
            advance.* = 1;
            return 0xFFFD;
        }
        advance.* = 3;
        return ((ch & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
    }
    if ((ch & 0xF8) == 0xF0 and pos + 3 < input.len) {
        const c1: u32 = input[pos + 1];
        const c2: u32 = input[pos + 2];
        const c3: u32 = input[pos + 3];
        if ((c1 & 0xC0) != 0x80 or (c2 & 0xC0) != 0x80 or (c3 & 0xC0) != 0x80) {
            advance.* = 1;
            return 0xFFFD;
        }
        advance.* = 4;
        return ((ch & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
    }

    advance.* = 1;
    return 0xFFFD; // replacement character for invalid input
}

/// Ports `append_utf8` (console.cpp:411 @c1d0e7a00).
fn appendUtf8(a: std.mem.Allocator, ch: u32, o: *std.ArrayList(u8)) !void {
    if (ch <= 0x7F) {
        try o.append(a, @intCast(ch));
    } else if (ch <= 0x7FF) {
        try o.appendSlice(a, &.{ @intCast(0xC0 | ((ch >> 6) & 0x1F)), @intCast(0x80 | (ch & 0x3F)) });
    } else if (ch <= 0xFFFF) {
        try o.appendSlice(a, &.{ @intCast(0xE0 | ((ch >> 12) & 0x0F)), @intCast(0x80 | ((ch >> 6) & 0x3F)), @intCast(0x80 | (ch & 0x3F)) });
    } else if (ch <= 0x10FFFF) {
        try o.appendSlice(a, &.{ @intCast(0xF0 | ((ch >> 18) & 0x07)), @intCast(0x80 | ((ch >> 12) & 0x3F)), @intCast(0x80 | ((ch >> 6) & 0x3F)), @intCast(0x80 | (ch & 0x3F)) });
    } else {
        // Invalid Unicode code point
    }
}

/// Ports `prev_utf8_char_pos` (console.cpp:432 @c1d0e7a00).
fn prevUtf8CharPos(line: []const u8, pos_in: usize) usize {
    if (pos_in == 0) return 0;
    var pos = pos_in - 1;
    while (pos > 0 and (line[pos] & 0xC0) == 0x80) pos -= 1;
    return pos;
}

/// Ports `next_utf8_char_pos` (console.cpp:441 @c1d0e7a00).
fn nextUtf8CharPos(line: []const u8, pos_in: usize) usize {
    if (pos_in >= line.len) return line.len;
    var pos = pos_in + 1;
    while (pos < line.len and (line[pos] & 0xC0) == 0x80) pos += 1;
    return pos;
}

/// The line `readline_advanced` edits: the text, each character's display
/// width, and the cursor as a character and a byte index.
const Editor = struct {
    a: std.mem.Allocator,
    line: *std.ArrayList(u8),
    widths: std.ArrayList(c_int) = .empty,
    char_pos: usize = 0, // current character index (one char can be multiple bytes)
    byte_pos: usize = 0, // current byte index

    /// Redraw from `p` for the characters from `from`, returning their width.
    fn redrawTail(self: *Editor, p_in: usize, from: usize) c_int {
        var p = p_in;
        var tail_width: c_int = 0;
        var i = from;
        while (i < self.widths.items.len) : (i += 1) {
            const next_p = nextUtf8CharPos(self.line.items, p);
            _ = putCodepoint(self.line.items[p..next_p], self.widths.items[i]);
            tail_width += self.widths.items[i];
            p = next_p;
        }
        return tail_width;
    }

    /// Ports `delete_at_cursor` (console.cpp:456 @c1d0e7a00).
    fn deleteAtCursor(self: *Editor) void {
        if (self.char_pos >= self.widths.items.len) return;

        const next_pos = nextUtf8CharPos(self.line.items, self.byte_pos);
        const w = self.widths.items[self.char_pos];
        const char_len = next_pos - self.byte_pos;

        self.line.replaceRangeAssumeCapacity(self.byte_pos, char_len, &.{});
        _ = self.widths.orderedRemove(self.char_pos);

        const tail_width = self.redrawTail(self.byte_pos, self.char_pos);

        var i: c_int = 0;
        while (i < w) : (i += 1) _ = fputc(' ', out());

        moveCursor(-(tail_width + w));
    }

    /// Ports `clear_current_line` (console.cpp:484 @c1d0e7a00).
    fn clearCurrentLine(self: *const Editor) void {
        var total_width: c_int = 0;
        for (self.widths.items) |w| total_width += if (w > 0) w else 1;

        if (total_width > 0) {
            var i: c_int = 0;
            while (i < total_width) : (i += 1) _ = fputc(' ', out());
            moveCursor(-total_width);
        }
    }

    /// Ports `set_line_contents` (console.cpp:497 @c1d0e7a00), with no
    /// cursor position: only history recall calls it here.
    fn setLineContents(self: *Editor, new_line: []const u8) !void {
        self.moveToLineStart();
        self.clearCurrentLine();

        const copy = try self.a.dupe(u8, new_line);
        defer self.a.free(copy);
        self.line.clearRetainingCapacity();
        try self.line.appendSlice(self.a, copy);
        self.widths.clearRetainingCapacity();
        self.byte_pos = 0;
        self.char_pos = 0;

        var idx: usize = 0;
        while (idx < self.line.items.len) {
            var advance: usize = 0;
            const cp = decodeUtf8(self.line.items, idx, &advance);
            const expected_width = estimateWidth(cp);
            var real_width = putCodepoint(self.line.items[idx .. idx + advance], expected_width);
            if (real_width < 0) real_width = 0;
            try self.widths.append(self.a, real_width);
            idx += advance;
            self.char_pos += 1;
            self.byte_pos = idx;
        }
    }

    /// Ports `move_to_line_start` (console.cpp:529 @c1d0e7a00).
    fn moveToLineStart(self: *Editor) void {
        var back_width: c_int = 0;
        for (self.widths.items[0..self.char_pos]) |w| back_width += w;
        moveCursor(-back_width);
        self.char_pos = 0;
        self.byte_pos = 0;
    }

    /// Ports `move_to_line_end` (console.cpp:539 @c1d0e7a00).
    fn moveToLineEnd(self: *Editor) void {
        var forward_width: c_int = 0;
        for (self.widths.items[self.char_pos..]) |w| forward_width += w;
        moveCursor(forward_width);
        self.char_pos = self.widths.items.len;
        self.byte_pos = self.line.items.len;
    }

    /// Ports `move_word_left` (console.cpp:581 @c1d0e7a00).
    fn moveWordLeft(self: *Editor) void {
        if (self.char_pos == 0) return;

        var new_char_pos = self.char_pos;
        var new_byte_pos = self.byte_pos;
        var move_width: c_int = 0;

        inline for (.{ false, true }) |stop_on_space| {
            while (new_char_pos > 0) {
                const prev_byte = prevUtf8CharPos(self.line.items, new_byte_pos);
                var advance: usize = 0;
                const cp = decodeUtf8(self.line.items, prev_byte, &advance);
                if (isSpaceCodepoint(cp) == stop_on_space) break;
                move_width += self.widths.items[new_char_pos - 1];
                new_char_pos -= 1;
                new_byte_pos = prev_byte;
            }
        }

        moveCursor(-move_width);
        self.char_pos = new_char_pos;
        self.byte_pos = new_byte_pos;
    }

    /// Ports `move_word_right` (console.cpp:619 @c1d0e7a00).
    fn moveWordRight(self: *Editor) void {
        if (self.char_pos >= self.widths.items.len) return;

        var new_char_pos = self.char_pos;
        var new_byte_pos = self.byte_pos;
        var move_width: c_int = 0;

        inline for (.{ false, true, false }) |stop_on_space| {
            while (new_char_pos < self.widths.items.len) {
                var advance: usize = 0;
                const cp = decodeUtf8(self.line.items, new_byte_pos, &advance);
                if (isSpaceCodepoint(cp) == stop_on_space) break;
                move_width += self.widths.items[new_char_pos];
                new_char_pos += 1;
                new_byte_pos += advance;
            }
        }

        moveCursor(move_width);
        self.char_pos = new_char_pos;
        self.byte_pos = new_byte_pos;
    }
};

/// Ports `has_ctrl_modifier` (console.cpp:549 @c1d0e7a00): a `5` among the
/// `;`-separated parameters.
fn hasCtrlModifier(params: []const u8) bool {
    var it = std.mem.splitScalar(u8, params, ';');
    while (it.next()) |field| {
        if (field.len == 0) continue;
        var value: c_int = 0;
        for (field) |ch| {
            if (!std.ascii.isDigit(ch)) {
                value = -1;
                break;
            }
            value = value *% 10 +% (ch - '0');
        }
        if (value == 5) return true;
    }
    return false;
}

/// Ports `is_space_codepoint` (console.cpp:577 @c1d0e7a00).
fn isSpaceCodepoint(cp: u32) bool {
    return iswspace(cp) != 0;
}

/// Ports `move_cursor` (console.cpp:666 @c1d0e7a00), the POSIX arm.
fn moveCursor(delta: c_int) void {
    if (delta == 0) return;
    if (delta < 0) {
        var i: c_int = 0;
        while (i < -delta) : (i += 1) _ = fprintf(out(), "\x08");
    } else {
        var i: c_int = 0;
        while (i < delta) : (i += 1) _ = fprintf(out(), "\x1b[C");
    }
}

/// Ports `history_t` (console.cpp:699 @c1d0e7a00): the lines read so far.
const History = struct {
    entries: std.ArrayList([]u8) = .empty,
    viewing_idx: usize = std.math.maxInt(usize),
    backup_line: std.ArrayList(u8) = .empty, // current line before viewing history

    fn add(self: *History, a: std.mem.Allocator, line: []const u8) !void {
        if (line.len == 0) return;
        // avoid duplicates with the last entry
        if (self.entries.items.len == 0 or !std.mem.eql(u8, self.entries.items[self.entries.items.len - 1], line)) {
            try self.entries.append(a, try a.dupe(u8, line));
        }
        // also clear viewing state
        self.endViewing();
    }

    fn prev(self: *History) ?[]const u8 {
        if (self.entries.items.len == 0) return null;
        if (self.viewing_idx == std.math.maxInt(usize)) return null;
        if (self.viewing_idx > 0) self.viewing_idx -= 1;
        return self.entries.items[self.viewing_idx];
    }

    fn next(self: *History) ?[]const u8 {
        if (self.entries.items.len == 0 or self.viewing_idx == std.math.maxInt(usize)) return null;
        self.viewing_idx += 1;
        if (self.viewing_idx >= self.entries.items.len) {
            return self.backup_line.items;
        }
        return self.entries.items[self.viewing_idx];
    }

    fn beginViewing(self: *History, a: std.mem.Allocator, line: []const u8) !void {
        self.backup_line.clearRetainingCapacity();
        try self.backup_line.appendSlice(a, line);
        self.viewing_idx = self.entries.items.len;
    }

    fn endViewing(self: *History) void {
        self.viewing_idx = std.math.maxInt(usize);
        self.backup_line.clearRetainingCapacity();
    }

    fn isViewing(self: *const History) bool {
        return self.viewing_idx != std.math.maxInt(usize);
    }
};

var history: History = .{};

/// Ports `readline_advanced` (console.cpp:753 @c1d0e7a00): a line edited
/// in place with the cursor keys, backspace and history.
fn readlineAdvanced(a: std.mem.Allocator, line: *std.ArrayList(u8), multiline_input: bool) !bool {
    if (out() != io.cStdout()) _ = fflush(io.cStdout());

    line.clearRetainingCapacity();
    var ed: Editor = .{ .a = a, .line = line };
    defer ed.widths.deinit(a);
    var is_special_char = false;
    var end_of_stream = false;

    while (true) {
        _ = fflush(out()); // Ensure all output is displayed before waiting for input
        const input_char = getchar32();

        if (input_char == '\r' or input_char == '\n') break;

        if (input_char == weof or input_char == 0x04) { // Ctrl+D
            end_of_stream = true;
            break;
        }

        if (is_special_char) {
            replaceLast(line.items[line.items.len - 1]);
            is_special_char = false;
        }

        if (input_char == '\x1b') { // Escape sequence
            var code = getchar32();
            if (code == '[') {
                var params: std.ArrayList(u8) = .empty;
                defer params.deinit(a);
                while (true) {
                    code = getchar32();
                    if ((code >= 'A' and code <= 'Z') or (code >= 'a' and code <= 'z') or code == '~' or code == weof) break;
                    try params.append(a, @truncate(code));
                }

                const ctrl_modifier = hasCtrlModifier(params.items);

                if (code == 'D') { // left
                    if (ctrl_modifier) {
                        ed.moveWordLeft();
                    } else if (ed.char_pos > 0) {
                        const w = ed.widths.items[ed.char_pos - 1];
                        moveCursor(-w);
                        ed.char_pos -= 1;
                        ed.byte_pos = prevUtf8CharPos(line.items, ed.byte_pos);
                    }
                } else if (code == 'C') { // right
                    if (ctrl_modifier) {
                        ed.moveWordRight();
                    } else if (ed.char_pos < ed.widths.items.len) {
                        const w = ed.widths.items[ed.char_pos];
                        moveCursor(w);
                        ed.char_pos += 1;
                        ed.byte_pos = nextUtf8CharPos(line.items, ed.byte_pos);
                    }
                } else if (code == 'H') { // home
                    ed.moveToLineStart();
                } else if (code == 'F') { // end
                    ed.moveToLineEnd();
                } else if (code == 'A' or code == 'B') {
                    // up/down
                    if (code == 'A') {
                        if (!history.isViewing()) try history.beginViewing(a, line.items);
                        if (history.prev()) |nl| try ed.setLineContents(nl);
                    } else {
                        if (history.isViewing()) {
                            if (history.next()) |nl| {
                                const copy = try a.dupe(u8, nl);
                                defer a.free(copy);
                                if (history.viewing_idx >= history.entries.items.len) history.endViewing();
                                try ed.setLineContents(copy);
                            }
                        }
                    }
                    is_special_char = false;
                } else if ((code == '~' or (code >= 'A' and code <= 'Z') or (code >= 'a' and code <= 'z')) and params.items.len != 0) {
                    var digits: std.ArrayList(u8) = .empty;
                    defer digits.deinit(a);
                    for (params.items) |ch| {
                        if (ch == ';') break;
                        if (std.ascii.isDigit(ch)) try digits.append(a, ch);
                    }

                    if (code == '~') {
                        const d = digits.items;
                        if (std.mem.eql(u8, d, "1") or std.mem.eql(u8, d, "7")) { // home
                            ed.moveToLineStart();
                        } else if (std.mem.eql(u8, d, "4") or std.mem.eql(u8, d, "8")) { // end
                            ed.moveToLineEnd();
                        } else if (std.mem.eql(u8, d, "3")) { // delete
                            ed.deleteAtCursor();
                        }
                    }
                }
            } else if (code == 0x1B) {
                // Discard the rest of the escape sequence
                while (true) {
                    code = getchar32();
                    if (code == weof) break;
                    if ((code >= 'A' and code <= 'Z') or (code >= 'a' and code <= 'z') or code == '~') break;
                }
            }
        } else if (input_char == 0x08 or input_char == 0x7F) { // Backspace
            if (ed.char_pos > 0) {
                const w = ed.widths.items[ed.char_pos - 1];
                moveCursor(-w);
                ed.char_pos -= 1;
                const prev_pos = prevUtf8CharPos(line.items, ed.byte_pos);
                const char_len = ed.byte_pos - prev_pos;
                ed.byte_pos = prev_pos;

                // remove the character
                line.replaceRangeAssumeCapacity(ed.byte_pos, char_len, &.{});
                _ = ed.widths.orderedRemove(ed.char_pos);

                // redraw tail
                const tail_width = ed.redrawTail(ed.byte_pos, ed.char_pos);

                // clear display
                var i: c_int = 0;
                while (i < w) : (i += 1) _ = fputc(' ', out());
                moveCursor(-(tail_width + w));
            }
        } else {
            // insert character
            var new_char_str: std.ArrayList(u8) = .empty;
            defer new_char_str.deinit(a);
            try appendUtf8(a, input_char, &new_char_str);
            const w = estimateWidth(input_char);

            if (ed.char_pos == ed.widths.items.len) {
                // insert at the end
                try line.appendSlice(a, new_char_str.items);
                var real_w = putCodepoint(new_char_str.items, w);
                if (real_w < 0) real_w = 0;
                try ed.widths.append(a, real_w);
                ed.byte_pos += new_char_str.items.len;
                ed.char_pos += 1;
            } else {
                // insert in middle
                try line.insertSlice(a, ed.byte_pos, new_char_str.items);

                var real_w = putCodepoint(new_char_str.items, w);
                if (real_w < 0) real_w = 0;

                try ed.widths.insert(a, ed.char_pos, real_w);

                // print the tail
                const tail_width = ed.redrawTail(ed.byte_pos + new_char_str.items.len, ed.char_pos + 1);

                moveCursor(-tail_width);

                ed.byte_pos += new_char_str.items.len;
                ed.char_pos += 1;
            }
        }

        if (line.items.len != 0 and (line.items[line.items.len - 1] == '\\' or line.items[line.items.len - 1] == '/')) {
            replaceLast(line.items[line.items.len - 1]);
            is_special_char = true;
        }
    }

    var has_more = multiline_input;
    if (is_special_char) {
        replaceLast(' ');
        popCursor();

        const last = line.pop().?;
        if (last == '\\') {
            try line.append(a, '\n');
            _ = fputc('\n', out());
            has_more = !has_more;
        } else {
            // llama will just eat the single space, it won't act as a space
            if (line.items.len == 1 and line.items[0] == ' ') {
                line.clearRetainingCapacity();
                popCursor();
            }
            has_more = false;
        }
    } else {
        if (end_of_stream) {
            has_more = false;
        } else {
            try line.append(a, '\n');
            _ = fputc('\n', out());
        }
    }

    if (!end_of_stream and line.items.len != 0) {
        // remove the trailing newline for history storage
        var hline: []const u8 = line.items;
        if (hline[hline.len - 1] == '\n') hline = hline[0 .. hline.len - 1];
        try history.add(std.heap.c_allocator, hline);
    }

    _ = fflush(out());
    return has_more;
}

/// Ports `readline_simple` (console.cpp:1046 @c1d0e7a00), the POSIX arm:
/// `std::getline(std::cin, line)` over the C library's `stdin`, which
/// libc++'s `cin` shares.
fn readlineSimple(a: std.mem.Allocator, line: *std.ArrayList(u8), multiline_input_in: bool) !bool {
    var multiline_input = multiline_input_in;
    line.clearRetainingCapacity();
    var buf: ?[*]u8 = null;
    var cap: usize = 0;
    defer free(buf);
    const n = getline(&buf, &cap, io.cStdin());
    if (n < 0) {
        // Input stream is bad or EOF received
        return false;
    }
    var got = buf.?[0..@intCast(n)];
    if (got.len != 0 and got[got.len - 1] == '\n') got = got[0 .. got.len - 1];
    try line.appendSlice(a, got);
    if (line.items.len != 0) {
        const last = line.items[line.items.len - 1];
        if (last == '/') { // Always return control on '/' symbol
            _ = line.pop();
            return false;
        }
        if (last == '\\') { // '\\' changes the default action
            _ = line.pop();
            multiline_input = !multiline_input;
        }
    }
    try line.append(a, '\n');

    // By default, continue input if multiline_input is set
    return multiline_input;
}

/// Ports `readline` (console.cpp:1083 @c1d0e7a00).
///
/// Return: whether more input follows (a multi-line continuation).
pub fn readline(a: std.mem.Allocator, line: *std.ArrayList(u8), multiline_input: bool) !bool {
    if (simple_io) return readlineSimple(a, line, multiline_input);
    return readlineAdvanced(a, line, multiline_input);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "utf8 helpers round-trip and step over continuation bytes" {
    const a = std.testing.allocator;
    var s: std.ArrayList(u8) = .empty;
    defer s.deinit(a);
    for ([_]u32{ 'a', 0xE9, 0x20AC, 0x1F600 }) |cp| try appendUtf8(a, cp, &s);
    var pos: usize = 0;
    var adv: usize = 0;
    for ([_]u32{ 'a', 0xE9, 0x20AC, 0x1F600 }) |cp| {
        try std.testing.expectEqual(cp, decodeUtf8(s.items, pos, &adv));
        try std.testing.expectEqual(pos + adv, nextUtf8CharPos(s.items, pos));
        pos += adv;
    }
    try std.testing.expectEqual(@as(usize, 6), prevUtf8CharPos(s.items, s.items.len));
}

test "ctrl modifier parameter" {
    try std.testing.expect(hasCtrlModifier("1;5"));
    try std.testing.expect(!hasCtrlModifier("1;2"));
    try std.testing.expect(!hasCtrlModifier("5x"));
}
